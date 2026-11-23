"""Self-taught driver (evolution strategies): network, episodes, training checkpoints."""
import numpy as np
import pytest

from carsim import learn
from carsim.learn import N_OBS, EpisodeConfig, LearnedDriver, Policy, TrainConfig, run_episode, train


def test_policy_outputs_are_bounded():
    pol = Policy(N_OBS).init_random(np.random.default_rng(0))
    out = pol.act(np.random.default_rng(1).standard_normal(N_OBS) * 5)
    assert out.shape == (2,) and np.all(np.abs(out) <= 1.0)
    assert pol.size == len(pol.theta)


def test_episode_ends_and_scores_progress():
    pol = Policy(N_OBS).init_random(np.random.default_rng(0))
    fit, info = run_episode(pol.theta, EpisodeConfig(track="short", max_time=5.0))
    assert np.isfinite(fit) and info["time"] <= 5.0 + 0.05
    assert info["reason"] in ("time", "off track", "stuck", "lap")


def test_random_start_places_the_car_on_track():
    tr = learn.TRACKS["short"]()
    zero = np.zeros(Policy(N_OBS).size)  # outputs 0: no steer, no pedal -> coasts straight on
    fit, info, rows = run_episode(zero, EpisodeConfig(track="short", max_time=1.0, start_s=500.0,
                                                      start_speed=20.0), record=True)
    x, y = rows[0][1], rows[0][2]
    i = tr.index_at(500.0)
    assert np.hypot(x - tr.x[i], y - tr.y[i]) < 25.0
    assert info["progress"] > 10.0


def test_training_saves_checkpoints_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    cfg = TrainConfig(track="short", population=4, starts=1, start_time=2.0,
                      episode=EpisodeConfig(track="short", max_time=3.0))
    train(cfg, workers=1, generations=2, log=lambda *a: None)
    d = np.load(tmp_path / "short" / "checkpoint.npz")
    assert int(d["gen"]) == 2
    assert (tmp_path / "short" / "policy.npy").exists()
    lines = (tmp_path / "short" / "progress.csv").read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 generations
    train(cfg, workers=1, generations=3, log=lambda *a: None)  # resumes: one more generation
    assert int(np.load(tmp_path / "short" / "checkpoint.npz")["gen"]) == 3
    drv = LearnedDriver(learn.TRACKS["short"](), "short")
    assert drv.policy.theta.shape == (Policy(N_OBS).size,)


def test_server_drives_with_the_learned_policy(tmp_path, monkeypatch):
    from carsim.server import Session
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    (tmp_path / "short").mkdir()
    np.save(tmp_path / "short" / "policy.npy", Policy(N_OBS).init_random(np.random.default_rng(3)).theta)
    sess = Session(track="short", fuel=5.0, learned=True)
    sess.step(2000)
    msg = sess.state_message()
    assert sess.track_message()["learned"] is True
    assert msg["t"] == pytest.approx(2.0, abs=1e-6)


def test_distance_cap_ends_the_episode():
    zero = np.zeros(Policy(N_OBS).size)  # coasts straight on from the line
    fit, info = run_episode(zero, EpisodeConfig(track="short", max_time=30.0, start_speed=20.0,
                                                max_distance=60.0))
    assert info["reason"] == "distance"
    assert 60.0 <= info["progress"] < 70.0


def test_crash_costs_a_penalty_and_surviving_beats_crashing():
    pol = Policy(N_OBS).init_random(np.random.default_rng(0))  # random net: leaves the track quickly
    base = EpisodeConfig(track="short", max_time=10.0)
    f0, i0 = run_episode(pol.theta, base)
    assert i0["reason"] in ("off track", "stuck")
    f1, _ = run_episode(pol.theta, EpisodeConfig(track="short", max_time=10.0, crash_penalty=250.0))
    assert f1 == pytest.approx(f0 - 250.0)
    # a slow run that survives the whole distance outscores a faster run that crashes before it
    zero = np.zeros(Policy(N_OBS).size)
    fs, info = run_episode(zero, EpisodeConfig(track="short", max_time=30.0, start_speed=15.0,
                                               max_distance=50.0, crash_penalty=250.0))
    assert info["reason"] == "distance" and fs > f1


def test_time_bonus_only_when_the_distance_is_reached():
    zero = np.zeros(Policy(N_OBS).size)
    kw = dict(track="short", max_time=30.0, start_speed=20.0, max_distance=60.0)
    f_plain, i = run_episode(zero, EpisodeConfig(**kw))
    f_bonus, _ = run_episode(zero, EpisodeConfig(**kw, time_bonus=10.0, ref_speed=10.0))
    assert f_bonus == pytest.approx(f_plain + 10.0 * max(0.0, 60.0 / 10.0 - i["time"]))
    assert f_bonus > f_plain


# ---- phase 13: realistic training car, 3 laps, migration, several tracks ---------------------
def test_training_car_is_the_3d_car():
    from carsim.learn import make_car
    tr = learn.TRACKS["short"]()
    car = make_car(tr, EpisodeConfig(track="short"), i=500)
    assert car.cond is not None and car.p.tyre_thermals and car.fuel == pytest.approx(40.0)
    assert car.cond.i == 500  # surface lookups start where the car is
    assert np.any(car.cond.line_offset != 0)  # rubber lies on the racing line
    assert car.p.chassis.actuator_limits and car.p.brakes.thermal


def test_migrated_weights_drive_exactly_like_the_old_network():
    from carsim.learn import OBS_NAMES, OBS_V1, migrate_theta
    rng = np.random.default_rng(4)
    old = Policy(len(OBS_V1)).init_random(rng)
    new_theta = migrate_theta(old.theta, OBS_V1)
    new = Policy(len(OBS_NAMES), theta=new_theta)
    obs_old = rng.standard_normal(len(OBS_V1))
    obs_new = np.concatenate([obs_old, rng.standard_normal(len(OBS_NAMES) - len(OBS_V1)) * 3])
    assert np.allclose(old.act(obs_old), new.act(obs_new))


def test_old_run_is_archived_and_training_starts_fresh(tmp_path, monkeypatch):
    from carsim.learn import OBS_V1
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    d = tmp_path / "short"
    d.mkdir()
    old = Policy(len(OBS_V1)).init_random(np.random.default_rng(5)).theta
    np.savez(d / "checkpoint.npz", theta=old, m=old * 0, v=old * 0, t=7, gen=7, best_theta=old, best_fitness=4000.0)
    np.save(d / "policy.npy", old)
    (d / "progress.csv").write_text("generation,minutes\n7,1.0\n")
    cfg = TrainConfig(track="short", population=2, starts=1, laps=1,
                      episode=EpisodeConfig(track="short", max_time=1.0, laps=1))
    train(cfg, workers=1, generations=1, log=lambda *a: None)
    assert (d / "legacy_1" / "policy.npy").exists() and (d / "legacy_1" / "progress.csv").exists()
    assert int(np.load(d / "checkpoint.npz")["gen"]) == 1
    assert np.load(d / "policy.npy").size == Policy(N_OBS).size
    # carry-over keeps the old weights
    for f in d.glob("*.*"):
        f.unlink()
    np.savez(d / "checkpoint.npz", theta=old, m=old * 0, v=old * 0, t=7, gen=7, best_theta=old, best_fitness=4000.0)
    np.save(d / "policy.npy", old)
    msgs = []
    train(cfg, workers=1, generations=0, log=msgs.append, carry_over=True)
    assert (d / "legacy_2").exists() and any("carrying over" in m for m in msgs)
    # load_policy also understands a legacy policy file (for watching an old driver)
    np.save(d / "policy.npy", old)
    (d / "config.json").unlink(missing_ok=True)
    assert learn.load_policy("short").size == Policy(N_OBS).size


def test_three_lap_score_counts_every_lap_and_rewards_speed():
    from carsim.learn import lap_score
    L = 1100.0
    assert lap_score([30.0], 3, L) == pytest.approx(1000.0)
    assert lap_score([30.0, 28.0], 3, L) == pytest.approx(2000.0)
    fast, slow = lap_score([30.0, 28.0, 28.0], 3, L), lap_score([30.0, 29.0, 29.0], 3, L)
    assert fast > slow > 3000.0


def test_practice_runs_cover_the_laps_and_vary_the_car():
    from carsim.learn import practice_runs
    cfg = TrainConfig(track="short", starts=6, laps=3)
    runs = practice_runs(cfg, np.random.default_rng(0), gen=0)
    L = learn.TRACKS["short"]().length
    assert runs[0].start_s == 0.0 and runs[0].max_distance == pytest.approx(3 * L)
    temps = {r.tyre_temp for r in runs[1:]}
    assert len(temps) == 5 and all(60.0 <= t <= 115.0 for t in temps)


def test_multi_track_training_tests_every_track(tmp_path, monkeypatch):
    from carsim.learn import practice_runs
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    cfg = TrainConfig(track="short", tracks=("short", "test"), population=2, starts=8, laps=1,
                      episode=EpisodeConfig(max_time=1.0, laps=1))
    assert cfg.run_name() == "multi_short-test"
    rng = np.random.default_rng(1)
    assert practice_runs(cfg, rng, 0)[0].track == "short" and practice_runs(cfg, rng, 1)[0].track == "test"
    assert {r.track for r in practice_runs(cfg, rng, 0)[1:]} == {"short", "test"}
    msgs = []
    cfg.starts = 1
    train(cfg, workers=1, generations=1, log=msgs.append)
    assert (tmp_path / "multi_short-test" / "policy.npy").exists()
    assert "short:" in msgs[-1] and "test:" in msgs[-1]


# ---- harder practice for Monza ------------------------------------------------------------
def test_power_circuits_and_fast_corner_starts_in_practice():
    from carsim.learn import braking_zones, practice_runs
    cfg = TrainConfig(track="random", starts=9, tracks_per_gen=4, power_share=1.0, fast_corner_share=1.0)
    runs = practice_runs(cfg, np.random.default_rng(2), 0)
    assert all(r.track.startswith("power:") for r in runs)
    fast = [r for r in runs[1:] if r.start_speed >= cfg.fast_corner_speed[0]]
    assert len(fast) == 8
    assert braking_zones(learn.TRACKS["monza"]())  # the straight into the first chicane


def test_steer_penalty_costs_fitness_per_radian():
    pol = Policy(N_OBS).init_random(np.random.default_rng(0))
    kw = dict(track="short", max_time=3.0)
    f0, info = run_episode(pol.theta, EpisodeConfig(**kw))
    f1, _ = run_episode(pol.theta, EpisodeConfig(**kw, steer_penalty=5.0))
    assert info["steer_travel"] > 0 and f1 == pytest.approx(f0 - 5.0 * info["steer_travel"])


def test_named_run_goes_to_its_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    cfg = TrainConfig(track="short", name="exp1", population=2, starts=1, laps=1,
                      episode=EpisodeConfig(track="short", max_time=1.0, laps=1))
    train(cfg, workers=1, generations=1, log=lambda *a: None)
    assert (tmp_path / "exp1" / "policy.npy").exists() and not (tmp_path / "short").exists()
