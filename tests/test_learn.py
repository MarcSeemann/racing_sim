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
