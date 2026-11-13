"""Varying track width, random circuits, Monza, the 2025-spec car and multi-track learning."""
import math

import numpy as np
import pytest

from carsim import f1_2025
from carsim.circuits import TRACKS, get_track, monza, track_arg
from carsim.params import F1_2025
from carsim.racing_line import min_curvature_offsets
from carsim.track import Track
from carsim.track import test_circuit as make_circuit
from carsim.trackgen import TrackGenConfig, _clear_of_itself, random_track, track_stats


# ---- varying width -------------------------------------------------------------
def test_constant_width_track_behaves_as_before():
    tr = make_circuit()
    assert np.allclose(tr.width_left, 6.0) and np.allclose(tr.width_right, 6.0)
    i = 100
    assert tr.normalised_offset(i, 3.0) == pytest.approx(0.5)
    assert tr.surface_at(5.0, i) == tr.surface_at(5.0) == "track"
    assert tr.surface_at(7.0, i) == "kerb" and tr.surface_at(-9.0, i) == "grass"


def test_asymmetric_width_edges_and_surfaces():
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    tr = Track.from_points("ring", 100 * np.cos(th), 100 * np.sin(th), width_left=8.0, width_right=4.0)
    i = 50
    assert tr.edges(i) == pytest.approx((8.0, 4.0))
    assert tr.width == pytest.approx(12.0)
    assert tr.surface_at(7.5, i) == "track" and tr.surface_at(-4.5, i) == "kerb"
    assert tr.off_track(i, -6.0) and not tr.off_track(i, 7.9)
    # The middle of the tarmac is 2 m left of the centreline.
    assert tr.normalised_offset(i, 2.0) == pytest.approx(0.0)
    assert tr.normalised_offset(i, 8.0) == pytest.approx(1.0)
    assert tr.normalised_offset(i, -4.0) == pytest.approx(-1.0)


def test_width_columns_load_from_csv(tmp_path):
    th = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    path = tmp_path / "ring.csv"
    wl = 5 + np.sin(th)
    np.savetxt(path, np.c_[80 * np.cos(th), 80 * np.sin(th), wl, np.full_like(th, 6.0)],
               delimiter=",", header="x,y,width_left,width_right", comments="")
    tr = Track.from_csv(str(path), "ring")
    assert tr.width_left.min() == pytest.approx(4.0, abs=0.05)
    assert tr.width_left.max() == pytest.approx(6.0, abs=0.05)
    assert np.allclose(tr.width_right, 6.0)


def test_elevation_keeps_the_widths():
    tr = random_track(3)
    hilly = tr.with_elevation(lambda s: 5 * math.sin(2 * math.pi * s / tr.length))
    assert np.array_equal(hilly.width_left, tr.width_left)


def test_racing_line_stays_inside_each_edge():
    tr = random_track(4)
    n = min_curvature_offsets(tr, margin=1.5)
    assert np.all(n <= tr.width_left - 1.5 + 1e-6)
    assert np.all(n >= -(tr.width_right - 1.5) - 1e-6)


# ---- random circuits -------------------------------------------------------------
def test_random_track_is_the_same_for_the_same_seed():
    a, b, c = random_track(11), random_track(11), random_track(12)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.width_left, b.width_left)
    assert a.length != c.length


@pytest.mark.parametrize("seed", range(0, 40, 3))
def test_random_tracks_are_valid_circuits(seed):
    cfg = TrackGenConfig()
    tr = random_track(seed, cfg)
    st = track_stats(tr)
    # Closes on itself: the last sample is about one step from the first.
    assert math.hypot(tr.x[-1] - tr.x[0], tr.y[-1] - tr.y[0]) < 1.5
    assert st["min_radius_m"] >= cfg.r_min * 0.95
    assert 0.75 * cfg.length[0] < tr.length < 1.15 * cfg.length[1]
    assert st["width_min_m"] >= 8.0
    assert _clear_of_itself(tr, cfg.runoff)
    # The lap starts on the longest straight.
    assert np.all(np.abs(tr.curvature[:20]) < 1e-9)


def test_random_tracks_vary():
    stats = [track_stats(random_track(s)) for s in range(15)]
    assert {s["direction"] for s in stats} == {"clockwise", "anticlockwise"}
    assert np.ptp([s["length_m"] for s in stats]) > 1500
    assert min(s["min_radius_m"] for s in stats) < 25  # some hairpins
    assert max(s["longest_straight_m"] for s in stats) > 500


def test_random_track_with_elevation():
    tr = random_track(5, elevation=12.0)
    assert tr.z.max() - tr.z.min() == pytest.approx(12.0, rel=0.05)
    assert abs(tr.z[0] - tr.z[-1]) < 0.5


def test_track_names():
    assert track_arg("monza") == "monza" and track_arg("random:7") == "random:7"
    assert "random:7" in TRACKS and "nowhere" not in TRACKS
    assert get_track("random:7").length == random_track(7).length
    with pytest.raises(Exception):
        track_arg("nowhere")


# ---- Monza ---------------------------------------------------------------------------
def test_monza_matches_the_real_circuit():
    tr = monza()
    assert tr.length == pytest.approx(5793, abs=20)  # official length 5.793 km
    st = track_stats(tr)
    assert st["direction"] == "clockwise"
    assert 8.5 < tr.width < 11  # between the white lines, from satellite images
    assert st["longest_straight_m"] > 800  # the main straight / run to the first chicane
    assert 8 < st["min_radius_m"] < 20  # the chicanes
    assert tr.z.max() == tr.z.min() == 0.0


# ---- the 2025 car ---------------------------------------------------------------------
def test_f1_2025_car_is_built_to_the_rules():
    p = f1_2025(0.0)
    c = p.chassis
    assert c.mass == F1_2025["min_mass_kg"] == 800.0
    assert c.wheelbase <= F1_2025["max_wheelbase_m"]
    assert c.track_front + F1_2025["tyre_width_m"][0] <= F1_2025["max_width_m"] + 1e-9
    assert c.track_rear + F1_2025["tyre_width_m"][1] <= F1_2025["max_width_m"] + 1e-9
    assert 2 * c.wheel_radius == pytest.approx(0.7225)
    assert f1_2025(0.0, downforce="low").aero.CdA < p.aero.CdA < f1_2025(0.0, downforce="high").aero.CdA


# ---- learning on many circuits ----------------------------------------------------------
def test_old_networks_still_load_and_upgrade_exactly():
    from carsim import learn
    from carsim.learn import N_OBS_V1, N_OBS_V2, Policy, make_car, obs_version_of, observe, upgrade_theta
    old = Policy(N_OBS_V1).init_random(np.random.default_rng(2)).theta
    assert obs_version_of(old) == 1
    new = upgrade_theta(old)
    assert obs_version_of(new) == 2
    tr = get_track("random:2")
    car = make_car("f1_2025")
    car.set_speed(30.0)
    a = Policy(N_OBS_V1, theta=old).act(observe(car, tr, 40, 1))
    b = Policy(N_OBS_V2, theta=new).act(observe(car, tr, 40, 2))
    assert np.allclose(a, b)


def test_episodes_on_monza_and_random_circuits():
    from carsim.learn import N_OBS, EpisodeConfig, Policy, run_episode
    pol = Policy(N_OBS).init_random(np.random.default_rng(0))
    for name in ("monza", "random:9"):
        fit, info = run_episode(pol.theta, EpisodeConfig(track=name, max_time=3.0))
        assert np.isfinite(fit) and info["reason"] in ("time", "off track", "stuck")


def test_training_on_random_circuits(tmp_path, monkeypatch):
    from carsim import learn
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    cfg = learn.TrainConfig(track="random", population=4, starts=2, start_distance=150.0,
                            validation=("random:0",), test_track="short", test_every=1)
    learn.train(cfg, workers=1, generations=1, log=lambda *a: None)
    rows = (tmp_path / "random" / "progress.csv").read_text().strip().splitlines()
    assert rows[0].startswith("generation,practice_mean") and "val_lap_fraction" in rows[0]
    assert len(rows) == 2
    drv = learn.LearnedDriver(get_track("monza"), "random")
    assert drv.version == 2


def test_new_run_can_start_from_an_old_driver(tmp_path, monkeypatch):
    from carsim import learn
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    (tmp_path / "short").mkdir()
    old = learn.Policy(learn.N_OBS_V1).init_random(np.random.default_rng(5)).theta
    np.save(tmp_path / "short" / "policy.npy", old)
    cfg = learn.TrainConfig(track="random", population=2, starts=2, start_distance=100.0, lr=0.0,
                            weight_decay=0.0, validation=("random:0",), test_track="", init_from="short")
    learn.train(cfg, workers=1, generations=1, log=lambda *a: None)
    d = np.load(tmp_path / "random" / "checkpoint.npz")
    assert np.allclose(d["theta"], learn.upgrade_theta(old))


def test_server_sends_track_edges():
    from carsim.server import Session
    sess = Session(track="random:1", fuel=5.0)
    msg = sess.track_message()
    assert len(msg["width_left"]) == len(msg["x"]) == len(msg["width_right"])
    assert msg["wheelbase"] == 3.6
