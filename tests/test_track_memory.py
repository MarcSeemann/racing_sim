"""Looking ~5 s ahead in coarsening windows, knowing the circuit's corners, and free practice."""
import numpy as np
import pytest

from carsim import learn, track_memory as tm
from carsim.circuits import get_track
from carsim.track import Track


def test_horizon_is_five_seconds_within_limits():
    assert tm.horizon(80.0) == pytest.approx(400.0)
    assert tm.horizon(2.0) == tm.HORIZON_MIN_M and tm.horizon(200.0) == tm.HORIZON_MAX_M


def test_monza_corners_are_where_they_are():
    tr = get_track("monza")
    cs = tm.corners(tr)
    starts = [c["start"] * tr.length / len(tr.s) for c in cs]
    assert any(900 < s < 940 for s in starts)  # Rettifilo, the first chicane
    first = min(cs, key=lambda c: abs(c["start"] - tr.index_at(920)))
    assert 1 / abs(first["peak"]) < 12.0  # R ~9 m
    assert 6 <= len(cs) <= 10  # chicanes are one corner each, not one per bend


def test_a_window_does_not_miss_a_corner_between_two_point_samples():
    # a straight with one short, tight kink 200 m ahead: between curv_170 and curv_230
    kink = [("arc", 15.0, 20.0), ("arc", 15.0, -20.0)]  # an S: same heading after, shifted sideways
    segs = ([("straight", 200.0)] + kink + [("straight", 300.0), ("arc", 60.0, 180.0), ("straight", 200.0)]
            + kink + [("straight", 300.0), ("arc", 60.0, 180.0)])  # the second S undoes the first
    tr = Track.from_segments("kink", segs)
    i = tr.index_at(0.0)
    points = [abs(tr.curvature[tr.index_at(d)]) for d in learn.LOOKAHEAD_M]
    windows = np.abs(tm.lookahead(tr, i, 80.0)[0::2])
    assert max(windows) > 1 / 20 * tm.CURV_SCALE  # the windows see R 15 m
    assert max(points) < 1 / 20 or max(windows) >= max(points) * tm.CURV_SCALE


def test_memory_counts_down_to_the_next_corner():
    tr = get_track("monza")
    d = [tm.memory(tr, tr.index_at(s))[0] * 500 for s in (300, 600, 900)]
    assert d[0] > d[1] > d[2] >= 0
    assert d[0] == pytest.approx(919 - 300, abs=15)
    after = tm.memory(tr, tr.index_at(1000))  # past the chicane: next is Roggia, ~1.1 km on
    assert after[0] * 500 == pytest.approx(2131 - 1000, abs=30)


def test_new_inputs_start_with_zero_weight():
    rng = np.random.default_rng(0)
    old = learn.Policy(len(learn.OBS_V3)).init_random(rng)
    new = learn.Policy(learn.N_OBS, theta=learn.migrate_theta(old.theta, learn.OBS_V3))
    for _ in range(50):
        o = rng.standard_normal(learn.N_OBS) * 3
        # same driver up to the order of floating-point additions
        assert np.allclose(old.act(o[:len(learn.OBS_V3)]), new.act(o), atol=1e-12)
    assert learn.adapt_theta(old.theta)[0].size == new.size  # size alone tells the old layout


def test_observation_is_finite_on_every_kind_of_circuit():
    pol = learn.Policy(learn.N_OBS).init_random(np.random.default_rng(1))
    for name in ("short", "monza", "power:3", "random:5"):
        fit, info, rows = learn.run_episode(pol.theta, learn.EpisodeConfig(track=name, max_time=2.0), record=True)
        assert np.isfinite(fit)
        tr = get_track(name)
        car = learn.make_car(tr, learn.EpisodeConfig(track=name), 0)
        o = learn.observe(car, tr, 0)
        assert o.shape == (learn.N_OBS,) and np.all(np.isfinite(o))


def test_practice_session_does_not_touch_saved_drivers(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    theta = learn.Policy(learn.N_OBS).init_random(np.random.default_rng(2)).theta
    r = learn.practice_session(theta, "short", generations=1, population=2, workers=1, laps=1, starts=1,
                               log=lambda *a: None)
    assert set(r) >= {"before", "after", "history", "theta"} and len(r["history"]) == 1
    assert not np.array_equal(r["theta"], theta)  # it learned something...
    assert not any(tmp_path.iterdir())  # ...and saved nothing
