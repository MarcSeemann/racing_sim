"""Run-off, barriers and track limits: leaving the track is allowed, but it costs."""
import numpy as np
import pytest

from carsim import learn
from carsim.circuits import get_track
from carsim.learn import EpisodeConfig, Policy, run_episode
from carsim.runoff import STREET, layout
from carsim.track import Track


def test_street_circuits_have_walls_and_permanent_ones_have_runoff():
    monaco, monza = get_track("f1:monaco"), get_track("f1:monza")
    assert "f1:monaco" in STREET and np.all(monaco.runoff_left <= 2.0)
    i = monaco.index_at(500.0)
    w = monaco._wl[i] + monaco.kerb_width
    assert monaco.surface_at(w + 1.0, i) == "asphalt" and monaco.surface_at(w + 2.0, i) == "barrier"
    kinds = set(monza.runoff_kind_left) | set(monza.runoff_kind_right)
    assert {"grass", "gravel", "asphalt"} <= kinds


def test_the_outside_of_a_slow_corner_is_asphalt():
    # a hairpin to the left: the car runs wide to the RIGHT
    tr = Track.from_segments("hairpin", [("straight", 400.0), ("arc", 30.0, 180.0), ("straight", 400.0),
                                         ("arc", 30.0, 180.0)])
    rl, rr, kl, kr = layout(tr)
    apex = tr.index_at(400.0 + np.pi * 30 / 2)
    assert kr[apex] == "asphalt" and rr[apex] >= 20.0 and rl[apex] < rr[apex]
    mid_straight = tr.index_at(200.0)
    assert kl[mid_straight] == kr[mid_straight] == "grass"


def _drift_left(track, start_s=300.0):
    th = np.zeros(Policy(learn.N_OBS).size)
    out = len(th) - learn.N_ACT  # the output biases, in ACTION_NAMES order
    th[out + learn.ACTION_NAMES.index("steer")] = 0.05  # a little left steering
    th[out + learn.ACTION_NAMES.index("pedal")] = 0.3  # gentle throttle
    return run_episode(th, EpisodeConfig(track="x", max_time=30.0, start_s=start_s, start_speed=20.0), track)


def test_off_track_too_long_is_disqualified_and_a_wall_is_a_crash():
    import dataclasses
    base = get_track("monza")
    n = len(base.s)
    # a huge field of grass: it never reaches a barrier, so it is the 3 s rule that ends the run
    wide = dataclasses.replace(base, runoff_left=np.full(n, 500.0), runoff_right=np.full(n, 500.0),
                               runoff_kind_left=["grass"] * n, runoff_kind_right=["grass"] * n,
                               grade=base.grade, vert_curv=base.vert_curv)
    fit, info = _drift_left(wide)
    assert info["reason"] == "disqualified" and info["violations"] == 1
    assert 3.0 < info["time"] - 0.0 < 30.0
    # the real Monza: 12 m of grass beside the main straight, then the barrier
    _, info = _drift_left(base)
    assert info["reason"] == "barrier"


def test_a_lap_with_a_violation_does_not_count():
    from carsim.learn import lap_score
    # two laps, the second one cut: only the clean one earns the lap bonus
    assert lap_score([30.0], 2, 1100.0) == pytest.approx(1000.0)


def test_offtrack_timer_counts_and_resets():
    from carsim.learn import update_offtrack
    class C:  # a stand-in car
        surfaces = ["kerb", "grass", "grass", "kerb"]
    c = C()
    assert update_offtrack(c, 0.5) and c.offtrack_time == pytest.approx(0.5)
    update_offtrack(c, 0.5)
    assert c.offtrack_time == pytest.approx(1.0)
    c.surfaces = ["track", "kerb", "kerb", "kerb"]  # one wheel on the line is enough
    assert not update_offtrack(c, 0.5) and c.offtrack_time == 0.0


def test_new_inputs_extend_the_layout_and_old_drivers_carry_over():
    old = Policy(len(learn.OBS_V4)).init_random(np.random.default_rng(3))
    theta, migrated = learn.adapt_theta(old.theta)
    assert migrated and theta.size == Policy(learn.N_OBS).size
    assert learn.PHYSICS_VERSION == 4
