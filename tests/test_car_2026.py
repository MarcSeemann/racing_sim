"""The 2026-spec car, aero zones and the driver's extra controls (DRS / straight mode, bias, ERS)."""
import numpy as np
import pytest

from carsim import learn
from carsim.aero import aero_forces
from carsim.aero_zones import zones
from carsim.circuits import get_track
from carsim.params import F1_2026, race_car
from carsim.powertrain import peak_power


def test_2026_car_is_built_to_the_rules():
    p = race_car(0.0, spec="2026")
    assert p.chassis.mass == pytest.approx(768.0) and p.chassis.wheelbase == pytest.approx(3.40)
    assert p.chassis.track_front + F1_2026["tyre_width_m"][0] == pytest.approx(1.90)
    assert p.ers.power == pytest.approx(350e3) and p.ers.exhaust_harvest == 0.0  # no MGU-H
    assert 700e3 < peak_power(p.engine)[0] < 800e3  # ~400 kW engine + 350 kW MGU-K
    assert p.aero.active_aero and p.ers.deploy_taper_kmh == (290.0, 355.0)


def test_straight_mode_cuts_drag_and_downforce_on_both_wings():
    a = race_car(0.0, spec="2026").aero
    closed = aero_forces(a, 1.2, 80.0, 80.0, a.ride_height_ref)
    open_ = aero_forces(a, 1.2, 80.0, 80.0, a.ride_height_ref, drs=True)
    assert open_.drag == pytest.approx(closed.drag * a.straight_cda_factor, rel=0.02)
    assert open_.downforce < closed.downforce and open_.front_share == pytest.approx(closed.front_share)


def test_aero_zones_are_on_the_long_straights_and_monaco_has_none():
    assert not zones(get_track("f1:monaco")).any()
    monza = get_track("monza")
    z = zones(monza)
    assert z[monza.index_at(300.0)]  # the main straight
    assert not z[monza.index_at(930.0)]  # the first chicane


def _episode(act_bias, track="monza", car="2026", t=6.0, start_s=100.0):
    """A driver that does nothing but the given output biases (steer, pedal, aero, bias, ers)."""
    th = np.zeros(learn.Policy(learn.N_OBS).size)
    th[-learn.N_ACT:] = np.arctanh(np.clip(act_bias, -0.999, 0.999))
    return learn.run_episode(th, learn.EpisodeConfig(track=track, max_time=t, start_s=start_s,
                                                     start_speed=60.0, car=car), record=True)


def test_straight_mode_is_faster_on_a_straight():
    _, closed, _ = _episode([0, 1, -1, 0, 0])
    _, opened, _ = _episode([0, 1, 1, 0, 0])
    assert opened["progress"] > closed["progress"]


def test_saving_energy_is_slower_but_keeps_the_battery():
    from carsim.dynamics import Car2D  # noqa: F401
    _, normal, _ = _episode([0, 1, -1, 0, 0])
    _, save, _ = _episode([0, 1, -1, 0, -1])
    assert save["progress"] < normal["progress"]


def test_old_two_control_drivers_carry_over_unchanged():
    old = learn.Policy(len(learn.OBS_V4), n_act=2).init_random(np.random.default_rng(1))
    theta, migrated = learn.adapt_theta(old.theta, obs_names=learn.OBS_V4)
    new = learn.Policy(learn.N_OBS, theta=theta)
    o = np.random.default_rng(2).standard_normal(learn.N_OBS)
    assert migrated and np.allclose(new.act(o)[:2], old.act(o[:len(learn.OBS_V4)]))
    assert np.allclose(new.act(o)[2:], 0.0)  # new controls neutral
