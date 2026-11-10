"""Phases 4–5: cornering dynamics and brake balance on the four-wheel model.

Tyre thermals are switched off here so these tests see the chassis alone;
phase 7 tests cover temperature and wear."""
import math

import numpy as np
import pytest

from carsim import Car2D, CarParams, Inputs, LongitudinalCar, run
from carsim.drivers import FullThrottleDriver, SkidpadDriver
from carsim.params import G


def skidpad(radius, params=None, t_end=120.0, v_start=10.0):
    d = SkidpadDriver(radius, v_start=v_start)
    car = Car2D(params or CarParams(tyre_thermals=False))
    car.set_speed(v_start)
    log = run(car, d, t_end=t_end, stop=lambda s: abs(d.radius_error(s)) > 3.0)
    err = np.hypot(log["x"], log["y"] - radius) - radius
    held = (np.abs(err) < 0.5) & (log["t"] > 2.0)  # skip the turn-in transient
    return log, held


def at_ay(log, held, g):
    """Index of the first held sample at lateral acceleration >= g."""
    idx = np.nonzero(held & (log["ay"] / G >= g))[0]
    assert len(idx), f"never reached {g} g"
    return idx[0]


@pytest.fixture(scope="module")
def pad50():
    return skidpad(50.0)


# ---- consistency ----------------------------------------------------------
def test_straight_line_matches_longitudinal_model():
    a = run(LongitudinalCar(), FullThrottleDriver(), t_end=8.0)
    b = run(Car2D(CarParams(tyre_thermals=False)), FullThrottleDriver(), t_end=8.0)
    # 5%: the 4-wheel model lags its load transfer (suspension), so its launch spins a bit more.
    assert b.first_time("v", 200 / 3.6) == pytest.approx(a.first_time("v", 200 / 3.6), rel=0.05)
    assert np.abs(b["y"]).max() == 0.0 and np.abs(b["psi"]).max() == 0.0


def test_left_and_right_turns_mirror():
    def drive(sign):
        car = Car2D(CarParams(tyre_thermals=False))
        car.set_speed(30.0)
        return run(car, lambda s, c: Inputs(throttle=0.3, steer=sign * 0.05, gear=s.gear), t_end=3.0)
    left, right = drive(1), drive(-1)
    assert left["y"][-1] == pytest.approx(-right["y"][-1], rel=1e-9)
    assert left["yaw_rate"][-1] > 0


def test_low_speed_steer_matches_geometry(pad50):
    log, held = pad50
    i = at_ay(log, held, 0.2)
    assert log["steer"][i] == pytest.approx(math.atan(3.6 / 50.0), rel=0.05)


# ---- handling -------------------------------------------------------------
def test_skidpad_lateral_grip(pad50):
    log, held = pad50
    peak = log["ay"][held].max() / G
    assert 1.8 < peak < 2.5  # slick tyres + some downforce at ~115 km/h


def test_downforce_raises_grip_in_faster_corners(pad50):
    log50, held50 = pad50
    log150, held150 = skidpad(150.0, v_start=40.0)  # start near the limit so the ramp reaches it
    assert log150["ay"][held150].max() > 1.2 * log50["ay"][held50].max()


def test_car_understeers(pad50):
    """Steering needed rises with lateral g, and the fronts run more slip than the rears."""
    log, held = pad50
    lo, hi = at_ay(log, held, 0.5), at_ay(log, held, 1.8)
    assert log["steer"][hi] > log["steer"][lo] + math.radians(0.15)
    a_front = abs(log["alpha_fl"][hi] + log["alpha_fr"][hi]) / 2
    a_rear = abs(log["alpha_rl"][hi] + log["alpha_rr"][hi]) / 2
    assert a_front > a_rear


def test_more_front_roll_stiffness_means_more_understeer():
    steer = []
    for share in (0.45, 0.65):
        p = CarParams(tyre_thermals=False)
        p.chassis.roll_stiffness_front = share
        log, held = skidpad(50.0, p, t_end=60.0)
        steer.append(log["steer"][at_ay(log, held, 1.6)])
    assert steer[1] > steer[0]


def test_lateral_load_transfer_loads_outside_wheels(pad50):
    log, held = pad50
    i = at_ay(log, held, 1.5)
    assert log["fz_fr"][i] > log["fz_fl"][i] and log["fz_rr"][i] > log["fz_rl"][i]  # left turn


# ---- brake balance (phase 5) -----------------------------------------------
def brake_straight(front_bias):
    p = CarParams(tyre_thermals=False)
    p.chassis.brake_front_bias = front_bias
    car = Car2D(p)
    car.set_speed(150 / 3.6)
    # A tiny steer input stands in for a real-world disturbance.
    return run(car, lambda s, c: Inputs(brake=1.0, steer=0.01, gear=s.gear), t_end=6.0,
               stop=lambda s: s.vx <= 0.5 and s.t > 0.5)


def test_front_biased_brakes_stay_straight():
    assert np.degrees(np.abs(brake_straight(0.60)["psi"]).max()) < 5


def test_rear_biased_brakes_spin_the_car():
    # The spin threshold sits near 30% front bias; 25% spins clearly.
    assert np.degrees(np.abs(brake_straight(0.25)["psi"]).max()) > 30
