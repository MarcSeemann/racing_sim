"""Phase 6: ride-height aerodynamics, ground effect, aero balance and DRS."""
import numpy as np
import pytest

from carsim import Car2D, CarParams, Environment, Inputs, LongitudinalCar, run
from carsim.aero import aero_forces, ride_heights
from carsim.drivers import FullThrottleDriver, ThresholdBrakeDriver
from carsim.params import Aero, Chassis

A, C = Aero(), Chassis()
RHO = Environment().air_density


# ---- the aero map on its own ---------------------------------------------
def test_reference_ride_height_gives_nominal_coefficients():
    r = aero_forces(A, RHO, 50.0, 50.0, A.ride_height_ref)
    assert r.ClA == pytest.approx(A.ClA) and r.CdA == pytest.approx(A.CdA)
    assert r.front_share == pytest.approx(A.front_balance)


def test_lower_floor_makes_more_downforce_until_it_stalls():
    cla = lambda h: aero_forces(A, RHO, 60.0, 60.0, h).ClA
    assert cla(0.030) < cla(0.020) < cla(0.012) < cla(A.stall_height)
    assert cla(0.003) < cla(A.stall_height)  # stalled


def test_downforce_costs_drag_and_moves_balance_forward_as_nose_drops():
    hi, lo = aero_forces(A, RHO, 60.0, 60.0, 0.030), aero_forces(A, RHO, 60.0, 60.0, 0.012)
    assert lo.CdA > hi.CdA
    assert lo.front_share > hi.front_share


def test_drs_cuts_rear_downforce_and_drag():
    shut, open_ = aero_forces(A, RHO, 80.0, 80.0, 0.02), aero_forces(A, RHO, 80.0, 80.0, 0.02, drs=True)
    assert open_.drag < shut.drag and open_.downforce < shut.downforce
    front = lambda r: r.downforce * r.front_share
    assert front(open_) == pytest.approx(front(shut))  # front wing untouched
    assert open_.front_share > shut.front_share


def test_ride_height_from_loads_and_skid_blocks():
    static_f = C.mass * 9.81 * C.front_weight
    static_r = C.mass * 9.81 * (1 - C.front_weight)
    assert ride_heights(C, static_f, static_r) == pytest.approx((C.ride_height_front, C.ride_height_rear))
    hf, _ = ride_heights(C, static_f + 1e6, static_r)
    assert hf == C.min_ride_height


# ---- on the car -------------------------------------------------------------
@pytest.fixture(scope="module")
def flat_out():
    return run(LongitudinalCar(), FullThrottleDriver(), t_end=40.0, log_every=10)


def test_car_squats_at_speed_and_gains_downforce(flat_out):
    log = flat_out
    fast = log["v"] > 80
    assert log["ride_front"][fast].min() < C.ride_height_front - 0.015
    assert log["ride_front"].min() >= C.min_ride_height
    assert log["ClA"][fast].mean() > A.ClA  # ground effect beats the reference value


def test_nose_dives_under_braking():
    car = LongitudinalCar()
    car.set_speed(300 / 3.6)
    cruise = run(car, lambda s, c: Inputs(throttle=1.0, gear=8), t_end=0.5)["ride_front"][-1]
    log = run(car, ThresholdBrakeDriver(0.0), t_end=1.0)
    assert log["ride_front"].min() < cruise - 0.003


def test_drs_raises_top_speed():
    shut = run(LongitudinalCar(), FullThrottleDriver(), t_end=50.0, log_every=50)["v"][-1]
    drs = lambda s, c: Inputs(throttle=1.0, gear=FullThrottleDriver()(s, c).gear, drs=True)
    opened = run(LongitudinalCar(), drs, t_end=50.0, log_every=50)["v"][-1]
    assert opened * 3.6 > shut * 3.6 + 8


def test_softer_front_springs_drop_the_nose_further():
    soft = CarParams()
    soft.chassis.heave_stiffness_front = 300e3
    stiff_log = run(LongitudinalCar(), FullThrottleDriver(), t_end=20.0, log_every=50)
    soft_log = run(LongitudinalCar(soft), FullThrottleDriver(), t_end=20.0, log_every=50)
    assert soft_log["ride_front"][-1] < stiff_log["ride_front"][-1]


def test_four_wheel_car_logs_aero_state():
    car = Car2D(CarParams(tyre_thermals=False))
    car.set_speed(60.0)
    log = run(car, lambda s, c: Inputs(throttle=0.5, gear=6), t_end=1.0)
    assert log["ride_front"][-1] < C.ride_height_front
    assert np.all(log["aero_balance"] > 0.3)
