"""Phase 2: straight-line performance against independent expectations."""
import numpy as np
import pytest

from carsim import CarParams, Environment, LongitudinalCar, run
from carsim.drivers import FullThrottleDriver, ThresholdBrakeDriver
from carsim.params import G
from carsim.powertrain import RPM_TO_RADS, full_throttle_torque, overall_ratio


def expected_top_speed(p: CarParams, env: Environment) -> float:
    """Highest speed where the best gear's wheel force still beats drag + rolling resistance."""
    c, gb = p.chassis, p.gearbox
    rho = env.air_density

    def surplus(v):
        best = 0.0
        for g in range(1, len(gb.ratios) + 1):
            rpm = v / c.wheel_radius * overall_ratio(gb, g) / RPM_TO_RADS
            f = full_throttle_torque(p.engine, rpm) * overall_ratio(gb, g) * gb.efficiency / c.wheel_radius
            best = max(best, f)
        v_air = v + env.headwind
        drag = 0.5 * rho * p.aero.CdA * v_air * abs(v_air)
        roll = c.crr * (c.mass * G + 0.5 * rho * p.aero.ClA * v * v)
        return best - drag - roll

    lo, hi = 50.0, 150.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if surplus(mid) > 0 else (lo, mid)
    return lo


@pytest.fixture(scope="module")
def flat_out():
    p = CarParams()
    # Constant aero so the power = drag check has a closed-form answer (ride-height aero: phase 6 tests).
    p.aero.ride_height_sensitivity = p.aero.induced_drag = p.aero.balance_per_m = 0.0
    car = LongitudinalCar(p)
    return car, run(car, FullThrottleDriver(), t_end=60.0)


def test_top_speed_matches_power_drag_balance(flat_out):
    car, log = flat_out
    # Within 1%: the rigid-wheel estimate ignores the ~1% tyre slip at top speed.
    assert log["v"][-1] == pytest.approx(expected_top_speed(car.p, car.env), rel=0.01)


def test_top_speed_is_in_single_seater_range(flat_out):
    _, log = flat_out
    assert 330 < log["v"][-1] * 3.6 < 380


def test_acceleration_times_plausible(flat_out):
    _, log = flat_out
    t100 = log.first_time("v", 100 / 3.6)
    t200 = log.first_time("v", 200 / 3.6)
    assert 2.0 < t100 < 3.5
    assert 4.0 < t200 < 7.0


def test_rev_limit_respected_and_gears_climb(flat_out):
    car, log = flat_out
    assert log["rpm"].max() <= car.p.engine.rev_limit_rpm
    assert np.all(np.diff(log["gear"]) >= 0)
    assert log["gear"][-1] >= 7


def test_braking_from_300(flat_out):
    car = LongitudinalCar()
    log = run(car, ThresholdBrakeDriver(300 / 3.6), t_end=40.0,
              stop=lambda s: s.v == 0.0 and s.t > 1.0)
    braking = log["brake"] > 0
    dist = log["x"][-1] - log["x"][braking][0]
    peak_g = -log["a"][braking].min() / G
    assert log["v"][-1] == 0.0
    assert 4.0 < peak_g < 6.5  # aero makes high-speed braking far stronger than 1 g
    assert 80 < dist < 200


def test_headwind_lowers_top_speed():
    still = run(LongitudinalCar(), FullThrottleDriver(), t_end=60.0)["v"][-1]
    windy = run(LongitudinalCar(env=Environment(headwind=5.0)), FullThrottleDriver(), t_end=60.0)["v"][-1]
    # Power-limited: P = k (v + w)^2 v gives dv ~ -2w/3 = -3.3 m/s; gearing trims it.
    assert windy < still - 2.5
