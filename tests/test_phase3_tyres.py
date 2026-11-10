"""Phase 3: tyre model on its own, then its effect on the car."""
import math

import numpy as np
import pytest

from carsim import CarParams, Environment, Inputs, LongitudinalCar, run
from carsim.drivers import BrakeFromSpeedDriver, FullThrottleDriver, ThresholdBrakeDriver
from carsim.tyres import TyreParams, combined, fx_pure, fy_pure

TP = TyreParams()
FZ = TP.nominal_load


# ---- the tyre alone ------------------------------------------------------
def test_longitudinal_peak_at_requested_slip_and_mu():
    k = np.linspace(0.0, 1.0, 4001)
    f = np.array([fx_pure(TP, x, FZ) for x in k])
    assert k[f.argmax()] == pytest.approx(TP.longitudinal.peak_slip, abs=0.002)
    assert f.max() / FZ == pytest.approx(TP.longitudinal.mu0, rel=1e-3)


def test_lateral_peak_at_requested_angle():
    a = np.radians(np.linspace(0, 20, 4001))
    f = np.array([fy_pure(TP, x, FZ) for x in a])
    assert a[f.argmax()] == pytest.approx(TP.lateral.peak_slip, abs=math.radians(0.1))


def test_locked_wheel_grips_less_than_peak():
    ratio = fx_pure(TP, -1.0, FZ) / fx_pure(TP, -TP.longitudinal.peak_slip, FZ)
    assert 0.55 < ratio < 0.9


def test_forces_are_odd_in_slip():
    assert fx_pure(TP, -0.05, FZ) == pytest.approx(-fx_pure(TP, 0.05, FZ))
    assert fy_pure(TP, -0.05, FZ) == pytest.approx(-fy_pure(TP, 0.05, FZ))


def test_load_sensitivity_grip_coefficient_falls_with_load():
    peak = lambda fz: fx_pure(TP, TP.longitudinal.peak_slip, fz) / fz
    assert peak(2000) > peak(4000) > peak(8000)
    assert fx_pure(TP, 0.11, 8000) > fx_pure(TP, 0.11, 4000)  # total force still rises


def test_combined_slip_stays_inside_friction_ellipse():
    """Braking at the peak leaves little lateral force (why you brake in a straight line)."""
    _, fy_alone = combined(TP, 0.0, TP.lateral.peak_slip, FZ)
    fx, fy = combined(TP, -TP.longitudinal.peak_slip, TP.lateral.peak_slip, FZ)
    assert fy < 0.8 * fy_alone
    for kappa in np.linspace(-0.3, 0.3, 13):
        for alpha in np.radians(np.linspace(-15, 15, 13)):
            fx, fy = combined(TP, kappa, alpha, FZ)
            dx = TP.mu(TP.longitudinal, FZ) * FZ
            dy = TP.mu(TP.lateral, FZ) * FZ
            assert (fx / dx) ** 2 + (fy / dy) ** 2 <= 1.0 + 1e-6


# ---- the tyre on the car -------------------------------------------------
def _no_tc():
    p = CarParams()
    p.chassis.traction_control = False
    return p


def test_full_throttle_launch_without_tc_spins_the_rears_and_is_slower():
    raw = run(LongitudinalCar(_no_tc()), FullThrottleDriver(), t_end=6.0)
    tc = run(LongitudinalCar(), FullThrottleDriver(), t_end=6.0)
    assert raw["kappa_rear"][raw["t"] < 1.0].max() > 1.0
    assert raw.first_time("v", 100 / 3.6) > tc.first_time("v", 100 / 3.6) + 0.5


def test_traction_control_holds_slip_near_peak():
    log = run(LongitudinalCar(), FullThrottleDriver(), t_end=4.0)
    k = log["kappa_rear"][(log["t"] > 0.5) & (log["clutch"] == "locked")]
    assert 0.05 < np.median(k) < 0.25


def test_full_pedal_locks_fronts_once_downforce_fades():
    """At 300 km/h downforce gives enough grip to take full pedal; lower down the fronts lock."""
    log = run(LongitudinalCar(), BrakeFromSpeedDriver(300 / 3.6), t_end=30.0,
              stop=lambda s: s.v == 0.0 and s.t > 1.0)
    braking = log["brake"] > 0
    locked_speeds = log["v"][braking & log["front_locked"]] * 3.6
    assert len(locked_speeds) > 0
    assert locked_speeds.max() < 290  # not locked at the start of the stop


def test_threshold_braking_beats_locking_the_wheels():
    def stop_dist(driver):
        log = run(LongitudinalCar(), driver, t_end=30.0, stop=lambda s: s.v == 0.0 and s.t > 1.0)
        i0 = int(np.argmax(log["brake"] > 0))
        return log["x"][-1] - log["x"][i0]

    locked = stop_dist(BrakeFromSpeedDriver(200 / 3.6))
    skilled = stop_dist(ThresholdBrakeDriver(200 / 3.6))
    assert skilled < 0.9 * locked


def test_lower_grip_lengthens_braking():
    wet = CarParams()
    wet.tyres.grip_scale = 0.6
    dist = []
    for p in (CarParams(), wet):
        log = run(LongitudinalCar(p), ThresholdBrakeDriver(200 / 3.6), t_end=30.0,
                  stop=lambda s: s.v == 0.0 and s.t > 1.0)
        i0 = int(np.argmax(log["brake"] > 0))
        dist.append(log["x"][-1] - log["x"][i0])
    assert dist[1] > 1.3 * dist[0]


def test_hot_thin_air_costs_power_and_top_speed_balance():
    """35 °C at altitude: less power AND less drag; net top speed effect is small but real."""
    cold = LongitudinalCar(env=Environment(air_temp_c=10.0))
    hot = LongitudinalCar(env=Environment(air_temp_c=35.0, pressure_pa=80000.0))
    assert hot.env.air_density < 0.8 * cold.env.air_density
    t_cold = run(cold, FullThrottleDriver(), t_end=6.0).first_time("v", 200 / 3.6)
    t_hot = run(hot, FullThrottleDriver(), t_end=6.0).first_time("v", 200 / 3.6)
    assert t_hot != pytest.approx(t_cold, abs=0.01)
