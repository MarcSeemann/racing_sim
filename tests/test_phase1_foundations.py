"""Phase 1: integrator accuracy, determinism, environment basics."""
import numpy as np
import pytest

from carsim import CarParams, Environment, Inputs, LongitudinalCar, run
from carsim.drivers import FullThrottleDriver
from carsim.integrator import euler_step, integrate, rk4_step


def test_rk4_matches_exponential_decay():
    t, y = integrate(lambda t, y: -2.0 * y, np.array([1.0]), 2.0, 0.01, rk4_step)
    assert abs(y[-1, 0] - np.exp(-4.0)) < 1e-8


def test_euler_is_first_order():
    errs = []
    for dt in (0.01, 0.005):
        _, y = integrate(lambda t, y: -2.0 * y, np.array([1.0]), 2.0, dt, euler_step)
        errs.append(abs(y[-1, 0] - np.exp(-4.0)))
    assert errs[0] / errs[1] == pytest.approx(2.0, rel=0.05)


def test_coast_down_matches_analytic_drag_solution():
    """Neutral, no rolling resistance: m_eff dv/dt = -k v^2  ->  v = v0 / (1 + k v0 t / m_eff).

    m_eff includes the spinning wheels (I / r^2), so this also checks wheel inertia.
    """
    p = CarParams()
    p.chassis.crr = 0.0
    p.aero.ride_height_sensitivity = p.aero.induced_drag = 0.0  # constant drag for the analytic case
    car = LongitudinalCar(p)
    car.set_speed(80.0)
    car.state.gear = 0
    log = run(car, lambda s, c: Inputs(throttle=0.0, gear=0), t_end=10.0)
    c = p.chassis
    m_eff = c.mass + (c.wheel_inertia_front + c.wheel_inertia_rear) / c.wheel_radius ** 2
    k = 0.5 * car.env.air_density * p.aero.CdA
    v_exact = 80.0 / (1 + k * 80.0 * 10.0 / m_eff)
    assert log["v"][-1] == pytest.approx(v_exact, rel=5e-4)


def test_simulation_is_deterministic():
    a = run(LongitudinalCar(), FullThrottleDriver(), t_end=5.0)
    b = run(LongitudinalCar(), FullThrottleDriver(), t_end=5.0)
    assert np.array_equal(a["v"], b["v"]) and np.array_equal(a["gear"], b["gear"])


def test_air_density_standard_day():
    assert Environment(air_temp_c=15.0).air_density == pytest.approx(1.225, abs=0.001)
