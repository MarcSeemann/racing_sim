"""Phase 13: physical limits — control rates, tyre relaxation, carbon brake temperature, shift protection."""
import math

import pytest

from carsim.dynamics import Car2D
from carsim.params import CarParams, brake_friction
from carsim.vehicle import Inputs


def car_at(v, **chassis):
    p = CarParams(tyre_thermals=False)
    for k, val in chassis.items():
        setattr(p.chassis, k, val)
    car = Car2D(p)
    car.set_speed(v)
    car.state.gear = 6
    return car


def drive(car, inp, t, dt=0.001):
    for _ in range(int(round(t / dt))):
        row = car.step(inp, dt)
    return row


def test_steering_cannot_flick_from_centre_to_lock():
    car = car_at(20.0)
    c = car.p.chassis
    drive(car, Inputs(steer=c.max_steer), 0.05)
    rate = c.max_steer_rate + (c.max_steer_rate_fast - c.max_steer_rate) * 20.0 / c.steer_rate_speed
    assert car.state.steer == pytest.approx(rate * 0.05, rel=0.05)
    assert car.state.steer < 0.5 * c.max_steer
    drive(car, Inputs(steer=c.max_steer), 0.5)
    assert car.state.steer == pytest.approx(c.max_steer)


def test_steering_is_slower_at_high_speed():
    slow, fast = car_at(15.0), car_at(80.0)
    for car in (slow, fast):
        drive(car, Inputs(steer=0.3, throttle=0.3), 0.05)
    assert fast.state.steer < 0.75 * slow.state.steer


def test_rapid_sawing_at_the_wheel_is_filtered_by_the_rack():
    """Commanding full left/right every 20 ms (what the old AI did) barely moves the wheels."""
    car = car_at(30.0)
    c = car.p.chassis
    peak = 0.0
    for k in range(50):
        drive(car, Inputs(steer=c.max_steer * (1 if k % 2 else -1), throttle=0.3), 0.02)
        peak = max(peak, abs(car.state.steer))
    assert peak < 0.15 * c.max_steer


def test_limits_can_be_switched_off():
    car = car_at(20.0, actuator_limits=False)
    car.step(Inputs(steer=0.3, brake=1.0), 0.001)
    assert car.state.steer == pytest.approx(0.3) and car.state.brake == 1.0


def test_brake_pressure_and_throttle_build_up_over_time():
    car = car_at(50.0)
    drive(car, Inputs(brake=1.0), 0.05)
    assert 0.4 < car.state.brake < 0.6
    drive(car, Inputs(brake=1.0), 0.06)
    assert car.state.brake == pytest.approx(1.0)
    car2 = car_at(30.0)
    drive(car2, Inputs(throttle=1.0), 0.05)
    assert car2.state.throttle == pytest.approx(0.4, abs=0.01)


def test_lateral_force_lags_the_slip_angle_over_the_relaxation_length():
    v = 30.0
    car = car_at(v, actuator_limits=False)
    sigma = car.p.chassis.relaxation_length
    car.state.vy = 1.0  # sudden sideways velocity: geometric slip angle jumps
    car.step(Inputs(), 0.001)
    s = car.state
    assert abs(s.alpha[2]) < 0.2 * abs(s.alpha_cmd[2])  # after 3 cm of rolling, little force yet
    instant = car_at(v, actuator_limits=False, relaxation_length=0.0)
    instant.state.vy = 1.0
    instant.step(Inputs(), 0.001)
    assert instant.state.alpha[2] == pytest.approx(instant.state.alpha_cmd[2])
    # after ~3 relaxation lengths the force has (nearly) caught up
    car3 = car_at(v, actuator_limits=False)
    car3.state.vy = 1.0
    drive(car3, Inputs(), 3 * sigma / v)
    assert abs(car3.state.alpha[2]) > 0.8 * abs(car3.state.alpha_cmd[2])


def test_carbon_brakes_have_a_temperature_window():
    b = CarParams().brakes
    assert brake_friction(b, 600.0) == 1.0
    assert brake_friction(b, 100.0) == pytest.approx(b.cold_mu)
    assert brake_friction(b, 250.0) < 1.0
    assert brake_friction(b, 1200.0) < 1.0
    assert brake_friction(b, 3000.0) == pytest.approx(b.min_mu)


def stop_distance(start_temp):
    p = CarParams(tyre_thermals=False)
    p.brakes.start_temp = start_temp
    car = Car2D(p)
    car.set_speed(55.0)
    car.state.gear = 5
    x0 = car.state.x
    while car.state.vx > 15.0:
        car.step(Inputs(brake=0.35), 0.001)  # below the grip limit: brake bite sets the distance
    return car.state.x - x0, car


def test_cold_brakes_stop_later_and_braking_heats_them():
    d_warm, warm = stop_distance(550.0)
    d_cold, cold = stop_distance(100.0)
    assert d_cold > d_warm + 3.0
    assert min(warm.state.brake_temp) > 550.0  # the stop put heat in
    assert warm.state.brake_temp[0] > warm.state.brake_temp[2]  # front bias: fronts run hotter


def test_brakes_cool_at_speed():
    car = car_at(70.0)
    car.state.brake_temp = [900.0] * 4
    drive(car, Inputs(throttle=0.6), 2.0)
    assert 700.0 < car.state.brake_temp[0] < 890.0


def test_gearbox_refuses_an_over_revving_downshift():
    car = car_at(80.0)
    car.state.gear = 8
    car.step(Inputs(throttle=0.5, gear=2), 0.001)
    assert car.state.gear == 8
    car.step(Inputs(throttle=0.5, gear=7), 0.001)
    assert car.state.gear == 7
