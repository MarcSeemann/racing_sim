"""Phase 10: fuel burn and the changing mass of the car."""
import numpy as np
import pytest

from carsim import Car2D, CarParams, Inputs, race_car, run, run_laps
from carsim.conditions import Conditions, Weather
from carsim.drivers import FullThrottleDriver, LapDriver
from carsim.params import G
from carsim.powertrain import RPM_TO_RADS
from carsim.track import short_circuit


def test_default_car_carries_no_fuel_model():
    car = Car2D(CarParams(tyre_thermals=False))
    assert car.mass == CarParams().chassis.mass and car.fuel == 0.0


def test_fuel_flow_follows_crank_power():
    p = race_car(50.0, tyre_thermals=False)
    car = Car2D(p)
    log = run(car, FullThrottleDriver(), t_end=10.0)
    i = int(np.argmax(log["fuel_flow"]))
    from carsim.powertrain import engine_torque
    p_crank = engine_torque(p.engine, log["rpm"][i], 1.0, car.env.air_density) * log["rpm"][i] * RPM_TO_RADS
    expected = p.fuel.idle_flow + p_crank / (p.fuel.efficiency * p.fuel.lhv)
    assert log["fuel_flow"][i] == pytest.approx(expected, rel=0.02)
    assert 0.024 < log["fuel_flow"].max() < 0.032  # ~90-115 kg/h at full power
    burned = 50.0 - log["fuel"][-1]
    assert burned == pytest.approx(np.sum(log["fuel_flow"]) * 0.001, rel=0.01)


def test_fuel_moves_mass_balance_and_cog():
    empty, full = Car2D(race_car(0.0)), Car2D(race_car(110.0))
    assert full.mass == pytest.approx(empty.mass + 110.0)
    assert full.front_weight > empty.front_weight  # tank slightly ahead of the car's centre of mass
    assert full.cog_height < empty.cog_height  # and low in the car
    assert full.yaw_inertia > empty.yaw_inertia


def test_heavy_car_accelerates_slower():
    t = []
    for fuel in (0.0, 100.0):
        log = run(Car2D(race_car(fuel, tyre_thermals=False)), FullThrottleDriver(), t_end=8.0)
        t.append(log.first_time("v", 200 / 3.6))
    assert t[1] > t[0] + 0.1


def test_empty_tank_stops_the_engine():
    car = Car2D(race_car(0.05, tyre_thermals=False))
    car.set_speed(40.0)
    car.state.gear = 4
    log = run(car, lambda s, c: Inputs(throttle=1.0, gear=4), t_end=6.0)
    assert log["fuel"][-1] == 0.0
    dry = log["fuel"] == 0.0
    v = log["v"][dry][5:]  # skip the step on which the tank ran dry
    assert np.all(np.diff(v) <= 1e-4)  # no more drive: the car only slows (tiny wheel-spin-down wobble aside)
    assert v[-1] < v[0] - 2.0


@pytest.fixture(scope="module")
def fuel_laps():
    tr = short_circuit()
    out = {}
    for fuel in (5.0, 60.0):
        cond = Conditions(tr, Weather.dry(), rubber=1e9)  # fully rubbered: no track evolution
        car = Car2D(race_car(fuel, tyre_thermals=False), conditions=cond)
        car.set_speed(30.0)
        log, laps = run_laps(car, LapDriver(tr, line="racing"), 3)
        out[fuel] = (log, laps)
    return out


def test_fuel_costs_lap_time(fuel_laps):
    light, heavy = fuel_laps[5.0][1][2], fuel_laps[60.0][1][2]
    per_10kg = (heavy - light) / (60.0 - 5.0) * 10.0
    assert 0.03 < per_10kg < 0.25  # ~0.1 s per 10 kg on a 1.1 km lap


def test_laps_get_faster_as_fuel_burns(fuel_laps):
    log, laps = fuel_laps[60.0]
    assert laps[2] < laps[1]
    per_lap = log["fuel"][0] - log["fuel"][-1]
    assert 0.3 < per_lap / 3 < 1.5  # kg per 1.1 km lap
