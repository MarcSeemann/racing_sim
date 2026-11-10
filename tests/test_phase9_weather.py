"""Phase 9: weather — rain, standing water, drying line, track temperature, wind, wet tyres."""
import math

import numpy as np
import pytest

from carsim import Car2D, CarParams, Inputs, run, run_laps
from carsim.conditions import COMPOUNDS, Conditions, SurfaceParams, Weather, WeatherPoint, aquaplane_speed, wet_grip
from carsim.drivers import LapDriver
from carsim.track import short_circuit
from carsim.track import test_circuit as make_circuit

NO_THERMALS = CarParams(tyre_thermals=False)
SLICK, INTER, WET = COMPOUNDS["slick"], COMPOUNDS["intermediate"], COMPOUNDS["wet"]


# ---- tyres in water ------------------------------------------------------------
def test_compound_order_flips_from_dry_to_wet():
    dry = [wet_grip(c, 0.0, 30.0) for c in (SLICK, INTER, WET)]
    soaked = [wet_grip(c, 2.0, 30.0) for c in (SLICK, INTER, WET)]
    assert dry[0] > dry[1] > dry[2]
    assert soaked[2] > soaked[1] > soaked[0]


def test_aquaplaning_above_the_critical_speed():
    v = aquaplane_speed(SLICK, 1.5)
    assert v < aquaplane_speed(WET, 1.5)
    assert wet_grip(SLICK, 1.5, 1.3 * v) < 0.6 * wet_grip(SLICK, 1.5, 0.9 * v)


# ---- water and temperature on track ------------------------------------------------
def test_rain_fills_the_track_and_the_line_dries_first():
    tr = short_circuit()
    cond = Conditions(tr, Weather.constant(rain_mm_h=10.0, cloud=1.0), cars_on_track=20)
    for _ in range(1800):
        cond.update(1.0, 0.0, 0.0)  # 30 min of rain, car parked at the line
    sp = cond.sp
    equilibrium = 10.0 / 3600 * sp.drain_tau
    assert cond._off_line == pytest.approx(equilibrium, rel=0.15)
    # Rain stops; the car laps (moving through every bin) and clears its line.
    cond.weather = Weather.constant(rain_mm_h=0.0, cloud=0.3)
    s, dt = 0.0, 0.05
    for _ in range(int(300 / dt)):  # 5 min at 50 m/s
        s = (s + 50 * dt) % tr.length
        cond.update(dt, s, 0.0)
    assert cond.water.mean() < 0.5 * cond._off_line


def test_sun_warms_and_rain_cools_the_track():
    cond = Conditions(short_circuit(), Weather.constant(air_temp_c=20.0, cloud=0.0), track_temp_c=20.0)
    for _ in range(3600):
        cond.update(1.0, 0.0, 0.0)
    sunny = cond.track_temp
    assert sunny > 20.0 + 0.8 * cond.sp.solar_gain
    cond.weather = Weather.constant(air_temp_c=20.0, rain_mm_h=5.0, cloud=1.0)
    for _ in range(900):
        cond.update(1.0, 0.0, 0.0)
    assert cond.track_temp < sunny - 10


def test_weather_timeline_interpolates():
    w = Weather([WeatherPoint(0.0, 20.0, 0.0), WeatherPoint(100.0, 10.0, 8.0)])
    mid = w.at(50.0)
    assert mid.air_temp_c == pytest.approx(15.0) and mid.rain_mm_h == pytest.approx(4.0)


# ---- wind ----------------------------------------------------------------------------
def straight_run(wind_ms, from_deg):
    tr = make_circuit()  # starts along +x
    cond = Conditions(tr, Weather.constant(wind_ms=wind_ms, wind_from_deg=from_deg), rubber=1e9)
    car = Car2D(NO_THERMALS, conditions=cond)
    car.set_speed(60.0)
    return run(car, lambda s, c: Inputs(throttle=1.0, gear=6), t_end=3.0)


def test_headwind_slows_and_crosswind_pushes_the_car():
    calm, head, cross = straight_run(0.0, 0.0), straight_run(10.0, 0.0), straight_run(10.0, 90.0)
    assert head["v"][-1] < calm["v"][-1] - 1.0
    # 10 m/s from the left: ~90 N of side force, a few cm of drift in 3 s, towards -y.
    assert abs(calm["y"][-1]) < 0.005
    assert cross["y"][-1] < -0.02


# ---- on track ---------------------------------------------------------------------------
FROZEN = SurfaceParams(drain_tau=1e12, evap_mm_h=0.0, line_clear_per_lap=0.0)


def lap(compound, water_mm):
    tr = short_circuit()
    cond = Conditions(tr, Weather.constant(air_temp_c=18.0, cloud=1.0), surface=FROZEN,
                      water_mm=water_mm, rubber=2e6)
    car = Car2D(NO_THERMALS, conditions=cond, compound=compound)
    car.set_speed(30.0)
    driver = LapDriver(tr)
    _, laps = run_laps(car, driver, 1)
    assert not driver.off_track
    return laps[0]


def test_slicks_win_in_the_dry_wets_win_in_standing_water():
    assert lap("slick", 0.0) < lap("intermediate", 0.0)
    assert lap("wet", 0.8) < lap("slick", 0.8) - 5.0


def test_water_cools_the_tyres():
    tr = short_circuit()
    temps = []
    for depth in (0.0, 1.0):
        cond = Conditions(tr, Weather.constant(air_temp_c=18.0, cloud=1.0), surface=FROZEN, water_mm=depth)
        car = Car2D(CarParams(), conditions=cond, compound="wet")
        car.set_speed(30.0)
        log, _ = run_laps(car, LapDriver(tr), 1)
        temps.append(np.mean([log[f"tsurf_{n}"][-1] for n in ("fl", "fr", "rl", "rr")]))
    assert temps[1] < temps[0] - 5
