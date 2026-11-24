"""Car-only lap-time reference (carsim.lapsim): the yardstick for calibrating the car."""
import dataclasses

import numpy as np
import pytest

from carsim.circuits import get_track
from carsim.lapsim import lap, wheel_power_curve
from carsim.params import race_car


@pytest.fixture(scope="module")
def short():
    return get_track("short")


def test_lap_is_finite_and_speeds_are_sensible(short):
    r = lap(short, race_car(10.0))
    assert 15.0 < r["time"] < 60.0
    assert 40.0 < r["min_speed_kmh"] < r["top_speed_kmh"] < 400.0
    assert len(r["v"]) == len(r["s"])


def test_more_downforce_is_faster_through_corners(short):
    base = race_car(10.0)
    more = dataclasses.replace(base, aero=dataclasses.replace(base.aero, ClA=base.aero.ClA * 1.3))
    assert lap(short, more)["time"] < lap(short, base)["time"]
    assert lap(short, more)["min_speed_kmh"] > lap(short, base)["min_speed_kmh"]


def test_less_drag_means_a_higher_top_speed(short):
    base = race_car(10.0)
    slick = dataclasses.replace(base, aero=dataclasses.replace(base.aero, CdA=base.aero.CdA * 0.7))
    assert lap(short, slick)["top_speed_kmh"] > lap(short, base)["top_speed_kmh"]


def test_thin_air_costs_time(short):
    hi = dataclasses.replace(short)
    hi.altitude_m = 2240.0  # Mexico City
    assert lap(hi, race_car(10.0))["air_density"] < 0.95
    assert lap(hi, race_car(10.0))["time"] > lap(short, race_car(10.0))["time"]


def test_power_curve_uses_the_gearbox():
    v, p, top = wheel_power_curve(race_car(0.0))
    assert np.all(p >= 0) and p.max() > 5e5  # several hundred kW at the wheels
    assert top * 3.6 > 380.0  # the top gear is long enough for Monza
