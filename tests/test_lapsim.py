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


def test_racing_line_is_no_noisier_than_the_centreline():
    # The line is a spline through the optimised points; it used to keep every few-metre wiggle of the
    # survey plus a kink every 4 m, curvature spikes that capped the corner speeds of this model.
    from carsim.lapsim import LINE_MARGIN, LINE_STEP
    from carsim.racing_line import cached_racing_line
    tr = get_track("monza")
    line = cached_racing_line(tr, LINE_MARGIN, LINE_STEP)[0]
    wiggle = lambda t: float(np.sum(np.abs(np.diff(t.curvature))))
    assert wiggle(line) < wiggle(tr)
    assert np.all(line.width_left >= LINE_MARGIN - 0.05) and np.all(line.width_right >= LINE_MARGIN - 0.05)


def test_calibrated_car_laps_monza_close_to_the_2025_pole():
    # docs/calibration/2025.md: Verstappen 1:18.792; the car-only lap is a few per cent slower
    r = lap(get_track("monza"), race_car(10.0, downforce="low"))
    assert 78.792 * 0.97 < r["time"] < 78.792 * 1.06
    assert 340.0 < r["top_speed_kmh"] < 370.0  # ~350-360 km/h with DRS
