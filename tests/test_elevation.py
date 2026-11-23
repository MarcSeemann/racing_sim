"""Height profiles of the real circuits and the air pressure at their altitude."""
import numpy as np
import pytest

from carsim.circuits import ELEVATION, get_track
from carsim.f1_calendar import CALENDAR, load
from carsim.learn import EpisodeConfig, make_car
from carsim.params import pressure_at_altitude


def test_every_calendar_circuit_has_a_height_file():
    assert {f.stem for f in ELEVATION.glob("*.csv")} == {c.key for c in CALENDAR}


def test_spa_climbs_and_silverstone_does_not():
    spa, silverstone = get_track("f1:spa"), get_track("f1:silverstone")
    assert 85 < np.ptp(spa.z) < 125  # ~102 m from the bottom of Eau Rouge to the top of the lap
    assert np.ptp(silverstone.z) < 20
    assert np.abs(spa.grade).max() > np.radians(8)  # Eau Rouge / Raidillon


def test_profiles_close_and_stay_plausible():
    for c in CALENDAR:
        tr = get_track(f"f1:{c.key}")
        assert np.abs(tr.grade).max() < np.radians(13), c.key  # no 25 % walls from buildings
        z = np.r_[tr.z, tr.z[0]]
        assert abs(z[-1] - z[-2]) < 1.0, c.key  # the lap ends where it started


def test_unreliable_terrain_stays_flat_at_the_right_height():
    mex = get_track("f1:mexico")
    assert np.ptp(mex.z) == 0.0 and 2100 < mex.altitude_m < 2350
    car = make_car(mex, EpisodeConfig(track="f1:mexico"))
    sea = make_car(get_track("f1:silverstone"), EpisodeConfig(track="f1:silverstone"))
    assert car.env.air_density / sea.env.air_density == pytest.approx(0.78, abs=0.04)


def test_elevation_can_be_left_out_and_made_up_circuits_are_unchanged():
    assert np.ptp(load("spa", elevation=False).z) == 0.0
    assert not hasattr(get_track("short"), "altitude_m")
    assert pressure_at_altitude(0.0) == pytest.approx(101325.0)
