"""Phase 8: track geometry, elevation, surfaces and rubbering-in."""
import math

import numpy as np
import pytest

from carsim import Car2D, CarParams, Inputs, run, run_laps
from carsim.conditions import COMPOUNDS, Conditions, Weather
from carsim.drivers import LapDriver
from carsim.params import G
from carsim.track import Track, hill_circuit, short_circuit
from carsim.track import test_circuit as make_circuit

NO_THERMALS = CarParams(tyre_thermals=False)


# ---- geometry ----------------------------------------------------------------
def test_track_from_surveyed_points_recovers_curvature(tmp_path):
    th = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    path = tmp_path / "circle.csv"
    np.savetxt(path, np.c_[80 * np.cos(th), 80 * np.sin(th), np.zeros_like(th)],
               delimiter=",", header="x,y,z", comments="")
    tr = Track.from_csv(str(path), "circle")
    assert tr.length == pytest.approx(2 * math.pi * 80, rel=1e-3)
    assert np.median(tr.curvature) == pytest.approx(1 / 80, rel=0.01)


def test_hill_circuit_has_climb_crest_and_compression():
    tr = hill_circuit()
    assert tr.z.max() - tr.z.min() > 15
    crest = np.argmin(np.abs(tr.s - 1600))
    dip = np.argmin(np.abs(tr.s - 2300))
    assert tr.vert_curv[crest] < 0 < tr.vert_curv[dip]
    assert abs(tr.z[0] - tr.z[-1]) < 0.2  # closes in height too


def test_surfaces_across_the_track():
    tr = make_circuit()
    edge = tr.width / 2
    assert tr.surface_at(0.0) == "track"
    assert tr.surface_at(edge + 0.5) == "kerb"
    assert tr.surface_at(-(edge + tr.kerb_width + 1)) == "grass"


def test_grass_grips_less_than_kerb_less_than_track():
    cond = Conditions(make_circuit(), rubber=1e9)
    slick = COMPOUNDS["slick"]
    g = lambda off: cond.wheel_surface(100.0, off, slick, 50.0)[0]
    assert g(0.0) > g(7.0) > g(10.0)


# ---- rubber -----------------------------------------------------------------------
def test_sliding_lays_rubber_and_rain_washes_it():
    cond = Conditions(short_circuit(), Weather.dry(), cars_on_track=20)
    b = cond.bin_of(50.0)
    before = cond.rubber_grip(b)
    for _ in range(5000):
        cond.update(0.001, 50.0, 40e3)  # 5 s of heavy sliding at s = 50 m
    laid = cond.rubber_grip(b)
    assert laid > before + 0.01
    cond.weather = Weather.constant(rain_mm_h=20.0)
    for _ in range(60):
        cond.update(1.0, 900.0, 0.0)  # a minute of heavy rain, car elsewhere
    assert cond.rubber_grip(b) < laid


# ---- on track ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def hill_run():
    tr = hill_circuit()
    cond = Conditions(tr, Weather.dry(), cars_on_track=20)
    car = Car2D(NO_THERMALS, conditions=cond)
    car.set_speed(30.0)
    driver = LapDriver(tr)
    log, laps = run_laps(car, driver, 3, log_every=5)
    assert not driver.off_track
    return tr, cond, log, laps


def test_crest_unloads_and_compression_loads_the_car(hill_run):
    tr, _, log, _ = hill_run
    total = log["fz_fl"] + log["fz_fr"] + log["fz_rl"] + log["fz_rr"]
    static = NO_THERMALS.chassis.mass * G * np.cos(log["grade"]) + log["downforce"]
    ratio = total / static
    s = log["s_track"]
    crest = (np.abs(s - 1600) < 15) & (log["v"] > 60)
    dip = (np.abs(s - 2300) < 15) & (log["v"] > 50)
    assert ratio[crest].mean() < 0.95
    assert ratio[dip].mean() > 1.05


def test_track_rubbers_in_and_laps_get_faster(hill_run):
    _, cond, _, laps = hill_run
    assert laps[2] < laps[1] - 0.05  # constant tyres (no thermals): the gain is the track
    corner = cond.bin_of(720.0)  # hairpin: lots of sliding
    straight = cond.bin_of(1600.0)
    assert cond.rubber[corner] > 2 * cond.rubber[straight]  # corners collect far more rubber
    assert cond.rubber_grip(corner) > cond.rubber_grip(straight)


def test_grass_slows_the_car():
    tr = make_circuit()

    def coast(offset):
        car = Car2D(NO_THERMALS, conditions=Conditions(tr, rubber=1e9))
        car.state.y = offset  # straight along +x from the start line
        car.set_speed(40.0)
        return run(car, lambda s, c: Inputs(throttle=0.0, gear=s.gear), t_end=2.0)
    on_track, on_grass = coast(0.0), coast(-12.0)
    assert on_grass["grass_wheels"].min() == 4 and on_track["grass_wheels"].max() == 0
    assert on_grass["v"][-1] < on_track["v"][-1] - 1.0
