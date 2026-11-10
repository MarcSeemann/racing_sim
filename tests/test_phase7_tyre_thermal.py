"""Phase 7: tyre temperature, pressure, wear and their effect over a stint."""
import numpy as np
import pytest

from carsim import Car2D, CarParams, Environment, run_laps
from carsim.track import test_circuit as make_circuit
from carsim.drivers import LapDriver
from carsim.thermal import ThermalParams, TyreThermalState, grip_factor, pressure, update

TP = ThermalParams()


# ---- the thermal model on its own -----------------------------------------
def test_grip_peaks_in_the_temperature_window():
    g = lambda t: grip_factor(TP, TyreThermalState(t, t))
    assert g(TP.t_opt) > g(TP.t_opt - 30) and g(TP.t_opt) > g(TP.t_opt + 30)
    assert g(TP.t_opt - 45) < 0.95
    assert g(TP.t_opt + 30) < g(TP.t_opt - 30)  # overheating hurts more than running cold


def test_pressure_follows_core_temperature():
    cold = TyreThermalState(TP.blanket_temp, TP.blanket_temp)
    hot = TyreThermalState(110.0, 110.0)
    assert pressure(TP, cold) == pytest.approx(TP.pressure_set)
    assert pressure(TP, hot) > pressure(TP, cold) + 0.2


def test_wear_costs_grip_and_the_cliff_costs_more():
    g = lambda w: grip_factor(TP, TyreThermalState(TP.t_opt, TP.t_opt, wear=w))
    before = g(TP.cliff) - g(TP.cliff - 0.1)
    after = g(TP.cliff + 0.1) - g(TP.cliff)
    assert g(0.0) > g(0.5) > g(TP.cliff)
    assert after < 3 * before  # (both negative) the drop past the cliff is much steeper


def test_sliding_heats_surface_first_and_idle_tyre_cools():
    st = TyreThermalState(70.0, 70.0)
    for _ in range(3000):  # 3 s of hard sliding
        update(TP, st, p_slide=40e3, fz=5000, v=60, t_air=20, t_track=30, dt=0.001)
    assert st.t_surface > st.t_core + 8
    for _ in range(600_000):  # 10 min parked
        update(TP, st, p_slide=0.0, fz=2000, v=0.0, t_air=20, t_track=30, dt=0.001)
    assert st.t_surface < 35 and st.t_core < 35


def test_hot_tread_wears_faster():
    cool, hot = TyreThermalState(90.0, 90.0), TyreThermalState(130.0, 130.0)
    update(TP, cool, 10e3, 4000, 60, 20, 30, 1.0)
    update(TP, hot, 10e3, 4000, 60, 20, 30, 1.0)
    assert hot.wear > 2 * cool.wear


# ---- on track -------------------------------------------------------------
def stint(laps, params=None, env=None):
    car = Car2D(params or CarParams(), env)
    car.set_speed(30.0)
    driver = LapDriver(make_circuit())
    log, times = run_laps(car, driver, laps)
    assert not driver.off_track
    return log, times


@pytest.fixture(scope="module")
def short_stint():
    return stint(4)


def test_test_circuit_closes_and_is_lapped():
    tr = make_circuit()
    assert 2400 < tr.length < 2600
    assert (tr.curvature < 0).any() and (tr.curvature > 0).any()  # left and right corners


def test_out_lap_is_slowest_as_tyres_warm_up(short_stint):
    log, times = short_stint
    assert times[0] > times[1] + 0.5
    assert log["tsurf_rr"][-1] > TP.blanket_temp + 10


def test_hardest_worked_tyre_runs_hottest_and_wears_most(short_stint):
    """Anticlockwise lap: the outside (right) rear works hardest, the inside front least."""
    log, _ = short_stint
    assert log["tsurf_rr"][-1] > log["tsurf_fl"][-1] + 15
    assert log["wear_rr"][-1] > log["wear_fl"][-1]
    assert log["press_rr"][-1] > log["press_fl"][-1]


def test_worn_tyres_slow_the_car():
    """Wear sped up 5x so a short run shows the fall-off after the best lap (before the cliff)."""
    p = CarParams()
    p.thermal.wear_rate *= 5
    log, times = stint(6, p)
    best = int(np.argmin(times[1:])) + 1  # ignore the out-lap
    assert best < len(times) - 1
    assert times[-1] > times[best] + 0.15
    assert all(b > a for a, b in zip(times[best:], times[best + 1:]))  # slower every lap
    assert log["wear_rr"][-1] > 0.35


def test_hot_track_runs_hotter_tyres():
    cool, _ = stint(2, env=Environment(track_temp_c=20.0))
    hot, _ = stint(2, env=Environment(track_temp_c=50.0))
    assert hot["tsurf_rr"][-1] > cool["tsurf_rr"][-1] + 1.5
