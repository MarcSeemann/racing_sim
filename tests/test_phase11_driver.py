"""Phase 11: racing line, live-input adapter, input record/replay, lap comparison."""
import numpy as np
import pytest

from carsim import Car2D, CarParams, Inputs, run, run_laps
from carsim.compare import delta, lap_trace
from carsim.conditions import Conditions, Weather
from carsim.drivers import LapDriver
from carsim.inputs import DeviceState, HumanDriver, Recorder, Replay
from carsim.racing_line import racing_line
from carsim.track import short_circuit

NO_THERMALS = CarParams(tyre_thermals=False)


@pytest.fixture(scope="module")
def line():
    tr = short_circuit()
    return tr, *racing_line(tr)


# ---- racing line ------------------------------------------------------------------
def test_racing_line_stays_inside_the_track_and_straightens_corners(line):
    tr, path, n = line
    assert np.abs(n).max() <= tr.width / 2 - 1.5 + 1e-9
    assert (path.curvature ** 2).sum() < 0.75 * (tr.curvature ** 2).sum()
    assert len(path.s) == len(tr.s)  # one line sample per centreline sample


def test_racing_line_uses_the_width_outside_apex_outside(line):
    tr, path, n = line
    hairpin = np.argmin(np.abs(tr.s - 320))  # middle of the R 25 left-hander
    entry = np.argmin(np.abs(tr.s - 240))  # on the straight before it
    assert n[hairpin] > 2.5  # apex: inside (left)
    assert n[entry] < -1.0  # entry: outside (right)


@pytest.fixture(scope="module")
def laps():
    tr = short_circuit()
    out = {}
    for name in ("centre", "racing"):
        car = Car2D(NO_THERMALS, conditions=Conditions(tr, Weather.dry(), rubber=1e9))
        car.set_speed(30.0)
        drv = LapDriver(tr, line=name)
        log, times = run_laps(car, drv, 2, log_every=10)
        assert not drv.off_track
        out[name] = (log, times)
    return tr, out


def test_racing_line_is_faster(laps):
    _, out = laps
    assert out["racing"][1][1] < out["centre"][1][1] - 1.0


def test_lap_delta_adds_up_to_the_lap_time_gap(laps):
    tr, out = laps
    a = lap_trace(*out["centre"], 1, tr)
    b = lap_trace(*out["racing"], 1, tr)
    s, d = delta(a, b)
    gap = out["racing"][1][1] - out["centre"][1][1]
    assert d[-1] == pytest.approx(gap, abs=0.3)
    assert d.min() < -0.5  # the racing line gains time along the lap


# ---- live input -----------------------------------------------------------------
def test_human_driver_steering_ratio_deadzone_and_rate_limit():
    car = Car2D(NO_THERMALS)
    car.set_speed(20.0)
    full_left = HumanDriver(lambda t: DeviceState(steer=1.0, throttle=0.3))
    log = run(car, full_left, t_end=1.0)
    max_road = np.radians(180.0 / 12.0)
    assert log["steer"][-1] == pytest.approx(min(max_road, car.p.chassis.max_steer), rel=1e-3)
    assert log["steer"][10] < 0.1 * max_road  # eased in, not instant
    still = HumanDriver(lambda t: DeviceState(steer=0.02))  # inside the deadzone
    assert run(Car2D(NO_THERMALS), still, t_end=0.2)["steer"].max() == 0.0


def test_paddles_shift_once_per_press_and_block_overrev():
    car = Car2D(NO_THERMALS)
    car.set_speed(60.0)
    car.state.gear = 6
    presses = lambda t: DeviceState(throttle=0.5, shift_down=(0.1 < t < 0.3) or (0.5 < t < 0.7) or (0.9 < t < 1.1))
    log = run(car, HumanDriver(presses), t_end=1.5)
    gears = sorted(set(log["gear"]))
    assert gears[0] >= 3  # three presses at most three gears down, and never into an over-rev
    assert log["rpm"].max() < car.p.engine.rev_limit_rpm


# ---- record and replay --------------------------------------------------------------
def test_replay_reproduces_the_lap_exactly_and_shows_a_setup_change():
    tr = short_circuit()

    def car(params):
        c = Car2D(params, conditions=Conditions(tr, Weather.dry(), rubber=1e9))
        c.set_speed(30.0)
        return c
    rec = Recorder(LapDriver(tr, line="racing"))
    original = run(car(NO_THERMALS), rec, t_end=12.0)
    same = run(car(NO_THERMALS), Replay.of(rec), t_end=12.0)
    assert np.array_equal(original["x"], same["x"]) and np.array_equal(original["y"], same["y"])
    heavy = CarParams(tyre_thermals=False)
    heavy.chassis.mass += 80.0
    changed = run(car(heavy), Replay.of(rec), t_end=12.0)
    assert changed["dist"][-1] < original["dist"][-1] - 2.0  # same inputs, heavier car: covers less ground


def test_recording_round_trips_through_csv(tmp_path):
    rec = Recorder(lambda s, c: Inputs(throttle=0.5, steer=0.01, gear=2))
    run(Car2D(NO_THERMALS), rec, t_end=0.05)
    path = tmp_path / "inputs.csv"
    rec.save(str(path))
    back = Replay.load(str(path))
    assert len(back.rows) == len(rec.rows)
    assert back.rows[3][1:] == pytest.approx(rec.rows[3][1:])
