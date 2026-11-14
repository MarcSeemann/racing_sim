"""Phase 14: sprung body on a dynamic suspension, road profile and kerbs, plank, camber, LSD, aero rake/yaw, ERS."""
import math

import numpy as np
import pytest

from carsim.aero import aero_forces
from carsim.conditions import Conditions, Weather
from carsim.dynamics import Car2D
from carsim.params import CarParams, G, race_car
from carsim.sim import run
from carsim.drivers import FullThrottleDriver, SkidpadDriver
from carsim.track import short_circuit
from carsim.tyres import TyreConsts, TyreParams, combined, combined_fast
from carsim.vehicle import Inputs


def car(**chassis):
    p = CarParams(tyre_thermals=False)
    for k, v in chassis.items():
        setattr(p.chassis, k, v)
    return Car2D(p)


def test_fast_tyre_matches_the_reference_model_and_its_slope():
    p = TyreParams()
    t = TyreConsts(p)
    for k, a, fz in [(0.05, 0.02, 4000), (-0.2, 0.1, 6500), (0.0, -0.12, 2500), (0.3, 0.0, 8000)]:
        fx, fy = combined(p, k, a, fz)
        fx2, fy2, slope = combined_fast(t, k, a, fz)
        num = (combined(p, k + 1e-6, a, fz)[0] - combined(p, k - 1e-6, a, fz)[0]) / 2e-6
        assert (fx2, fy2) == pytest.approx((fx, fy), rel=1e-9, abs=1e-6)
        assert slope == pytest.approx(num, rel=1e-4, abs=1.0)


def test_parked_car_sits_still_at_its_static_loads():
    c = car()
    for _ in range(2000):
        c.step(Inputs(), 0.001)
    s = c.state
    assert abs(s.z) < 1e-6 and abs(s.th) < 1e-6 and abs(s.ph) < 1e-6
    assert sum(s.fz) == pytest.approx(c.mass * G, rel=1e-3)
    assert s.hf == pytest.approx(c.p.chassis.ride_height_front, abs=1e-5)


def test_downforce_squats_the_car_on_its_springs_and_tyres():
    c = car()
    c.set_speed(80.0)
    c.state.gear = 8
    for _ in range(500):
        row = c.step(Inputs(throttle=0.8, gear=8), 0.001)
    drop_f = c.p.chassis.ride_height_front - row["ride_front"]
    # front axle aero load over the heave stiffness (springs and tyres in series)
    expect = row["downforce"] * row["aero_balance"] / c.p.chassis.heave_stiffness_front
    assert drop_f == pytest.approx(expect, rel=0.2)
    assert sum(c.state.fz) == pytest.approx(c.mass * G + row["downforce"], rel=0.02)


def test_roll_gradient_matches_the_setup_target():
    c = car()
    c.set_speed(15.0)
    log = run(c, SkidpadDriver(60.0, v_start=15.0, ramp=0.4), 30.0)
    k = len(log["t"]) // 2
    per_g = np.median(log["roll_deg"][k:] / (log["ay"][k:] / G))
    assert per_g == pytest.approx(c.p.chassis.roll_gradient_deg_per_g, rel=0.1)


def test_dynamic_and_quasi_static_suspension_agree_on_steady_cornering_grip():
    out = {}
    for mode in ("dynamic", "quasi_static"):
        c = car(suspension=mode)
        c.set_speed(15.0)
        out[mode] = max(run(c, SkidpadDriver(60.0, v_start=15.0, ramp=0.4), 40.0)["ay"])
    assert out["dynamic"] == pytest.approx(out["quasi_static"], rel=0.04)


def test_braking_pitches_the_nose_down_and_loads_the_front():
    rows = {}
    for name, inp in (("coast", Inputs()), ("brake", Inputs(brake=0.8))):
        c = car()
        c.set_speed(60.0)
        c.state.gear = 7
        for _ in range(150):
            rows[name] = c.step(inp, 0.001)
        rows[name + "_fz"] = list(c.state.fz)
    assert rows["brake"]["pitch_deg"] > rows["coast"]["pitch_deg"] + 0.05  # dives
    assert rows["brake_fz"][0] > rows["coast_fz"][0] + 500.0


def test_kerbs_lift_the_wheels_and_shake_the_load():
    tr = short_circuit()
    loads = {}
    for name, off in (("track", 0.0), ("kerb", tr.width / 2 + 0.5)):
        cond = Conditions(tr, Weather.dry())
        c = Car2D(CarParams(tyre_thermals=False), conditions=cond)
        s = c.state
        s.x, s.y = float(tr.x[50]), float(tr.y[50] + off)  # first straight runs along +x
        c.set_speed(40.0)
        s.gear = 5
        fz = []
        for _ in range(1000):
            c.step(Inputs(throttle=0.35, gear=5), 0.001)
            fz.append(s.fz[0])
        loads[name] = np.std(fz[200:])
        if name == "kerb":
            assert max(s.road) > 0.005 or np.std(fz[200:]) > 0
    assert loads["kerb"] > 3 * loads["track"]


def test_plank_takes_load_when_the_floor_bottoms_out():
    c = car(heave_stiffness_front=60e3, heave_stiffness_rear=60e3, bump_stop_rate=0.0)
    c.set_speed(85.0)
    c.state.gear = 8
    plank = 0.0
    for _ in range(800):
        row = c.step(Inputs(throttle=0.8, gear=8), 0.001)
        plank = max(plank, row["plank"])
    assert plank > 0 and row["ride_front"] < c.p.chassis.min_ride_height + 0.002
    assert row["ride_front"] > 0.0


def test_camber_helps_the_outside_wheel_in_roll():
    c = car()
    c.set_speed(15.0)
    log = run(c, SkidpadDriver(40.0, v_start=15.0, ramp=0.5), 15.0)
    # left-hand circle: the right-hand (outside) front gains negative camber relative to the road
    assert log["camber_fr"][-1] > c.p.chassis.camber_front_deg  # rolls out of its static camber
    assert log["camber_fl"][-1] < c.p.chassis.camber_front_deg


def test_lsd_locks_more_on_power_than_preload_alone():
    c = car()
    p = c.p.chassis
    c.set_speed(10.0)
    c.state.gear = 2
    c.state.w[2] += 3.0  # inside rear spinning faster
    row = c.step(Inputs(throttle=1.0, gear=2), 0.001)
    assert abs(row["diff_lock"]) > p.diff_preload  # torque-sensing ramp adds to the preload


def test_aero_loses_downforce_in_yaw_and_gains_with_rake():
    a = CarParams().aero
    base = aero_forces(a, 1.2, 60.0, 60.0, 0.02, h_rear=a.rear_height_ref)
    yawed = aero_forces(a, 1.2, 60.0, 60.0, 0.02, h_rear=a.rear_height_ref, yaw=math.radians(5))
    raked = aero_forces(a, 1.2, 60.0, 60.0, 0.02, h_rear=a.rear_height_ref + 0.03)
    assert 0.84 < yawed.ClA / base.ClA < 0.92
    assert raked.ClA > base.ClA and raked.front_share > base.front_share
    assert aero_forces(a, 1.2, 60.0, 60.0, 0.02).ClA == pytest.approx(a.ClA)  # no rear height: as before


def test_ers_deploys_on_power_and_harvests_under_braking():
    c = Car2D(race_car(20.0, tyre_thermals=False))
    log = run(c, FullThrottleDriver(), t_end=6.0)
    assert log["ers_deploy_kw"].max() == pytest.approx(c.p.ers.power / 1e3, rel=0.05)
    used = 1 - log["ers_soc"][-1]
    assert used > 0.05
    soc0 = c.ers_energy
    for _ in range(1500):
        row = c.step(Inputs(brake=0.7), 0.001)
    assert c.ers_energy > soc0 and row["ers_harvest_kw"] > 0 or c.state.vx < c.p.ers.min_harvest_speed


def test_flat_battery_costs_power():
    full = Car2D(race_car(20.0, tyre_thermals=False))
    flat_p = race_car(20.0, tyre_thermals=False)
    flat_p.ers.start_soc = 0.0
    flat = Car2D(flat_p)
    v = []
    for c in (full, flat):
        v.append(run(c, FullThrottleDriver(), t_end=6.0)["v"][-1])
    assert v[1] < v[0] - 3.0
