"""What-if studies — roadmap phase 12.

Run a baseline scenario and variants of it, and compare lap time and how the car
behaves. A change is plain data, so studies can be saved and run in parallel:

    ("set", "car.aero.ClA", 4.4)                 # set a car parameter
    ("add", "weather.air_temp_c", 5.0)           # +5 C on the whole weather timeline
    ("mul", "car.tyres.grip_scale", 0.95)        # scale a value
    ("set", "scenario.compound", "intermediate") # change the scenario itself

Paths start with `car.` (CarParams), `weather.` (every WeatherPoint), `surface.`
(SurfaceParams) or `scenario.` (the Scenario fields below).

    from carsim.whatif import Scenario, run_study, PRESETS
    results = run_study(Scenario(), PRESETS["hot-day"])
    print(results_table(results))
"""
import copy
import dataclasses
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .conditions import Conditions, SurfaceParams, Weather, WeatherPoint
from .dynamics import Car2D
from .drivers import LapDriver
from .params import CarParams, race_car
from .sim import run_laps
from .track import hill_circuit, short_circuit
from .track import test_circuit as long_circuit

TRACKS = {"short": short_circuit, "test": long_circuit, "hill": hill_circuit}


@dataclass
class Scenario:
    track: str = "short"
    laps: int = 3  # the last lap is the one compared
    line: str = "racing"
    compound: str = "slick"
    fuel_kg: float = 0.0  # > 0 uses race_car(): 798 kg + fuel
    tyre_thermals: bool = True
    rubber: float = 1e9  # J per bin: 1e9 = fully rubbered (no track evolution)
    water_mm: float = 0.0
    cars_on_track: int = 20
    weather: list = field(default_factory=lambda: [WeatherPoint(0.0, 22.0, 0.0, 0.2)])
    car: CarParams | None = None  # None = default car (or race_car when fuel_kg > 0)
    surface: SurfaceParams = field(default_factory=SurfaceParams)


def _resolve(root, path):
    obj = root
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def apply_change(sc: Scenario, change) -> Scenario:
    """A copy of the scenario with one change, e.g. ("add", "weather.air_temp_c", 5)."""
    sc = copy.deepcopy(sc)
    if sc.car is None:
        sc.car = race_car(sc.fuel_kg) if sc.fuel_kg > 0 else CarParams()
    op, path, value = change
    head, _, rest = path.partition(".")

    def update(obj, attr):
        old = getattr(obj, attr)
        new = {"set": lambda: value, "add": lambda: old + value, "mul": lambda: old * value}[op]()
        setattr(obj, attr, new)

    if head == "weather":
        for wp in sc.weather:
            update(wp, rest)
    elif head == "car":
        update(*_resolve(sc.car, rest))
    elif head == "surface":
        update(*_resolve(sc.surface, rest))
    elif head == "scenario":
        update(sc, rest)
    else:
        raise ValueError(f"unknown change target {path!r}")
    return sc


def run_scenario(sc: Scenario) -> dict:
    """Drive the scenario; return the last lap's time and key telemetry."""
    tr = TRACKS[sc.track]()
    params = copy.deepcopy(sc.car) if sc.car is not None else (race_car(sc.fuel_kg) if sc.fuel_kg > 0 else CarParams())
    if sc.fuel_kg > 0:
        params.fuel.start_kg = sc.fuel_kg
    params.tyre_thermals = sc.tyre_thermals
    cond = Conditions(tr, Weather(copy.deepcopy(sc.weather)), surface=copy.deepcopy(sc.surface),
                      cars_on_track=sc.cars_on_track, rubber=sc.rubber, water_mm=sc.water_mm)
    car = Car2D(params, conditions=cond, compound=sc.compound)
    car.set_speed(30.0)
    driver = LapDriver(tr, line=sc.line)
    log, laps = run_laps(car, driver, sc.laps, log_every=10)
    if driver.off_track or len(laps) < sc.laps:
        return {"lap_time": math.nan, "off_track": True}
    t0 = sum(laps[:-1])
    m = log["t"] > t0
    tyres = ("fl", "fr", "rl", "rr")
    return {
        "lap_time": laps[-1], "off_track": False, "laps": laps,
        "top_speed_kmh": float(log["v"][m].max() * 3.6),
        "min_speed_kmh": float(log["v"][m].min() * 3.6),
        "max_lat_g": float(np.abs(log["ay"][m]).max() / 9.81),
        "tyre_temp_c": float(np.mean([log[f"tsurf_{n}"][m].mean() for n in tyres])),
        "hottest_tyre_c": float(max(log[f"tsurf_{n}"][m].max() for n in tyres)),
        "wear_rr": float(log["wear_rr"][-1]),
        "fuel_used_kg": float(log["fuel"][0] - log["fuel"][-1]),
        "track_temp_c": float(log["track_temp"][m].mean()),
    }


def _run_named(item):
    name, sc = item
    return name, run_scenario(sc)


def run_study(base: Scenario, changes: dict, workers: int = 2) -> list:
    """Run the baseline and each named change (a change or a list of changes).
    Returns [(name, result)], baseline first."""
    jobs = [("baseline", base)]
    for name, ch in changes.items():
        sc = base
        for c in (ch if isinstance(ch, list) else [ch]):
            sc = apply_change(sc, c)
        jobs.append((name, sc))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_run_named, jobs))
    return [_run_named(j) for j in jobs]


def results_table(results: list) -> str:
    """Markdown table of the study, deltas against the baseline."""
    base = results[0][1]
    rows = ["| Change | Lap time (s) | Delta (s) | Top speed (km/h) | Min speed (km/h) | Max lateral (g) | Tyre temp (C) |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results:
        if r.get("off_track"):
            rows.append(f"| {name} | went off | | | | | |")
            continue
        d = r["lap_time"] - base["lap_time"]
        rows.append(f"| {name} | {r['lap_time']:.3f} | {d:+.3f} | {r['top_speed_kmh']:.1f} | "
                    f"{r['min_speed_kmh']:.1f} | {r['max_lat_g']:.2f} | {r['tyre_temp_c']:.1f} |")
    return "\n".join(rows)


PRESETS = {
    "hot-day": {"+5 C air": ("add", "weather.air_temp_c", 5.0),
                "+10 C air": ("add", "weather.air_temp_c", 10.0),
                "-5 C air": ("add", "weather.air_temp_c", -5.0)},
    "downforce": {"+10% downforce": ("mul", "car.aero.ClA", 1.10),
                  "-10% downforce": ("mul", "car.aero.ClA", 0.90),
                  "+10% drag": ("mul", "car.aero.CdA", 1.10)},
    "balance": {"brake bias 55%": ("set", "car.chassis.brake_front_bias", 0.55),
                "brake bias 62%": ("set", "car.chassis.brake_front_bias", 0.62),
                "aero balance 45%": ("set", "car.aero.front_balance", 0.45),
                "roll stiffness 65% front": ("set", "car.chassis.roll_stiffness_front", 0.65)},
    "weight": {"+10 kg": ("add", "car.chassis.mass", 10.0),
               "+50 kg fuel": ("set", "scenario.fuel_kg", 50.0)},
    "tyres": {"pressures +0.1 bar": ("add", "car.thermal.pressure_set", 0.1),
              "pressures -0.1 bar": ("add", "car.thermal.pressure_set", -0.1),
              "intermediates": ("set", "scenario.compound", "intermediate"),
              "5% less grip": ("mul", "car.tyres.grip_scale", 0.95)},
    "track": {"green track": ("set", "scenario.rubber", 0.0),
              "centreline (no racing line)": ("set", "scenario.line", "centre"),
              "5 m/s wind (head-on along the main straight)": [("set", "weather.wind_ms", 5.0),
                                                      ("set", "weather.wind_from_deg", 0.0)]},
}
