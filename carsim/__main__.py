"""Command line — roadmap phase 12.

    python -m carsim lap --track test                       # one hot lap on the racing line
    python -m carsim stint --track test --laps 20 --fuel 60 # a stint: tyres, fuel, lap times
    python -m carsim whatif --preset hot-day                # what-if study vs a baseline
    python -m carsim whatif --change car.aero.ClA=*1.1 --change weather.air_temp_c=+5
    python -m carsim compare --change car.aero.ClA=*1.1 --out compare.html   # interactive lap viewer
    python -m carsim presets                                # list the what-if presets
"""
import argparse
import sys

from .whatif import PRESETS, Scenario, TRACKS, apply_change, results_table, run_study


def parse_change(text: str):
    """'car.aero.ClA=*1.1' -> ('mul', ...), '=+5' -> ('add', ...), '=4.2' or '=wet' -> ('set', ...)."""
    path, _, val = text.partition("=")
    op = "set"
    if val[:1] == "*":
        op, val = "mul", val[1:]
    elif val[:1] in "+-" and len(val) > 1 and (val[1].isdigit() or val[1] == "."):
        op = "add"
    try:
        num = float(val)
        value = int(num) if num.is_integer() and op == "set" and "." not in val else num
    except ValueError:
        value = {"true": True, "false": False}.get(val.lower(), val)
    return (op, path, value)


def scenario_from(a) -> Scenario:
    return Scenario(track=a.track, laps=a.laps, line=a.line, compound=a.compound, fuel_kg=a.fuel,
                    tyre_thermals=not a.no_thermals)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m carsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lap", "stint", "whatif", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--track", choices=TRACKS, default="short")
        p.add_argument("--laps", type=int, default={"stint": 15}.get(name, 3))
        p.add_argument("--line", choices=("racing", "centre"), default="racing")
        p.add_argument("--compound", choices=("slick", "intermediate", "wet"), default="slick")
        p.add_argument("--fuel", type=float, default=0.0, help="kg of fuel (uses the 798 kg race car)")
        p.add_argument("--no-thermals", action="store_true", help="tyres always at optimum")
        if name in ("whatif", "compare"):
            p.add_argument("--preset", choices=PRESETS)
            p.add_argument("--change", action="append", default=[], help="path=value, path=+delta or path=*factor")
        if name == "compare":
            p.add_argument("--out", default="compare.html")
            p.add_argument("--name", help="label for the changed run (default: the change itself)")
    sub.add_parser("presets")
    a = ap.parse_args(argv)

    if a.cmd == "presets":
        for k, v in PRESETS.items():
            print(f"{k}: " + ", ".join(v))
        return
    base = scenario_from(a)

    if a.cmd in ("lap", "stint"):
        from .whatif import run_scenario
        r = run_scenario(base)
        if r.get("off_track"):
            print("went off track")
            return 1
        for i, t in enumerate(r["laps"], 1):
            print(f"lap {i:2d}  {t:7.3f} s")
        print(f"top speed {r['top_speed_kmh']:.0f} km/h, max lateral {r['max_lat_g']:.2f} g, "
              f"tyres {r['tyre_temp_c']:.0f} C (hottest {r['hottest_tyre_c']:.0f} C), rear-right wear {100 * r['wear_rr']:.0f}%"
              + (f", fuel used {r['fuel_used_kg']:.2f} kg" if a.fuel else ""))
        return

    changes = dict(PRESETS[a.preset]) if a.preset else {}
    for c in a.change:
        changes[c] = parse_change(c)
    if not changes:
        ap.error("give --preset or at least one --change")

    if a.cmd == "whatif":
        print(results_table(run_study(base, changes)))
        return

    # compare: baseline vs the first change, as an interactive HTML viewer
    import copy

    from .conditions import Conditions, Weather
    from .drivers import LapDriver
    from .dynamics import Car2D
    from .params import CarParams, race_car
    from .sim import run_laps
    from .viewer import write_viewer
    name, change = next(iter(changes.items()))
    name = a.name or name
    runs = []
    for label, sc in (("Baseline", base), (name, _chain(base, change))):
        tr = TRACKS[sc.track]()
        params = copy.deepcopy(sc.car) if sc.car else (race_car(sc.fuel_kg) if sc.fuel_kg > 0 else CarParams())
        params.tyre_thermals = sc.tyre_thermals
        cond = Conditions(tr, Weather(copy.deepcopy(sc.weather)), surface=sc.surface, rubber=sc.rubber)
        car = Car2D(params, conditions=cond, compound=sc.compound)
        car.set_speed(30.0)
        log, laps = run_laps(car, LapDriver(tr, line=sc.line), sc.laps, log_every=5)
        runs.append((label, log, laps))
        print(f"{label}: {laps[-1]:.3f} s")
    write_viewer(a.out, TRACKS[base.track](), runs, title=f"{name} vs baseline ({base.track} circuit)")
    print(f"wrote {a.out}")


def _chain(sc, change):
    for c in (change if isinstance(change, list) else [change]):
        sc = apply_change(sc, c)
    return sc


if __name__ == "__main__":
    sys.exit(main())
