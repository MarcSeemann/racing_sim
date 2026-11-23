"""Calibrate the car against real F1: lap time, top speed and corner speeds of the pole lap.

    python tools/calibrate.py compare --year 2025 --all                     # car vs real pole, every circuit
    python tools/calibrate.py compare --year 2025 --circuits monza,spa --car-kw downforce=low
    python tools/calibrate.py fetch   --year 2025 --circuits monza,spa      # speed traces (F1 login)

Real data, cached in ~/.cache/carsim-fastf1 and NOT committed (the timing data belongs to Formula
One Management; both sources ask for non-commercial use):
  * pole lap times: Jolpica-F1 (https://api.jolpi.ca, the Ergast successor), no account needed
  * speed traces of the pole lap: FastF1 (https://docs.fastf1.dev). Since 2026 F1's timing archive
    needs an F1 account: set F1_USERNAME and F1_PASSWORD in your own shell to use it. Without it,
    `compare` uses the lap times only.

The car side is carsim.lapsim.lap: what the car can do with no driver and no safety margin.
`compare` prints, for each circuit: lap time, top speed and the mean of the slowest corner speeds
(the corner apexes) for real and simulated, and the gaps. A good car model is within a few per
cent everywhere; a pattern in the gaps says what to change (straights fast + corners slow = too
little downforce or grip, too little drag; everything slow = too little power or grip, ...).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carsim.circuits import get_track  # noqa: E402
from carsim.f1_calendar import BY_KEY, CALENDAR  # noqa: E402

CACHE = Path.home() / ".cache" / "carsim-fastf1"


# Jolpica/Ergast circuitId of each calendar circuit
ERGAST = {"melbourne": "albert_park", "shanghai": "shanghai", "suzuka": "suzuka", "bahrain": "bahrain",
          "jeddah": "jeddah", "miami": "miami", "montreal": "villeneuve", "monaco": "monaco",
          "barcelona": "catalunya", "spielberg": "red_bull_ring", "silverstone": "silverstone", "spa": "spa",
          "hungaroring": "hungaroring", "zandvoort": "zandvoort", "monza": "monza", "madring": "madring",
          "baku": "baku", "singapore": "marina_bay", "austin": "americas", "mexico": "rodriguez",
          "interlagos": "interlagos", "las_vegas": "vegas", "lusail": "losail", "yas_marina": "yas_marina"}


def _seconds(t: str) -> float:
    m, _, sec = t.rpartition(":")
    return (int(m) * 60 if m else 0) + float(sec)


def poles(year: int, cache: Path = CACHE) -> dict:
    """{calendar key: (pole lap time s, driver)} for a season, from Jolpica-F1 (cached)."""
    import json
    import urllib.request
    f = cache / f"poles_{year}.json"
    if not f.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"https://api.jolpi.ca/ergast/f1/{year}/qualifying/1.json?limit=100"
        req = urllib.request.Request(url, headers={"User-Agent": "carsim-calibrate"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    races = json.loads(f.read_text())["MRData"]["RaceTable"]["Races"]
    by_id = {v: k for k, v in ERGAST.items()}
    out = {}
    for r in races:
        key = by_id.get(r["Circuit"]["circuitId"])
        q = r["QualifyingResults"][0]
        times = [_seconds(q[x]) for x in ("Q3", "Q2", "Q1") if q.get(x)]
        if key and times:
            out[key] = (min(times), q["Driver"].get("code", ""))
    return out


def event_name(key: str) -> str:
    """FastF1 event name of a calendar circuit ("Italian GP" -> "Italian Grand Prix")."""
    return BY_KEY[key].grand_prix.replace(" GP", " Grand Prix")


def fetch(year: int, key: str, cache: Path = CACHE, log=print):
    """(lap time s, telemetry DataFrame: Distance m, Speed km/h, Throttle, Brake, nGear, RPM) of the
    pole lap, from cache/<year>_<key>.csv or FastF1."""
    import pandas as pd
    out = cache / f"{year}_{key}.csv"
    if out.exists():
        d = pd.read_csv(out, comment="#")
        return float(out.read_text().split("\n", 1)[0].split("lap_s=")[1]), d
    import fastf1
    (cache / "fastf1").mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache / "fastf1"))
    s = fastf1.get_session(year, event_name(key), "Q")
    s.load(laps=True, telemetry=True, weather=False, messages=False)
    lap = s.laps.pick_fastest()
    tel = lap.get_car_data().add_distance()
    d = tel[["Distance", "Speed", "Throttle", "Brake", "nGear", "RPM"]].copy()
    d["Brake"] = d["Brake"].astype(int)
    t = float(lap["LapTime"].total_seconds())
    with open(out, "w") as fh:
        fh.write(f"# {year} {s.event['EventName']} qualifying, {lap['Driver']}, lap_s={t:.3f}\n")
        d.to_csv(fh, index=False)
    log(f"{key}: {year} pole {lap['Driver']} {t:.3f} s")
    return t, d


def apex_speeds(dist, speed, window_m: float = 120.0, max_kmh: float = 260.0):
    """Local speed minima (corner apexes): slowest point within +-window_m, below max_kmh."""
    dist, speed = np.asarray(dist, float), np.asarray(speed, float)
    out = []
    for i in range(len(speed)):
        near = np.abs(dist - dist[i]) <= window_m
        if speed[i] <= speed[near].min() + 1e-9 and speed[i] < max_kmh:
            if not out or dist[i] - out[-1][0] > window_m:
                out.append((float(dist[i]), float(speed[i])))
    return out


def compare(year: int, keys, car_kw: dict, log=print):
    from carsim.lapsim import lap
    from carsim.params import race_car
    rows = []
    pole = poles(year)
    nan = float("nan")
    for key in keys:
        if key not in pole:
            log(f"{key:12s} no {year} qualifying result")
            continue
        t_real = pole[key][0]
        try:  # speed trace, if an F1 login is set up
            _, d = fetch(year, key, log=lambda *a: None)
            top_real = float(d["Speed"].max())
            real_apex = [v for _, v in apex_speeds(d["Distance"], d["Speed"])]
        except Exception:
            top_real, real_apex = nan, []
        r = lap(get_track(f"f1:{key}"), race_car(10.0, **car_kw))
        sim_apex = [v for _, v in apex_speeds(r["s"], r["v"] * 3.6)]
        ra = float(np.mean(real_apex)) if real_apex else nan
        sa = float(np.mean(sim_apex)) if sim_apex else nan
        rows.append((key, t_real, r["time"], top_real, r["top_speed_kmh"], ra, sa))
        trace = (f"   top real {top_real:4.0f} sim {r['top_speed_kmh']:4.0f}   apex real {ra:4.0f} sim {sa:4.0f} km/h"
                 if real_apex else f"   top sim {r['top_speed_kmh']:4.0f}   apex mean sim {sa:4.0f} km/h")
        log(f"{key:12s} lap real {t_real:7.3f} ({pole[key][1]}) sim {r['time']:7.2f} ({r['time'] / t_real - 1:+6.1%}){trace}")
    if rows:
        a = np.array([r[1:] for r in rows], float)
        gaps = a[:, 1] / a[:, 0] - 1
        log(f"{'gap':12s} mean {np.mean(gaps):+5.1%}  min {gaps.min():+5.1%}  max {gaps.max():+5.1%}  "
            f"({len(rows)} circuits)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "compare"])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--circuits", help="comma-separated calendar keys (default: --all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--car-kw", action="append", default=[], help="race_car keyword, e.g. downforce=low")
    a = ap.parse_args()
    keys = [c.key for c in CALENDAR] if a.all or not a.circuits else a.circuits.split(",")
    kw = dict(x.split("=", 1) for x in a.car_kw)
    if a.cmd == "fetch":
        for k in keys:
            try:
                fetch(a.year, k)
            except Exception as e:
                print(f"{k}: {e}")
    else:
        compare(a.year, keys, kw)


if __name__ == "__main__":
    main()
