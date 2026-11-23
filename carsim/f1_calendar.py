"""The 2026 Formula 1 calendar as a HELD-OUT test set: 24 real circuits the driver never trains on.

    get_track("f1:suzuka")                      one circuit (see CALENDAR for the keys)
    python -m carsim.f1_calendar list           the circuits: round, length, width, data source
    python -m carsim.f1_calendar drive --policy random --laps 3 --workers 8
                                                drive a trained driver round the whole calendar
                                                (report only: nothing is trained or selected)

These circuits are for REPORTING only, like Monza in `learn.py`: they are driven and the lap
times written down, but never practised on and never used to pick the best driver; if they
picked the driver they would stop being unseen. `learn.py train` refuses them as training tracks.

Data (carsim/data/tracks/, see the README there; built by tools/build_f1_2026.py):

    "tum"     surveyed centreline and widths from the TUM racetrack database, unchanged
    "hybrid"  the TUM survey where the layout is unchanged, the current layout (f1-circuits
              GeoJSON) where the circuit was rebuilt since the survey; widths from the survey
    "geo"     current layout from the f1-circuits GeoJSON only (traced on maps, no widths):
              a constant nominal width; lap times on these are a few per cent less certain

Bahrain and Saudi Arabia were cancelled during the 2026 season (Middle East war); they stay in
the test set (`cancelled=True`): they are real F1 circuits all the same.
"""
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Circuit:
    round: int  # 2026 calendar order
    key: str  # track name "f1:<key>", data file <key>.csv
    name: str
    grand_prix: str
    length_m: int  # official lap length
    source: str  # "tum", "hybrid" or "geo" (see the module docstring)
    tum: str | None = None  # TUM racetrack-database file (tracks/<tum>.csv)
    geo: str | None = None  # bacinger/f1-circuits file (circuits/<geo>.geojson)
    width_m: float = 12.0  # "geo" only: nominal width (FIA Grade 1 minimum is 12 m)
    reverse: bool = False  # "geo"/"hybrid": the GeoJSON points run against the race direction
    cancelled: bool = False


CALENDAR = (
    Circuit(1, "melbourne", "Albert Park Circuit", "Australian GP", 5278, "hybrid", "Melbourne", "au-1953"),
    Circuit(2, "shanghai", "Shanghai International Circuit", "Chinese GP", 5451, "tum", "Shanghai", "cn-2004"),
    Circuit(3, "suzuka", "Suzuka International Racing Course", "Japanese GP", 5807, "tum", "Suzuka", "jp-1962"),
    Circuit(4, "bahrain", "Bahrain International Circuit", "Bahrain GP", 5412, "tum", "Sakhir", "bh-2002",
            cancelled=True),
    Circuit(5, "jeddah", "Jeddah Corniche Circuit", "Saudi Arabian GP", 6174, "geo", None, "sa-2021",
            cancelled=True),
    Circuit(6, "miami", "Miami International Autodrome", "Miami GP", 5412, "geo", None, "us-2022"),
    Circuit(7, "montreal", "Circuit Gilles-Villeneuve", "Canadian GP", 4361, "tum", "Montreal", "ca-1978"),
    Circuit(8, "monaco", "Circuit de Monaco", "Monaco GP", 3337, "geo", None, "mc-1929", width_m=9.0),
    Circuit(9, "barcelona", "Circuit de Barcelona-Catalunya", "Barcelona-Catalunya GP", 4657, "hybrid",
            "Catalunya", "es-1991"),
    Circuit(10, "spielberg", "Red Bull Ring", "Austrian GP", 4318, "tum", "Spielberg", "at-1969"),
    Circuit(11, "silverstone", "Silverstone Circuit", "British GP", 5891, "tum", "Silverstone", "gb-1948"),
    Circuit(12, "spa", "Circuit de Spa-Francorchamps", "Belgian GP", 7004, "tum", "Spa", "be-1925"),
    Circuit(13, "hungaroring", "Hungaroring", "Hungarian GP", 4381, "tum", "Budapest", "hu-1986"),
    Circuit(14, "zandvoort", "Circuit Zandvoort", "Dutch GP", 4259, "hybrid", "Zandvoort", "nl-1948"),
    Circuit(15, "monza", "Autodromo Nazionale Monza", "Italian GP", 5793, "tum", "Monza", "it-1922"),
    Circuit(16, "madring", "Madring", "Spanish GP", 5474, "geo", None, "es-2026"),
    Circuit(17, "baku", "Baku City Circuit", "Azerbaijan GP", 6003, "geo", None, "az-2016"),
    Circuit(18, "singapore", "Marina Bay Street Circuit", "Singapore GP", 4927, "geo", None, "sg-2008",
            reverse=True),
    Circuit(19, "austin", "Circuit of the Americas", "United States GP", 5513, "tum", "Austin", "us-2012"),
    Circuit(20, "mexico", "Autódromo Hermanos Rodríguez", "Mexico City GP", 4304, "tum", "MexicoCity", "mx-1962"),
    Circuit(21, "interlagos", "Autódromo José Carlos Pace", "São Paulo GP", 4309, "tum", "SaoPaulo", "br-1940"),
    Circuit(22, "las_vegas", "Las Vegas Strip Circuit", "Las Vegas GP", 6201, "geo", None, "us-2023"),
    Circuit(23, "lusail", "Lusail International Circuit", "Qatar GP", 5419, "geo", None, "qa-2004"),
    Circuit(24, "yas_marina", "Yas Marina Circuit", "Abu Dhabi GP", 5281, "hybrid", "YasMarina", "ae-2009"),
)
BY_KEY = {c.key: c for c in CALENDAR}
PREFIX = "f1:"
DATA = Path(__file__).parent / "data" / "tracks"


def is_calendar(name: str) -> bool:
    """True for a held-out calendar circuit name ("f1:<key>")."""
    return isinstance(name, str) and name.startswith(PREFIX)


def track_names(include_cancelled: bool = True) -> list:
    return [PREFIX + c.key for c in CALENDAR if include_cancelled or not c.cancelled]


def load(key: str):
    """The Track of calendar circuit `key` (without the "f1:" prefix)."""
    from .circuits import load_tum_csv, monza
    if key not in BY_KEY:
        raise KeyError(f"unknown calendar circuit {key!r}: choose from {', '.join(BY_KEY)}")
    if key == "monza":
        return monza()  # the very same Monza as get_track("monza")
    c = BY_KEY[key]
    return load_tum_csv(DATA / f"{key}.csv", c.name)


# ---- report: drive a trained driver round the calendar -------------------------------------
def _drive_one(args):
    run, name, laps = args
    from . import learn
    cfg = learn.EpisodeConfig(track=name, laps=laps, max_time=learn.default_max_time(name, laps))
    _, info = learn.run_episode(learn.load_policy(run), cfg, hidden=learn._run_hidden(learn.run_dir(run)))
    return name, info


def drive(run: str, laps: int = 3, workers: int = 1, names=None, log=print) -> list:
    """Drive the best driver of training run `run` (e.g. "random") `laps` laps from the line on
    each calendar circuit. Returns one row per circuit; nothing is saved into the run."""
    from concurrent.futures import ProcessPoolExecutor
    names = list(names or track_names())
    jobs = [(run, n, laps) for n in names]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_drive_one, jobs))
    else:
        results = [_drive_one(j) for j in jobs]
    rows = []
    for name, info in results:
        c = BY_KEY[name[len(PREFIX):]]
        lt = info["laps"][:laps]
        done = len(lt) >= laps
        rows.append({"round": c.round, "circuit": c.key, "laps_done": len(lt), "finished": done,
                     "best_lap_s": min(lt) if lt else math.nan, "total_s": sum(lt) if done else math.nan,
                     "progress_m": info["progress"], "reason": "" if done else info["reason"],
                     "source": c.source, "cancelled": c.cancelled})
        txt = " ".join(f"{x:.2f}" for x in lt) or "-"
        if not done:
            txt += f" then {info['reason']} at {info['progress']:.0f} m"
        log(f"R{c.round:02d} {c.key:12s} {txt}")
    n_done = sum(r["finished"] for r in rows)
    log(f"finished {n_done}/{len(rows)} circuits ({laps} laps each) with the driver of run {run!r}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m carsim.f1_calendar", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("drive")
    d.add_argument("--policy", "--run", dest="run", default="random", help="training run whose best driver drives")
    d.add_argument("--laps", type=int, default=3)
    d.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1))
    d.add_argument("--only", help="comma-separated circuit keys (default: the whole calendar)")
    d.add_argument("--csv", help="also write the results to this CSV file")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        from .circuits import get_track
        print(f"{'rnd':>3}  {'track name':15s} {'grand prix':24s} {'km':>6} {'official':>8} {'width m':>8}  source")
        for c in CALENDAR:
            tr = get_track(PREFIX + c.key)
            note = "  (cancelled)" if c.cancelled else ""
            print(f"{c.round:3d}  {PREFIX + c.key:15s} {c.grand_prix:24s} {tr.length / 1000:6.3f} "
                  f"{c.length_m / 1000:8.3f} {tr.width:8.1f}  {c.source}{note}")
    else:
        names = [PREFIX + k.strip() for k in a.only.split(",")] if a.only else None
        rows = drive(a.run, a.laps, a.workers, names)
        if a.csv:
            import csv
            with open(a.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
