"""Every circuit the simulator knows, by name.

    get_track("short")       1.1 km club circuit          (made-up)
    get_track("test")        2.5 km test circuit          (made-up)
    get_track("hill")        the test circuit with 23 m of elevation
    get_track("monza")       Autodromo Nazionale Monza, 5.79 km, real centreline and widths
    get_track("random:42")   a random circuit (carsim.trackgen), the same one for the same seed
    get_track("power:42")    a random Monza-like circuit: narrow, long straights, tight chicanes

Real circuits come from surveyed data in carsim/data/tracks (see the README there for
sources and licence).
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .track import Track, hill_circuit, short_circuit
from .track import test_circuit as long_circuit
from .trackgen import power_track, random_track

DATA = Path(__file__).parent / "data" / "tracks"


def load_tum_csv(path, name: str, ds: float = 1.0) -> Track:
    """A circuit in the TUM racetrack-database format: x_m, y_m, w_tr_right_m, w_tr_left_m.

    The surveyed points (~5 m apart) are joined with a periodic cubic spline and sampled every
    `ds` m, so curvature is smooth instead of jumping at every survey point."""
    d = np.loadtxt(path, delimiter=",", comments="#")
    x, y, w_right, w_left = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
    u = np.concatenate([[0.0], np.cumsum(seg)])
    close = lambda a: np.append(a, a[0])
    splines = [CubicSpline(u, close(a), bc_type="periodic") for a in (x, y, w_left, w_right)]
    n = int(round(u[-1] / ds))
    t = np.linspace(0.0, u[-1], n, endpoint=False)
    xs, ys, wl, wr = (sp(t) for sp in splines)
    return Track.from_points(name, xs, ys, ds=ds, smooth=5, width_left=wl, width_right=wr)


def monza() -> Track:
    """Autodromo Nazionale Monza, the F1 layout (5.793 km officially), run clockwise.

    The lap starts on the main straight, just before the start/finish line; no elevation
    (Monza is nearly flat; the data has none)."""
    return load_tum_csv(DATA / "monza.csv", "Monza")


NAMED = {"short": short_circuit, "test": long_circuit, "hill": hill_circuit, "monza": monza}
REAL = {"monza"}  # circuits built from survey data


def get_track(name: str) -> Track:
    """A circuit by name: one of NAMED, "random:<seed>" (also "random" = seed 0) or "power:<seed>"."""
    if name.startswith("power:"):
        return power_track(int(name.partition(":")[2]))
    if name.startswith("random"):
        _, _, seed = name.partition(":")
        return random_track(int(seed or 0))
    if name not in NAMED:
        raise KeyError(f"unknown track {name!r}: choose from {', '.join(NAMED)} or random:<seed>")
    return NAMED[name]()


def track_arg(value: str) -> str:
    """argparse type: a valid track name."""
    import argparse
    if value in NAMED:
        return value
    if value.startswith("random"):
        _, _, seed = value.partition(":")
        if seed == "" or seed.isdigit():
            return value
    if value.startswith("power:") and value.partition(":")[2].isdigit():
        return value
    raise argparse.ArgumentTypeError(f"unknown track {value!r}: choose from {', '.join(NAMED)}, random:<seed> "
                                     "or power:<seed>")


class _Registry(dict):
    """dict-like TRACKS that also answers random:<seed> (old code does TRACKS[name]())."""

    def __getitem__(self, name):
        if name in NAMED:
            return NAMED[name]
        return lambda: get_track(name)

    def __contains__(self, name):
        try:
            track_arg(name)
            return True
        except Exception:
            return False


TRACKS = _Registry(NAMED)
