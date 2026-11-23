"""Random circuits — for teaching the AI driver to drive ANY track, not one it memorised.

A random circuit is built the way real ones are drawn: straights joined by constant-radius
corners. Steps:

1. A random star-shaped polygon (corner points at sorted random angles and random distances
   from the middle). Star-shaped means it can never cross itself; pulling one point in deep
   between two far-out ones makes a hairpin, a nearly straight run of points a fast sweep.
2. It is scaled to a target lap length and every corner point is rounded off with an arc of
   random radius (tight hairpins to fast sweepers), never longer than half of either straight
   next to it, so neighbouring corners never overlap.
3. The lap starts part-way down the longest straight (the "main straight"), runs clockwise or
   anticlockwise at random, and gets a width that varies slowly round the lap (each side on
   its own) and, optionally, rolling elevation.
4. A circuit is rejected and redrawn if a corner would be tighter than `r_min`, or if two parts
   of the track that are far apart along the lap come too close side by side (no overlapping
   tarmac and room for run-off).

Everything comes from one seed, so `random_track(42)` is the same circuit every time.

    from carsim.trackgen import random_track
    tr = random_track(7)                       # 2.5-6 km, 11-16 m wide
    tr = random_track(7, elevation=15.0)       # with up to ~15 m of climbs and drops
    tr = power_track(7)                        # Monza-like: narrow, long straights, tight chicanes
"""
import dataclasses
import math

import numpy as np
from scipy.spatial import cKDTree

from .track import Track


@dataclasses.dataclass
class TrackGenConfig:
    length: tuple = (2500.0, 6000.0)  # m, target lap length range
    corners: tuple = (7, 16)  # number of corner points of the polygon (inclusive)
    width: tuple = (11.0, 16.0)  # m, nominal track width range (FIA Grade 1 minimum is 12 m)
    width_variation: float = 1.0  # m, how much each side wanders from nominal
    r_min: float = 15.0  # m, tightest corner allowed (a slow F1 hairpin)
    r_max: float = 300.0  # m, widest corner radius drawn
    dent: float = 0.45  # corner points sit 1-dent .. 1 of the way out: bigger = more hairpins
    deep_dent: float = 0.2  # chance that a corner point is pulled in much deeper (hairpins, infields)
    chicane: float = 0.35  # chance that a straight over 350 m gets a chicane
    chicane_radius: tuple = (15.0, 45.0)  # m
    chicane_angle: tuple = (25.0, 55.0)  # deg, how far the first part of a chicane turns
    chicane_min_straight: float = 350.0  # m, only straights longer than this can get a chicane
    min_half_width: float = 4.0  # m, the narrowest each side of the centreline may get
    runoff: float = 10.0  # m of space required between two separate parts of the track
    elevation: float = 0.0  # m, peak-to-peak height range (0 = flat)
    kerb_width: float = 1.5


def _turn_angle(a, b):
    """Signed angle (rad) from direction a to direction b, anticlockwise positive."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def _smooth_periodic(rng, n, amplitude, modes=(1, 2, 3, 5)):
    """A smooth random function round the lap, n samples, peak-to-peak about `amplitude`."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    f = np.zeros(n)
    for k in modes:
        f += rng.uniform(0.3, 1.0) / k * np.sin(k * t + rng.uniform(0, 2 * math.pi))
    span = f.max() - f.min()
    return f * (amplitude / span) if span > 0 else f


def _segments(rng, cfg: TrackGenConfig):
    """One attempt: a list of ("straight", L) / ("arc", R, deg) that closes, or None."""
    n = int(rng.integers(cfg.corners[0], cfg.corners[1] + 1))
    ang = np.sort((np.arange(n) + rng.uniform(-0.35, 0.35, n)) * 2 * math.pi / n)
    rad = rng.uniform(1.0 - cfg.dent, 1.0, n)
    deep = rng.random(n) < cfg.deep_dent
    rad[deep] = rng.uniform(0.15, 0.5, deep.sum())
    pts = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    # Scale so the polygon is a little longer than the target (rounding corners shortens it).
    target = rng.uniform(*cfg.length)
    edges = np.roll(pts, -1, axis=0) - pts
    pts *= 1.06 * target / np.hypot(edges[:, 0], edges[:, 1]).sum()
    edges = np.roll(pts, -1, axis=0) - pts  # edge k runs from point k to point k+1
    lens = np.hypot(edges[:, 0], edges[:, 1])
    dirs = edges / lens[:, None]
    turns = np.array([_turn_angle(dirs[k - 1], dirs[k]) for k in range(n)])  # at point k
    radii, tans = np.zeros(n), np.zeros(n)
    log_lo, log_hi = math.log(cfg.r_min), math.log(cfg.r_max)
    for k in range(n):
        half_turn = abs(turns[k]) / 2
        if half_turn < 1e-3:
            continue
        r = math.exp(rng.uniform(log_lo, log_hi))
        t_max = 0.5 * min(lens[k - 1], lens[k])
        if r * math.tan(half_turn) > t_max:
            r = t_max / math.tan(half_turn)
        if r < cfg.r_min:
            return None
        radii[k], tans[k] = r, r * math.tan(half_turn)
    segs = []
    for k in range(n):  # corner k, then the straight from corner k to corner k+1
        if radii[k] > 0:
            segs.append(("arc", float(radii[k]), float(math.degrees(turns[k]))))
        straight = lens[k] - tans[k] - tans[(k + 1) % n]
        if straight > cfg.chicane_min_straight and rng.random() < cfg.chicane:
            segs += _chicane(rng, cfg, straight)
        elif straight > 0.5:
            segs.append(("straight", float(straight)))
    return segs


def _chicane(rng, cfg: TrackGenConfig, straight: float):
    """A straight of this length with a chicane in it. Both shapes leave the car on the same
    line and heading as the plain straight would, so the lap still closes:
      "bump":  left-right-left (or mirrored), like Monza's first chicane,
      "s-s":   an S-bend out and an S-bend back, with a short straight in between."""
    r = rng.uniform(*cfg.chicane_radius)
    a = rng.uniform(*cfg.chicane_angle)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    along = 4 * r * math.sin(math.radians(a))  # distance covered along the straight's direction
    if rng.random() < 0.5:
        shape, mid = [("arc", r, sign * a), ("arc", r, -2 * sign * a), ("arc", r, sign * a)], 0.0
    else:
        mid = rng.uniform(0.0, 60.0)
        shape = [("arc", r, sign * a), ("arc", r, -sign * a), ("straight", mid),
                 ("arc", r, -sign * a), ("arc", r, sign * a)]
    rest = straight - along - mid
    if rest < 150.0:
        return [("straight", straight)]
    pre = rest * rng.uniform(0.3, 0.8)
    return [("straight", pre)] + shape + [("straight", rest - pre)]


def _start_on_main_straight(segs, frac=0.6):
    """Rotate the segment list so the lap starts `frac` of the way down the longest straight."""
    k = max((i for i, s in enumerate(segs) if s[0] == "straight"), key=lambda i: segs[i][1])
    L = segs[k][1]
    return [("straight", L * (1 - frac))] + segs[k + 1:] + segs[:k] + [("straight", L * frac)]


def _clear_of_itself(tr: Track, runoff: float) -> bool:
    """No two parts of the track that are far apart along the lap overlap or nearly touch."""
    pts = np.c_[tr.x, tr.y][::2]
    s = tr.s[::2]
    half = ((tr.width_left + tr.width_right) / 2)[::2]
    need = 2 * half.max() + 2 * tr.kerb_width + runoff
    pairs = cKDTree(pts).query_pairs(need, output_type="ndarray")
    if len(pairs) == 0:
        return True
    i, j = pairs[:, 0], pairs[:, 1]
    gap_s = np.abs(s[i] - s[j])
    gap_s = np.minimum(gap_s, tr.length - gap_s)
    d = np.hypot(*(pts[i] - pts[j]).T)
    # Points close along the lap are naturally close in space; for points that are far apart
    # along the lap, the straight-line distance must leave room for both widths and run-off.
    far = gap_s > np.maximum(3.0 * d, 2.5 * need)
    req = half[i] + half[j] + 2 * tr.kerb_width + runoff
    return not np.any(far & (d < req))


def random_track(seed: int, cfg: TrackGenConfig | None = None, name: str | None = None,
                 elevation: float | None = None, max_tries: int = 500) -> Track:
    """A random closed circuit, the same one for the same seed (and config)."""
    cfg = cfg or TrackGenConfig()
    if elevation is not None:
        cfg = dataclasses.replace(cfg, elevation=elevation)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        segs = _segments(rng, cfg)
        if segs is None:
            continue
        clockwise = rng.random() < 0.5
        if clockwise:
            segs = [(s[0], s[1], -s[2]) if s[0] == "arc" else s for s in segs]
        segs = _start_on_main_straight(segs)
        try:
            tr = Track.from_segments(name or f"Random {seed}", segs, width=12.0)
        except ValueError:
            continue
        n = len(tr.s)
        w = rng.uniform(*cfg.width)
        wl = np.maximum(w / 2 + _smooth_periodic(rng, n, cfg.width_variation), cfg.min_half_width)
        wr = np.maximum(w / 2 + _smooth_periodic(rng, n, cfg.width_variation), cfg.min_half_width)
        tr = dataclasses.replace(tr, width_left=wl, width_right=wr, kerb_width=cfg.kerb_width,
                                 grade=None, vert_curv=None)
        if not _clear_of_itself(tr, cfg.runoff):
            continue
        if cfg.elevation > 0:
            z = _smooth_periodic(rng, n, cfg.elevation, modes=(1, 2, 3))
            z -= z[0]
            tr = tr.with_elevation(lambda s, z=z, L=tr.length: float(np.interp(s, tr.s, z, period=L)))
        return tr
    raise RuntimeError(f"no valid circuit for seed {seed} after {max_tries} tries")


# "Power circuits", like Monza: few corners on a long lap (so long straights), narrow (~8-11 m,
# Monza averages 9.4 m), and tight chicanes at the end of the straights (Monza's first chicane is
# R ~9 m after 940 m flat out). The ordinary random circuits never go below R 15 m or 10.6 m wide.
POWER = TrackGenConfig(length=(4500.0, 7000.0), corners=(5, 9), width=(8.0, 11.0), r_min=15.0,
                       chicane=0.7, chicane_radius=(9.0, 20.0), chicane_angle=(35.0, 70.0),
                       chicane_min_straight=300.0, min_half_width=3.6)


def power_track(seed: int, elevation: float | None = None) -> Track:
    """A random Monza-like circuit (see POWER), the same one for the same seed."""
    return random_track(seed, POWER, name=f"Power {seed}", elevation=elevation)


def track_stats(tr: Track) -> dict:
    """Summary numbers used in tests and reports."""
    k = np.abs(tr.curvature)
    straight = k < 1 / 1000
    # Longest run of straight-ish track (wrapping round the lap).
    run = best = 0
    for v in np.concatenate([straight, straight]):
        run = run + 1 if v else 0
        best = max(best, run)
    ds = tr.length / len(tr.s)
    return {"length_m": tr.length, "min_radius_m": 1 / k.max() if k.max() > 0 else math.inf,
            "corners": int(np.sum((k[1:] > 1 / 500) & (k[:-1] <= 1 / 500))),
            "longest_straight_m": min(best, len(tr.s)) * ds, "width_mean_m": tr.width,
            "width_min_m": float((tr.width_left + tr.width_right).min()),
            "direction": "anticlockwise" if np.sum(tr.curvature) > 0 else "clockwise",
            "elevation_range_m": float(tr.z.max() - tr.z.min())}
