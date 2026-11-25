"""Racing line — roadmap phase 11.

Minimum-curvature line: choose a lateral offset n(s) from the centreline at every
point, within the track limits less a safety margin, that minimises the summed
squared curvature of the resulting path. Curvature is approximated by the second
difference of the path points, which is linear in the offsets, so this is a
bounded linear least-squares problem (scipy lsq_linear).

A minimum-curvature line is the classic first approximation of the fastest line:
it straightens corners (outside-apex-outside), so corner speeds go up.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from scipy.sparse import lil_matrix

from .track import Track

# Optional disk cache of the solved line (off by default: training makes thousands of random circuits).
# The 3D server turns it on, so a circuit's line is solved once (minutes on a 5-km circuit) and then
# loaded at every start. The key hashes the track's geometry, so a changed track is solved afresh.
_DISK_CACHE: Path | None = None
_CACHE_VERSION = 1  # bump when _solve changes


def use_disk_cache(path) -> None:
    """Keep solved racing lines in `path` (None: off)."""
    global _DISK_CACHE
    _DISK_CACHE = Path(path) if path else None


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "carsim" / "racing_lines"


def _cache_file(track: Track, margin: float, step: int) -> Path:
    h = hashlib.sha1(f"v{_CACHE_VERSION}:{margin!r}:{step}".encode())
    for a in (track.s, track.x, track.y, track.heading, track.width_left, track.width_right):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return _DISK_CACHE / f"{h.hexdigest()}.npz"


def _solve(track: Track, margin: float, step: int):
    """The bounded least-squares problem on evenly spaced coarse samples: (idx, offsets there)."""
    if _DISK_CACHE is None:
        return _solve_lsq(track, margin, step)
    f = _cache_file(track, margin, step)
    try:
        with np.load(f) as d:
            return d["idx"], d["n"]
    except (OSError, KeyError, ValueError):
        pass
    idx, n = _solve_lsq(track, margin, step)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, idx=idx, n=n)
        tmp.replace(f)
    except OSError:
        pass  # a cache that cannot be written only costs time
    return idx, n


def _solve_lsq(track: Track, margin: float, step: int):
    # Evenly spaced samples all the way round, including the gap that closes the lap
    # (an odd short gap at the start line would put a kink in the line).
    m_pts = max(len(track.s) // step, 8)
    idx = (np.arange(m_pts) * len(track.s) / m_pts).astype(int)
    x, y, h = track.x[idx], track.y[idx], track.heading[idx]
    nx, ny = -np.sin(h), np.cos(h)  # left-pointing normals
    m = len(idx)
    A = lil_matrix((2 * m, m))
    b = np.zeros(2 * m)
    for i in range(m):
        im, ip = (i - 1) % m, (i + 1) % m
        b[2 * i] = -(x[im] - 2 * x[i] + x[ip])
        b[2 * i + 1] = -(y[im] - 2 * y[i] + y[ip])
        for j, w in ((im, 1.0), (i, -2.0), (ip, 1.0)):
            A[2 * i, j] += w * nx[j]
            A[2 * i + 1, j] += w * ny[j]
    # Per-sample limits: the track can be wider on one side of its centreline than the other.
    lo = -(track.width_right[idx] - margin)
    hi = track.width_left[idx] - margin
    bad = lo >= hi  # narrower than twice the margin: stay on the middle of the tarmac
    mid = (hi + lo) / 2
    lo, hi = np.where(bad, mid - 1e-3, lo), np.where(bad, mid + 1e-3, hi)
    res = lsq_linear(A.tocsr(), b, bounds=(lo, hi), method="trf", lsmr_tol="auto", max_iter=500)
    return idx, res.x


def _line_points(track: Track, margin: float, step: int):
    """Dense (x, y) of the line and its lateral offsets from the centreline at every centreline sample.

    The line is a periodic cubic spline through the optimised coarse points themselves, not
    "centreline + interpolated offset": the latter kept every few-metre wiggle of the surveyed
    centreline (and, with linear interpolation, a kink at each coarse point). Those curvature spikes
    capped the corner speeds of the lap-time model by ~10 % although the car could drive straight
    through them."""
    idx, n_c = _solve(track, margin, step)
    h = track.heading[idx]
    xc, yc = track.x[idx] - n_c * np.sin(h), track.y[idx] + n_c * np.cos(h)
    s_c = np.append(track.s[idx], track.length)
    x = CubicSpline(s_c, np.append(xc, xc[0]), bc_type="periodic")(track.s)
    y = CubicSpline(s_c, np.append(yc, yc[0]), bc_type="periodic")(track.s)
    n = (y - track.y) * np.cos(track.heading) - (x - track.x) * np.sin(track.heading)
    lo, hi = -(track.width_right - margin), track.width_left - margin
    return x, y, np.clip(n, np.minimum(lo, hi), np.maximum(lo, hi))


def min_curvature_offsets(track: Track, margin: float = 1.5, step: int = 4) -> np.ndarray:
    """Lateral offsets (m, + = left of centreline) at every centreline point."""
    return _line_points(track, margin, step)[2]


def racing_line(track: Track, margin: float = 1.5, step: int = 4):
    """(line Track with one sample per centreline sample, offsets array)."""
    x, y, n = _line_points(track, margin, step)
    line = Track.from_samples(f"{track.name} racing line", x, y, track.z, track.width,
                              width_left=track.width_left - n, width_right=track.width_right + n)
    return line, n


def cached_racing_line(track: Track, margin: float = 1.5, step: int = 4):
    """racing_line(), computed once per track object and kept ON that object, so it is freed with
    the track (training on random circuits makes thousands of tracks; a global cache keyed by id()
    grew without bound, and a new track could even reuse a freed track's id)."""
    cache = track.__dict__.setdefault("_racing_lines", {})
    key = (margin, step)
    if key not in cache:
        cache[key] = racing_line(track, margin, step)
    return cache[key]
