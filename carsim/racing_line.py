"""Racing line — roadmap phase 11.

Minimum-curvature line: choose a lateral offset n(s) from the centreline at every
point, within the track limits less a safety margin, that minimises the summed
squared curvature of the resulting path. Curvature is approximated by the second
difference of the path points, which is linear in the offsets, so this is a
bounded linear least-squares problem (scipy lsq_linear).

A minimum-curvature line is the classic first approximation of the fastest line:
it straightens corners (outside-apex-outside), so corner speeds go up.
"""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import lil_matrix

from .track import Track


def min_curvature_offsets(track: Track, margin: float = 1.5, step: int = 4) -> np.ndarray:
    """Lateral offsets (m, + = left of centreline) at every centreline point."""
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
    half = track.width / 2 - margin
    res = lsq_linear(A.tocsr(), b, bounds=(-half, half), method="trf", lsmr_tol="auto", max_iter=500)
    # Back to full resolution (periodic interpolation), lightly smoothed.
    s_c = track.s[idx]
    n_full = np.interp(track.s, np.append(s_c, track.length), np.append(res.x, res.x[0]))
    k = np.ones(5) / 5
    n_full = np.convolve(np.concatenate([n_full[-5:], n_full, n_full[:5]]), k, "same")[5:-5]
    return np.clip(n_full, -half, half)


def racing_line(track: Track, margin: float = 1.5, step: int = 4):
    """(line Track with one sample per centreline sample, offsets array)."""
    n = min_curvature_offsets(track, margin, step)
    x = track.x - n * np.sin(track.heading)
    y = track.y + n * np.cos(track.heading)
    line = Track.from_samples(f"{track.name} racing line", x, y, track.z, track.width)
    return line, n


_CACHE: dict = {}


def cached_racing_line(track: Track, margin: float = 1.5):
    """racing_line(), computed once per track object."""
    key = (id(track), margin)
    if key not in _CACHE:
        _CACHE[key] = racing_line(track, margin)
    return _CACHE[key]
