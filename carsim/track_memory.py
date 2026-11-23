"""What the driver sees far ahead, and what it knows about the circuit before it gets there.

Two kinds of driver input, both computed from the track alone (never from the car's future):

Look-ahead (sight): five windows that together cover the next HORIZON_S seconds at the current
speed (between HORIZON_MIN_M and HORIZON_MAX_M metres). The windows get longer the further out they
are: near the car the driver sees detail, far away only "how tight is the worst bit, and roughly
where". Each window gives the tightest curvature in it (signed) and the average curvature. Unlike
single points, a window cannot miss a short chicane that falls between two samples.

Track memory (knowledge): what a driver who has studied the circuit knows about the next corners,
however far away they are: distance to the next two significant corners, how tight they are, how
long the first one is, its width and gradient, and the straight after it (where the next big stop
is). Corners are found once per circuit (`corners`) and cached on the Track object.

    from carsim.track_memory import lookahead, memory
    lookahead(track, i, speed)   # 10 numbers
    memory(track, i)             # 8 numbers
"""
import numpy as np

HORIZON_S = 5.0  # s: how far ahead the driver looks, in time at the current speed
HORIZON_MIN_M = 60.0  # m: even when slow it looks at least this far
HORIZON_MAX_M = 550.0  # m
WINDOW_EDGES = (0.0, 0.08, 0.2, 0.38, 0.62, 1.0)  # fractions of the horizon: windows grow with distance
CORNER_RADIUS_M = 100.0  # a "significant corner" is tighter than this (needs braking at speed)
CORNER_MERGE_M = 40.0  # bends closer than this count as one corner (a chicane is one corner)
CORNER_MIN_LEN_M = 4.0
FAR_M = 2000.0  # distances are reported up to this (then "far")
CURV_SCALE = 30.0  # same scale as learn.py's curv_<d> inputs

LOOKAHEAD_NAMES = [f"look_{kind}_{w}" for w in range(1, len(WINDOW_EDGES)) for kind in ("max", "mean")]
MEMORY_NAMES = ["mem_c1_dist", "mem_c1_curv", "mem_c1_len", "mem_c1_width", "mem_c1_grade",
                "mem_straight_after", "mem_c2_dist", "mem_c2_curv"]


class _View:
    """Per-track arrays and corners, computed once."""

    def __init__(self, track):
        self.n = len(track.s)
        self.L = float(track.length)
        self.ds = self.L / self.n
        self.k = np.asarray(track.curvature, float)
        self.width = np.asarray(track._wl, float) + np.asarray(track._wr, float)
        self.grade = np.asarray(track.grade, float) if track.grade is not None else np.zeros(self.n)
        self.corners = corners(track, self.k, self.ds)
        starts = np.array([c["start"] for c in self.corners], float)
        self.starts_s = starts * self.ds
        # for every sample: the index (in self.corners) of the first corner that starts strictly ahead
        self.next_corner = (np.searchsorted(starts, np.arange(self.n), side="right") % max(len(starts), 1)
                            if len(starts) else None)


def _smooth(a, n):
    """Moving average over n samples, round the lap (surveyed curvature is noisy)."""
    if n <= 1:
        return a
    pad = np.concatenate([a[-n:], a, a[:n]])
    out = np.convolve(pad, np.ones(n) / n, mode="same")
    return out[n:-n]


def corners(track, k=None, ds=None) -> list:
    """Significant corners round the lap: list of dicts with start/end (sample index), peak (signed
    curvature, 1/m), length (m), width (narrowest, m), grade (mean). Chicanes count as one corner."""
    k = np.asarray(track.curvature if k is None else k, float)
    n = len(k)
    ds = track.length / n if ds is None else ds
    tight = _smooth(np.abs(k), max(int(round(9 / ds)), 1)) > 1.0 / CORNER_RADIUS_M
    if not tight.any():
        return []
    if tight.all():
        return [dict(start=0, end=n - 1, peak=float(k[np.argmax(np.abs(k))]), length=track.length,
                     width=float(np.min(np.asarray(track._wl) + np.asarray(track._wr))), grade=0.0)]
    # runs of "tight", starting the scan just after a straight sample so no run wraps the line
    first = int(np.argmin(tight))
    order = (np.arange(n) + first) % n
    runs, cur = [], None
    for i in order:
        if tight[i]:
            cur = [i, i] if cur is None else [cur[0], i]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    merge = int(round(CORNER_MERGE_M / ds))
    merged = [runs[0]]
    for a, b in runs[1:]:
        gap = (a - merged[-1][1]) % n
        if gap <= merge:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if len(merged) > 1 and (merged[0][0] - merged[-1][1]) % n <= merge:  # last joins first over the line
        merged[0][0] = merged.pop()[0]
    width = np.asarray(track._wl, float) + np.asarray(track._wr, float)
    grade = np.asarray(track.grade, float) if track.grade is not None else np.zeros(n)
    out = []
    for a, b in merged:
        idx = (np.arange((b - a) % n + 1) + a) % n
        if len(idx) * ds < CORNER_MIN_LEN_M:
            continue
        j = idx[np.argmax(np.abs(k[idx]))]
        out.append(dict(start=int(a), end=int(b), peak=float(k[j]), length=float(len(idx) * ds),
                        width=float(width[idx].min()), grade=float(grade[idx].mean())))
    out.sort(key=lambda c: c["start"])
    return out


def view(track) -> _View:
    v = getattr(track, "_memory_view", None)
    if v is None or v.n != len(track.s):
        v = _View(track)
        track._memory_view = v
    return v


def horizon(speed: float) -> float:
    return float(np.clip(speed * HORIZON_S, HORIZON_MIN_M, HORIZON_MAX_M))


def lookahead(track, i: int, speed: float) -> list:
    """Tightest (signed) and mean curvature in each look-ahead window, scaled like curv_<d>."""
    v = view(track)
    h = horizon(speed)
    out = []
    for a, b in zip(WINDOW_EDGES[:-1], WINDOW_EDGES[1:]):
        lo, hi = int(a * h / v.ds), max(int(b * h / v.ds), int(a * h / v.ds) + 1)
        seg = v.k[(np.arange(lo, hi) + i) % v.n]
        j = int(np.argmax(np.abs(seg)))
        out += [float(seg[j]) * CURV_SCALE, float(seg.mean()) * CURV_SCALE]
    return out


def memory(track, i: int) -> list:
    """The next two significant corners ahead of sample i, as a driver who knows the circuit would."""
    v = view(track)
    if not v.corners:
        return [FAR_M / 500.0, 0.0, 0.0, 0.0, 0.0, FAR_M / 500.0, FAR_M / 500.0, 0.0]
    c1i = int(v.next_corner[i])
    c2i = (c1i + 1) % len(v.corners)
    c1, c2 = v.corners[c1i], v.corners[c2i]
    s_i = i * v.ds

    def ahead(idx):  # metres from here to the start of corner idx, going forwards
        d = (v.corners[idx]["start"] * v.ds - s_i) % v.L
        return min(d, FAR_M)

    d1 = ahead(c1i)
    d2 = ahead(c2i) if len(v.corners) > 1 else d1 + v.L
    straight = (c2["start"] - c1["end"]) % v.n * v.ds if len(v.corners) > 1 else v.L - c1["length"]
    return [d1 / 500.0, c1["peak"] * CURV_SCALE, c1["length"] / 100.0, c1["width"] / 12.0,
            c1["grade"] * 10.0, min(straight, FAR_M) / 500.0, min(d2, FAR_M) / 500.0, c2["peak"] * CURV_SCALE]
