"""Pit lanes: where they leave the track, run alongside it, and join it again (for races later).

Nothing drives in them yet: this module only holds the geometry, so a race (pit stops, tyre
changes) can be built on it.

A pit lane is described relative to the track it belongs to: at every point, `s` (m along the lap
of the track sample it runs beside; may run past the lap's end or start below 0, i.e. before the
start line) and `offset` (m from the track centreline, + left, - right, like Track.lateral_offset).
It leaves the track at `entry_s` and joins it again at `exit_s`; at both ends the offset meets the
track edge on its side. `x`, `y` are the same points in the track's frame.

  get_pit_lane("f1:monza")    from data/tracks/pitlane/<key>.csv (OpenStreetMap, tools/build_pitlanes.py)
  get_pit_lane("random:3")    estimated(track): no mapped pit lane, so one is laid out the usual way

Estimated pit lanes (any circuit without a mapped one) run parallel to the start straight on the
inside of the lap, ESTIMATE_OFFSET_M beyond the track edge, from ESTIMATE_BEFORE_M before the line
to ESTIMATE_AFTER_M after it.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "tracks" / "pitlane"
SPEED_LIMIT_KMH = 80.0  # FIA default; per-circuit limits (some street circuits use 60) to be set before races
WIDTH_M = 12.0  # fast lane + working lane in front of the garages, nominal
ESTIMATE_OFFSET_M = 15.0
ESTIMATE_BEFORE_M, ESTIMATE_AFTER_M = 350.0, 250.0
BLEND_M = 60.0  # m over which an estimated lane moves away from / back to the track edge
SMOOTH_M = 4.0  # m, Gaussian smoothing of every lane along its length
ALIAS = {"monza": "monza", "sepang": "sepang"}  # named circuits that are also calendar circuits


@dataclass(frozen=True)
class PitLane:
    track: str
    source: str  # "osm" (mapped) or "estimated"
    side: int  # +1 left of the track, -1 right
    s: np.ndarray
    offset: np.ndarray
    x: np.ndarray
    y: np.ndarray
    note: str = ""
    speed_limit_kmh: float = SPEED_LIMIT_KMH
    width_m: float = WIDTH_M

    @property
    def entry_s(self) -> float:
        return float(self.s[0])

    @property
    def exit_s(self) -> float:
        return float(self.s[-1])

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    def min_radius(self, window_m: float = 5.0) -> float:
        """The lane's tightest bend (m), its heading change measured over window_m."""
        h = np.unwrap(np.arctan2(np.diff(self.y), np.diff(self.x)))
        ds = np.hypot(np.diff(self.x), np.diff(self.y))
        n = max(int(round(window_m)), 1)
        turn = np.abs(h[n:] - h[:-n])
        run = np.convolve(ds, np.ones(n), "valid")[1:len(turn) + 1]
        return float(np.min(run / np.maximum(turn, 1e-9))) if len(turn) else float("inf")

    def save(self, path: Path) -> None:
        meta = {"track": self.track, "source": self.source, "side": self.side, "note": self.note,
                "entry_s": round(self.entry_s, 1), "exit_s": round(self.exit_s, 1), "length_m": round(self.length, 1)}
        hdr = ("pit lane; built by tools/build_pitlanes.py (derived from OpenStreetMap, ODbL: see README.md)\n"
               + json.dumps(meta) + "\ns_m,offset_m,x_m,y_m")
        np.savetxt(path, np.c_[self.s, self.offset, self.x, self.y], fmt="%.2f", delimiter=",", header=hdr)


def edge_offset(track, side: int, s) -> np.ndarray:
    """The track's white line on `side` at s (m, any lap)."""
    s = np.asarray(s, float) % track.length
    w = track.width_left if side > 0 else track.width_right
    return side * np.interp(s, track.s, w, period=track.length)


def to_xy(track, s, offset):
    """Points (s along the lap, offset from the centreline) in the track's frame."""
    s = np.asarray(s, float) % track.length
    L = track.length
    x = np.interp(s, track.s, track.x, period=L)
    y = np.interp(s, track.s, track.y, period=L)
    c = np.interp(s, track.s, np.cos(track.heading), period=L)  # cos/sin: no 360 degree jump at the lap's end
    n = np.interp(s, track.s, np.sin(track.heading), period=L)
    r = np.hypot(c, n)
    return x - n / r * offset, y + c / r * offset


def locate(track, P, near: float = 80.0):
    """(s, offset) of the points of a line P (in order, track frame): s along the lap, continuous along the
    line (it may run below 0 or past the lap's end), offset + left. Of the track samples within `near`, each
    point takes the one nearest along the lap to the previous point's, so a line between two parallel parts
    of the track (Monaco's pit lane, between the pit straight and the harbour) does not jump between them."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    L, n_s = track.length, len(track.s)
    tree = cKDTree(np.c_[track.x, track.y])
    d, cand = tree.query(P, k=min(400, n_s), distance_upper_bound=near)
    i = tree.query(P)[1]
    for n in range(1, len(P)):
        ok = np.isfinite(d[n])
        if not ok.any():
            continue
        along = np.abs((track.s[cand[n][ok]] - track.s[i[n - 1]] + L / 2) % L - L / 2)
        if (along < 30.0).any():
            i[n] = cand[n][ok][np.argmin(np.where(along < 30.0, d[n][ok], np.inf))]
        else:
            i[n] = cand[n][ok][0]
    h = track.heading[i]
    off = (P[:, 0] - track.x[i]) * -np.sin(h) + (P[:, 1] - track.y[i]) * np.cos(h)
    s = (track.s[i] + L / 2) % L - L / 2
    s = s[0] + np.r_[0.0, np.cumsum((np.diff(s) + L / 2) % L - L / 2)]  # continuous along the line
    return s, off


def clean_line(x, y, tangent=None, sigma: float = SMOOTH_M):
    """A drivable line: points that step back against the track's direction (`tangent`, unit vectors per
    point: a fold on the inside of a tight corner) dropped, resampled every metre, Gaussian-smoothed along
    its length (ends kept in place)."""
    from scipy.ndimage import gaussian_filter1d
    P = np.c_[x, y]
    keep = [0]
    for n in range(1, len(P)):
        step = P[n] - P[keep[-1]]
        ahead = np.dot(step, tangent[n]) if tangent is not None else np.hypot(*step)
        if ahead >= 0.5:
            keep.append(n)
    if keep[-1] != len(P) - 1:
        keep[-1] = len(P) - 1  # the end stays where the lane joins the track
    P = P[keep]
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    t = np.linspace(0, seg[-1], max(int(seg[-1]), 2))
    P = np.c_[np.interp(t, seg, P[:, 0]), np.interp(t, seg, P[:, 1])]
    pad = int(4 * sigma)
    Q = np.r_[np.repeat(P[:1], pad, 0), P, np.repeat(P[-1:], pad, 0)]
    Q = gaussian_filter1d(Q, sigma, axis=0)[pad:-pad]
    w = np.clip(np.minimum(t, t[-1] - t) / (2 * sigma), 0, 1)[:, None]  # ends stay on the white line
    P = w * Q + (1 - w) * P
    return P[:, 0], P[:, 1]


def from_profile(track, name: str, source: str, side: int, s, offset, note: str = "") -> PitLane:
    """The lane through (s, offset) points, cleaned up as a line in the track frame (see clean_line)."""
    s = np.asarray(s, float)
    x, y = to_xy(track, s, np.asarray(offset, float))
    q = s % track.length
    tangent = np.c_[np.interp(q, track.s, np.cos(track.heading), period=track.length),
                    np.interp(q, track.s, np.sin(track.heading), period=track.length)]
    x, y = clean_line(x, y, tangent)
    return from_xy(track, name, source, x, y, float(s[0]), note)


def from_xy(track, name: str, source: str, x, y, entry_s: float, note: str = "") -> PitLane:
    """The lane through the points (x, y) in the race direction; `entry_s` (m along the lap, roughly) puts
    its s on the right lap. It is on the side where most of it lies."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s, off = locate(track, np.c_[x, y])
    s = s - round((s[0] - entry_s) / track.length) * track.length
    return PitLane(name, source, 1 if np.median(off) > 0 else -1, s, off, x, y, note)


def estimated(track, name: str | None = None) -> PitLane:
    """A pit lane laid out the usual way: alongside the start straight, on the inside of the lap."""
    turning = float(np.sum(np.diff(np.unwrap(track.heading))))
    side = 1 if turning > 0 else -1  # anticlockwise laps turn left: the inside is on the left
    s = np.arange(-ESTIMATE_BEFORE_M, ESTIMATE_AFTER_M + 0.5, 1.0)
    edge = edge_offset(track, side, s)
    ramp = np.clip(np.minimum(s - s[0], s[-1] - s) / BLEND_M, 0, 1)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)  # smooth in and out
    offset = edge + side * ESTIMATE_OFFSET_M * ramp
    return from_profile(track, name or track.name, "estimated", side, s, offset,
                        f"estimated: parallel to the start straight, {ESTIMATE_OFFSET_M:.0f} m beyond the edge")


def load(path: Path, track, name: str) -> PitLane:
    lines = path.read_text().splitlines()
    meta = json.loads(lines[1].lstrip("# "))
    d = np.loadtxt(path, delimiter=",", comments="#")
    return PitLane(name, meta["source"], int(meta["side"]), d[:, 0], d[:, 1], d[:, 2], d[:, 3], meta.get("note", ""))


def get_pit_lane(name: str, track=None) -> PitLane:
    """The pit lane of the circuit `name` (as carsim.circuits.get_track): mapped if there is one, else estimated."""
    if track is None:
        from .circuits import _get_track
        track = _get_track(name)
    key = name[3:] if name.startswith("f1:") else ALIAS.get(name)
    path = DATA / f"{key}.csv" if key else None
    if path is not None and path.exists():
        return load(path, track, name)
    return estimated(track, name)
