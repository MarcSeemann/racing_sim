"""Closed circuits — roadmap phase 8.

A track is a centreline sampled every ~1 m: position, heading, curvature,
elevation, grade and vertical curvature against distance s along the lap.
Build one from straights and arcs (`from_segments`), from surveyed points
(`from_points`, `from_csv` — e.g. a real circuit's x, y, z), and add elevation
with `with_elevation`.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    name: str
    s: np.ndarray  # distance along the centreline, m
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray  # rad
    curvature: np.ndarray  # 1/m, positive = left
    length: float
    width: float = 12.0
    z: np.ndarray | None = None  # elevation, m
    grade: np.ndarray | None = None  # slope dz/ds (rad, small-angle), positive = uphill
    vert_curv: np.ndarray | None = None  # d(grade)/ds, 1/m: positive = dip (compression), negative = crest
    kerb_width: float = 1.5  # m of kerb outside each track edge

    def __post_init__(self):
        if self.z is None:
            self.z = np.zeros_like(self.s)
        if self.grade is None or self.vert_curv is None:
            self._derive_vertical()

    def _derive_vertical(self):
        ds = np.gradient(self.s)
        dz = np.gradient(self.z)
        # Closed loop: wrap the ends.
        dz[0] = (self.z[1] - self.z[-1]) / 2
        dz[-1] = (self.z[0] - self.z[-2]) / 2
        self.grade = np.arctan(dz / ds)
        g = self.grade
        dg = np.gradient(g)
        dg[0] = (g[1] - g[-1]) / 2
        dg[-1] = (g[0] - g[-2]) / 2
        self.vert_curv = dg / ds

    def with_elevation(self, z_of_s, name: str | None = None) -> "Track":
        """Copy of this track with elevation z(s) (m). z_of_s must return to its start value."""
        z = np.array([z_of_s(v) for v in self.s])
        return Track(name or self.name, self.s, self.x, self.y, self.heading, self.curvature,
                     self.length, self.width, z, kerb_width=self.kerb_width)

    @classmethod
    def from_points(cls, name: str, x, y, z=None, width: float = 12.0, ds: float = 1.0, smooth: int = 7):
        """Closed track through surveyed centreline points (the loop closes from last to first).

        Points are resampled every `ds` m along the polyline, then heading and curvature are
        taken from a lightly smoothed (moving average over `smooth` samples) centreline.
        """
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.zeros_like(x) if z is None else np.asarray(z, float)
        xc, yc, zc = np.append(x, x[0]), np.append(y, y[0]), np.append(z, z[0])
        seg = np.hypot(np.diff(xc), np.diff(yc))
        sc = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(sc[-1])
        n = int(round(length / ds))
        s = np.linspace(0.0, length, n, endpoint=False)
        xs, ys, zs = (np.interp(s, sc, a) for a in (xc, yc, zc))
        if smooth > 1:
            k = np.ones(smooth) / smooth
            wrap = lambda a: np.convolve(np.concatenate([a[-smooth:], a, a[:smooth]]), k, "same")[smooth:-smooth]
            xs, ys, zs = wrap(xs), wrap(ys), wrap(zs)
        dx, dy = np.gradient(xs), np.gradient(ys)
        dx[0], dy[0] = (xs[1] - xs[-1]) / 2, (ys[1] - ys[-1]) / 2
        dx[-1], dy[-1] = (xs[0] - xs[-2]) / 2, (ys[0] - ys[-2]) / 2
        heading = np.unwrap(np.arctan2(dy, dx))
        dh = np.gradient(heading)
        dh[0] = math.remainder(heading[1] - heading[-1], 2 * math.pi) / 2
        dh[-1] = math.remainder(heading[0] - heading[-2], 2 * math.pi) / 2
        curvature = dh / np.hypot(dx, dy)
        return cls(name, s, xs, ys, heading, curvature, length, width, zs)

    @classmethod
    def from_samples(cls, name: str, x, y, z=None, width: float = 12.0):
        """Closed track through already-dense samples, keeping one sample per input point
        (used for racing lines, so line index i matches centreline index i)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.zeros_like(x) if z is None else np.asarray(z, float)
        seg = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
        s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        dx = (np.roll(x, -1) - np.roll(x, 1)) / 2
        dy = (np.roll(y, -1) - np.roll(y, 1)) / 2
        heading = np.unwrap(np.arctan2(dy, dx))
        dh = np.array([math.remainder(a, 2 * math.pi) for a in (np.roll(heading, -1) - np.roll(heading, 1)) / 1.0]) / 2
        curvature = dh / np.maximum(np.hypot(dx, dy), 1e-9)
        return cls(name, s, x, y, heading, curvature, float(seg.sum()), width, z)

    @classmethod
    def from_csv(cls, path: str, name: str | None = None, width: float = 12.0):
        """Load a centreline from a CSV with columns x, y and optionally z (metres, header row)."""
        data = np.genfromtxt(path, delimiter=",", names=True)
        z = data["z"] if "z" in data.dtype.names else None
        return cls.from_points(name or path, data["x"], data["y"], z, width)

    @classmethod
    def from_segments(cls, name: str, segments, ds: float = 1.0, width: float = 12.0):
        """segments: ("straight", length) or ("arc", radius, angle_deg) — angle > 0 turns left."""
        xs, ys, hs, ks = [0.0], [0.0], [0.0], [0.0]
        x = y = h = 0.0
        for seg in segments:
            if seg[0] == "straight":
                n, k = max(int(round(seg[1] / ds)), 1), 0.0
                step = seg[1] / n
            else:
                radius, ang = seg[1], math.radians(seg[2])
                length = radius * abs(ang)
                n = max(int(round(length / ds)), 1)
                step, k = length / n, math.copysign(1.0 / radius, ang)
            for _ in range(n):
                h_mid = h + k * step / 2
                x += step * math.cos(h_mid)
                y += step * math.sin(h_mid)
                h += k * step
                xs.append(x), ys.append(y), hs.append(h), ks.append(k)
        xs, ys = np.array(xs), np.array(ys)
        gap = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        if gap > 0.5 or abs(math.remainder(hs[-1], 2 * math.pi)) > 1e-3:
            raise ValueError(f"track {name!r} does not close: gap {gap:.2f} m")
        d = np.hypot(np.diff(xs), np.diff(ys))
        s = np.concatenate([[0.0], np.cumsum(d)])
        # Drop the duplicate closing point.
        return cls(name, s[:-1], xs[:-1], ys[:-1], np.array(hs[:-1]), np.array(ks[:-1]), float(s[-1]), width)

    def lateral_offset(self, i: int, x: float, y: float) -> float:
        return self.cross_track_error(i, x, y)

    def surface_at(self, offset: float) -> str:
        """'track', 'kerb' or 'grass' for a lateral offset from the centreline (m)."""
        edge = self.width / 2
        if abs(offset) <= edge:
            return "track"
        if abs(offset) <= edge + self.kerb_width:
            return "kerb"
        return "grass"

    def index_at(self, s: float) -> int:
        return int(np.searchsorted(self.s, s % self.length, side="right") - 1)

    def nearest(self, x: float, y: float, guess: int, window: int = 60) -> int:
        """Nearest centreline index, searching around a guess (cars move forward)."""
        n = len(self.s)
        idx = (np.arange(guess - 10, guess + window) % n)
        d2 = (self.x[idx] - x) ** 2 + (self.y[idx] - y) ** 2
        return int(idx[int(np.argmin(d2))])

    def cross_track_error(self, i: int, x: float, y: float) -> float:
        """Signed lateral offset from the centreline at index i, positive = left of it."""
        h = self.heading[i]
        return -(x - self.x[i]) * math.sin(h) + (y - self.y[i]) * math.cos(h)


def test_circuit() -> Track:
    """2.5 km test circuit, anticlockwise: a chicane on the main straight (left-right,
    then right-left), a hairpin (R 30), a fast right-angle (R 120), a medium corner
    (R 60) and a flat-out sweeper (R 200)."""
    r1, r2, r3, r4 = 30.0, 120.0, 60.0, 200.0
    bottom, right = 700.0, 250.0
    top = bottom + r1 - r2 - r3 + r4
    left = r1 + right + r2 - r3 - r4
    rc, ac = 50.0, 40.0  # chicane arcs; the two S-bends cancel sideways
    s_len = 2 * rc * math.sin(math.radians(ac))
    pre, mid = 250.0, 100.0
    post = bottom - pre - mid - 2 * s_len
    return Track.from_segments("Test circuit", [
        ("straight", pre), ("arc", rc, ac), ("arc", rc, -ac), ("straight", mid),
        ("arc", rc, -ac), ("arc", rc, ac), ("straight", post),
        ("arc", r1, 90), ("straight", right), ("arc", r2, 90),
        ("straight", top), ("arc", r3, 90), ("straight", left), ("arc", r4, 90),
    ])


def hill_circuit() -> Track:
    """The test circuit with elevation: a long climb and descent (about 23 m), a crest
    on the flat-out top straight (s ~ 1600 m, the car goes light) and a compression
    in the R 200 sweeper (s ~ 2300 m, the car is pressed into the track)."""
    base = test_circuit()
    L = base.length
    bump = lambda s, s0, h, w: h * math.exp(-((s - s0) / w) ** 2)
    z = lambda s: (9.0 * math.sin(2 * math.pi * s / L - 0.6) + 3.0 * math.sin(6 * math.pi * s / L)
                   + bump(s, 1600.0, 1.5, 80.0) + bump(s, 2300.0, -1.0, 60.0))
    return base.with_elevation(z, name="Hill circuit")


def short_circuit() -> Track:
    """1.1 km club circuit (anticlockwise): hairpin, two medium corners, a fast one.
    Short laps for quick experiments and tests."""
    r1, r2, r3, r4 = 25.0, 60.0, 40.0, 80.0
    bottom, right = 300.0, 120.0
    top = bottom + r1 - r2 - r3 + r4
    left = r1 + right + r2 - r3 - r4
    return Track.from_segments("Short circuit", [
        ("straight", bottom), ("arc", r1, 90), ("straight", right), ("arc", r2, 90),
        ("straight", top), ("arc", r3, 90), ("straight", left), ("arc", r4, 90),
    ])
