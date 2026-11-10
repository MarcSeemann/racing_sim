"""Lap comparison — where one lap gains or loses time against another (phase 11).

`lap_trace` cuts one lap out of a telemetry log and puts it on a distance axis
along the track; `delta` gives time difference against distance, the standard
"delta" trace race engineers read to see where a driver or a setup change
gains time.
"""
import numpy as np
from scipy.spatial import cKDTree


def lap_window(log, lap_times, lap: int):
    """Boolean mask of the log rows that belong to lap number `lap` (0-based)."""
    ends = np.cumsum(lap_times)
    start = 0.0 if lap == 0 else ends[lap - 1]
    return (log["t"] > start) & (log["t"] <= ends[lap])


def lap_trace(log, lap_times, lap: int, track, grid_m: float = 5.0) -> dict:
    """One lap on a distance grid: elapsed time, speed, throttle, brake, steer vs distance (m)."""
    m = lap_window(log, lap_times, lap)
    x, y, t = log["x"][m], log["y"][m], log["t"][m]
    _, idx = cKDTree(np.c_[track.x, track.y]).query(np.c_[x, y])
    s = track.s[idx].astype(float)
    # Unwrap the lap (it starts just after the line): distance must increase.
    s[s < s[0] - track.length / 2] += track.length
    s = s - s[0]
    s = np.maximum.accumulate(s)
    grid = np.arange(0.0, s[-1], grid_m)
    out = {"s": grid, "t": np.interp(grid, s, t - t[0])}
    for key in ("v", "throttle", "brake", "steer"):
        if key in log._rows[0]:
            out[key] = np.interp(grid, s, log[key][m])
    return out


def delta(a: dict, b: dict) -> tuple:
    """(distance, time of b minus time of a) on their common distance range. Positive = b slower."""
    n = min(len(a["s"]), len(b["s"]))
    return a["s"][:n], b["t"][:n] - a["t"][:n]


class LiveDelta:
    """Running time gap to the best lap so far, as a lap is driven (for a live HUD)."""

    def __init__(self, track):
        self.track = track
        self.best = None  # time into the lap at each centreline index, best lap
        self.best_time = None
        self._cur = np.full(len(track.s), np.nan)
        self._start = None
        self._last_i = None

    def update(self, i: int, t: float):
        """Car at centreline index i at time t. Returns the gap to the best lap (s) or None."""
        if self._last_i is not None and i < self._last_i - len(self.track.s) // 2:  # crossed the line
            if self._start is not None:
                lap = t - self._start
                if self.best_time is None or lap < self.best_time:
                    filled = self._cur.copy()
                    ok = ~np.isnan(filled)
                    if ok.sum() > 10:
                        idx = np.arange(len(filled))
                        self.best = np.interp(idx, idx[ok], filled[ok])
                        self.best_time = lap
            self._start = t
            self._cur[:] = np.nan
        self._last_i = i
        if self._start is None:
            return None
        self._cur[i] = t - self._start
        return None if self.best is None else (t - self._start) - self.best[i]
