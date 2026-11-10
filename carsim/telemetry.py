"""Per-step telemetry log."""
import csv

import numpy as np


class Telemetry:
    def __init__(self):
        self._rows: list[dict] = []
        self._arrays: dict | None = None

    def log(self, row: dict):
        self._rows.append(row)
        self._arrays = None

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key: str) -> np.ndarray:
        if self._arrays is None:
            keys = self._rows[0].keys() if self._rows else []
            self._arrays = {k: np.array([r[k] for r in self._rows]) for k in keys}
        return self._arrays[key]

    def first_time(self, key: str, threshold: float) -> float | None:
        """First logged time at which `key` reaches `threshold`, linearly interpolated."""
        vals, t = self[key], self["t"]
        idx = np.nonzero(vals >= threshold)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        if i == 0:
            return float(t[0])
        frac = (threshold - vals[i - 1]) / (vals[i] - vals[i - 1])
        return float(t[i - 1] + frac * (t[i] - t[i - 1]))

    def to_csv(self, path: str):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self._rows[0].keys()))
            w.writeheader()
            w.writerows(self._rows)
