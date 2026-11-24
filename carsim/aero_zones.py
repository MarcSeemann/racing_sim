"""Aero zones: where the driver may open DRS (2025 car) or use straight mode (2026 car).

The FIA designates these per circuit on the long straights; here they are found from the track
shape: every stretch where the track is nearly straight (radius above STRAIGHT_R) for at least
MIN_LEN_M, starting ENTRY_M after the stretch begins. The car closes the wings itself when the
driver brakes (and at the end of the zone), as in the rules. Monaco has no stretch long enough,
which matches the real 2026 Monaco (no straight-mode zone).

Without other cars on track, DRS works as in qualifying: free use in the zones.

    from carsim.aero_zones import zones
    zones(track)   # bool per track sample, cached on the Track
"""
import numpy as np

STRAIGHT_R = 900.0  # m
MIN_LEN_M = 450.0
ENTRY_M = 60.0
SMOOTH_M = 40.0


def _find(track) -> np.ndarray:
    k = np.abs(np.asarray(track.curvature, float))
    n = len(k)
    ds = track.length / n
    w = max(int(SMOOTH_M / ds), 1)
    pad = np.concatenate([k[-w:], k, k[:w]])
    k = np.convolve(pad, np.ones(w) / w, mode="same")[w:-w]
    straight = k < 1.0 / STRAIGHT_R
    out = np.zeros(n, bool)
    if straight.all():
        return np.ones(n, bool)
    start = int(np.argmin(straight))  # scan from a curved sample so no run wraps the line
    run = []
    for j in range(n + 1):
        i = (start + j) % n
        if j < n and straight[i]:
            run.append(i)
            continue
        if len(run) * ds >= MIN_LEN_M:
            out[run[int(ENTRY_M / ds):]] = True
        run = []
    return out


def zones(track) -> np.ndarray:
    z = getattr(track, "_aero_zones", None)
    if z is None or len(z) != len(track.s):
        z = _find(track)
        track._aero_zones = z
    return z
