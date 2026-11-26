"""Run-off areas and barriers beyond the kerbs: what happens when the car leaves the track.

The circuit data has white lines and kerbs but no run-off or barrier positions, so they are
estimated the way circuits are laid out (FIA Grade 1 practice, roughly):

  street circuits (STREET): a wall WALL_M beyond the kerb, on both sides, all the way round
  permanent circuits, from the local corner speed (radius of the smoothed curvature):
    outside of a slow corner (R < SLOW_R)            asphalt, ASPHALT_M wide (modern chicanes, hairpins)
    outside of a medium/fast corner (R < FAST_R)     gravel, GRAVEL_M wide (catches a car that runs wide)
    straights, and the inside of every corner        grass, GRASS_M wide
  The "outside" zone starts BEFORE_M before the corner and runs AFTER_M past it (a car that
  misses its braking point goes straight on into it).

The estimates only decide what the car meets when it goes off; the rules for track limits are in
learn.run_episode (all four wheels beyond the white line = a violation; off for more than
3 s = disqualified; a barrier = a crash).

Kerbs sit only where a car uses them: through corners (smoothed radius below KERB_R, KERB_PAD_M
either side), not along the straights. On the real circuits the estimate is replaced by what is
mapped (`data/tracks/runoff/<key>.npz`, built from OpenStreetMap by tools/build_runoff.py): gravel
traps, grass, asphalt run-off and the barriers, metre by metre beyond the white line.
"""
from pathlib import Path

import numpy as np

STREET = {"f1:monaco", "f1:singapore", "f1:baku", "f1:jeddah", "f1:las_vegas"}
WALL_M = 1.5
SLOW_R, FAST_R = 70.0, 400.0  # m
ASPHALT_M, GRAVEL_M, GRASS_M, INSIDE_M = 25.0, 30.0, 12.0, 6.0
BEFORE_M, AFTER_M = 60.0, 120.0
SMOOTH_M = 30.0
KERB_R, KERB_PAD_M = 350.0, 15.0  # kerbs where the smoothed radius is below KERB_R (m), extended KERB_PAD_M
PROFILE_M = 60  # metres of surface stored beyond the white line
CODE = {"asphalt": 0, "grass": 1, "gravel": 2, "barrier": 3}  # = track.SURFACES
DATA = Path(__file__).resolve().parent / "data" / "tracks" / "runoff"


def _smooth(a, n):
    if n <= 1:
        return a
    pad = np.concatenate([a[-n:], a, a[:n]])
    return np.convolve(pad, np.ones(n) / n, mode="same")[n:-n]


def layout(track, street: bool = False):
    """(room_left, room_right, kind_left, kind_right) arrays for `track`."""
    n = len(track.s)
    if street:
        room = np.full(n, WALL_M)
        return room, room.copy(), ["asphalt"] * n, ["asphalt"] * n
    ds = track.length / n
    k = _smooth(np.asarray(track.curvature, float), max(int(SMOOTH_M / ds), 1))
    room_l, room_r = np.full(n, GRASS_M), np.full(n, GRASS_M)
    kind_l, kind_r = ["grass"] * n, ["grass"] * n
    before, after = int(BEFORE_M / ds), int(AFTER_M / ds)
    # the outside of each corner: turning left (k > 0) the car runs wide to the right
    order = np.argsort(-np.abs(k))  # tightest first, so a slow corner's asphalt wins over gravel
    done_l, done_r = np.zeros(n, bool), np.zeros(n, bool)
    for i in order:
        r = 1.0 / abs(k[i]) if abs(k[i]) > 1e-9 else np.inf
        if r >= FAST_R:
            break
        kind, width = ("asphalt", ASPHALT_M) if r < SLOW_R else ("gravel", GRAVEL_M)
        idx = (np.arange(i - before, i + after + 1)) % n
        if k[i] > 0:  # left-hander: outside is the right
            new = idx[~done_r[idx]]
            room_r[new] = width
            for j in new:
                kind_r[j] = kind
            done_r[new] = True
            fresh = idx[~done_l[idx]]
            room_l[fresh] = INSIDE_M
            done_l[fresh] = True
        else:
            new = idx[~done_l[idx]]
            room_l[new] = width
            for j in new:
                kind_l[j] = kind
            done_l[new] = True
            fresh = idx[~done_r[idx]]
            room_r[fresh] = INSIDE_M
            done_r[fresh] = True
    return room_l, room_r, kind_l, kind_r


def kerbs(track) -> np.ndarray:
    """Where the kerbs are (bool per sample, both sides): through the corners."""
    n = len(track.s)
    ds = track.length / n
    k = np.abs(_smooth(np.asarray(track.curvature, float), max(int(SMOOTH_M / ds), 1)))
    m = k > 1.0 / KERB_R
    pad = max(int(KERB_PAD_M / ds), 1)
    return np.convolve(np.concatenate([m[-pad:], m, m[:pad]]).astype(float), np.ones(2 * pad + 1), "same")[pad:-pad] > 0


def profile_from_layout(track, room, kind, kerb_w: float) -> np.ndarray:
    """Surface codes metre by metre beyond the white line from a (room beyond the kerb, kind) layout."""
    j = np.arange(PROFILE_M)[None, :] + 0.5
    codes = np.array([CODE[k] for k in kind], np.uint8)[:, None].repeat(PROFILE_M, 1)
    codes[j >= kerb_w + np.asarray(room)[:, None]] = CODE["barrier"]
    return codes


def mapped_profile(key: str, n: int):
    """(surf_left, surf_right) from data/tracks/runoff/<key>.npz, or None (not mapped / other sampling)."""
    f = DATA / f"{key}.npz"
    if not f.exists():
        return None
    with np.load(f) as d:
        left, right = d["left"], d["right"]
    if len(left) != n:
        return None
    return left, right


def _room_kind(prof, kerb, kerb_w):
    """Compatibility: run-off width beyond the kerb before the barrier, and its first surface."""
    bar = prof == CODE["barrier"]
    first = np.where(bar.any(1), bar.argmax(1), PROFILE_M + 40).astype(float)
    room = np.maximum(first - kerb_w, 0.5)
    names = {v: k for k, v in CODE.items()}
    j0 = min(int(np.ceil(kerb_w)), PROFILE_M - 1)
    kind = [names[int(c)] if c != CODE["barrier"] else "asphalt" for c in prof[:, j0]]
    return room, kind


def calendar_key(name: str) -> str | None:
    if name.startswith("f1:"):
        return name[3:]
    return name if name in ("monza", "sepang") else None


def with_runoff(track, name: str = ""):
    """A copy of `track` with run-off and barriers (street circuits: walls), kerbs only in corners,
    and the mapped surfaces where a real circuit has them."""
    import dataclasses
    rl, rr, kl, kr = layout(track, street=name in STREET)
    kerb = kerbs(track)
    kw = track.kerb_width
    key = calendar_key(name)
    mapped = mapped_profile(key, len(track.s)) if key else None
    if mapped is not None:
        pl, pr = mapped
        rl, kl = _room_kind(pl, kerb, kw)
        rr, kr = _room_kind(pr, kerb, kw)
    else:
        pl, pr = profile_from_layout(track, rl, kl, kw), profile_from_layout(track, rr, kr, kw)
    out = dataclasses.replace(track, runoff_left=rl, runoff_right=rr, runoff_kind_left=kl, runoff_kind_right=kr,
                              kerb_left=kerb, kerb_right=kerb.copy(), surf_left=pl, surf_right=pr,
                              grade=track.grade, vert_curv=track.vert_curv)
    for attr in ("altitude_m",):  # keep what the loaders attached
        if hasattr(track, attr):
            setattr(out, attr, getattr(track, attr))
    return out
