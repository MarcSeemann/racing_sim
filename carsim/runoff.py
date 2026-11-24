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
"""
import numpy as np

STREET = {"f1:monaco", "f1:singapore", "f1:baku", "f1:jeddah", "f1:las_vegas"}
WALL_M = 1.5
SLOW_R, FAST_R = 70.0, 400.0  # m
ASPHALT_M, GRAVEL_M, GRASS_M, INSIDE_M = 25.0, 30.0, 12.0, 6.0
BEFORE_M, AFTER_M = 60.0, 120.0
SMOOTH_M = 30.0


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


def with_runoff(track, name: str = ""):
    """A copy of `track` with run-off and barriers (street circuits: walls)."""
    import dataclasses
    rl, rr, kl, kr = layout(track, street=name in STREET)
    out = dataclasses.replace(track, runoff_left=rl, runoff_right=rr, runoff_kind_left=kl, runoff_kind_right=kr,
                              grade=track.grade, vert_curv=track.vert_curv)
    for attr in ("altitude_m",):  # keep what the loaders attached
        if hasattr(track, attr):
            setattr(out, attr, getattr(track, attr))
    return out
