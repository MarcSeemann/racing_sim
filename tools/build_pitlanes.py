"""The pit lanes of the real circuits, from OpenStreetMap: carsim/data/tracks/pitlane/<key>.csv.

    python tools/build_pitlanes.py --sources /some/dir            # from the fetch of tools/build_runoff.py
    python tools/build_pitlanes.py --sources /some/dir --check    # only report, write nothing

Uses the OSM answers tools/build_runoff.py keeps in <sources>/osm/ (no new download).

How:
  1. Put the circuit on the map exactly as tools/build_runoff.py does (fit to the GeoJSON trace, refined
     onto OSM's raceway lines).
  2. The pit lane ways: `highway=raceway` lines named or tagged as a pit lane (incl. its entry and exit
     roads, any language), without other layouts' pit lanes (karting, national/support circuits...).
     Where none is named (Shanghai): unnamed raceway lines that run mostly off the track, close to it.
  3. Join them into routes the way OSM connects them (shared end nodes, in their one-way direction; two
     ways also join across a gap of up to LINK_M that carries on in both ways' direction: a piece not
     mapped as raceway, e.g. Monaco's lane and its exit road past Sainte Devote). Garage access roads and other branches stay separate routes.
  4. The pit lane is the longest route that runs alongside the start line (off the track, within NEAR_M
     of it, covering s = 0 give or take SLACK_M), in the race direction. Its ends are cut where it reaches
     the white line: that is where it leaves (entry) and joins (exit) the track.
  5. Cleaned and smoothed as a line (carsim.pitlane.clean_line), then put on the track as (s, offset).
     Where the fit puts it on our tarmac (map-traced circuits), it is moved out to WALL_M beyond the line.
Circuits without a mapped pit lane get none here: carsim.pitlane estimates one when asked.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_runoff import densify, place, refine, shapes_of  # noqa: E402
from carsim import pitlane  # noqa: E402
from carsim.circuits import _get_track  # noqa: E402
from carsim.f1_calendar import CALENDAR  # noqa: E402

PIT = re.compile(r"\bpit|pitstraat|boxes|boxen|bokszutca|\bstands\b")
OTHER = re.compile(r"kart|support|\bnational\b|west circuit|stowe|motogp|paddock")
NEAR_M = 80.0  # m from the centreline
OFF_M = 1.0  # m beyond the white line: closer is the track itself
LINK_M = 160.0  # m between the ends of two ways that still join, if the gap carries on in both directions
LINK_COS = 0.6  # cos of the largest angle (53 deg) between a way's direction at its end and the gap
START_M = 1000.0  # unnamed raceway lines: only those near the start line (median |s| below this)
SLACK_M = 100.0  # the start line may lie this far beyond the lane's mapped ends
WALL_M = 3.0  # m: the lane runs at least this far beyond the white line (but at its ends)
CLEAR_FADE_M = 50.0
UNNAMED_OFF_SHARE = 0.6  # unnamed raceway lines (no named pit lane): share of points off the track, close to it


class Way:
    def __init__(self, P, first, last, oneway):
        self.P, self.first, self.last, self.oneway = P, first, last, oneway
        self.length = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))

    def reversed(self):
        return Way(self.P[::-1], self.last, self.first, self.oneway)


def raceway_ways(osm: dict, to_xy):
    """(named pit lane ways, unnamed raceway ways), each a Way in the track frame, 1 m points."""
    named, unnamed = [], []
    for e in osm.get("elements", []):
        tg = e.get("tags", {})
        if e.get("type") != "way" or "geometry" not in e or tg.get("highway") != "raceway":
            continue
        name = " ".join(tg.get(k, "") for k in ("name", "name:en")).lower()
        if OTHER.search(name):
            continue
        P = densify(to_xy(np.array([[p["lon"], p["lat"]] for p in e["geometry"]])), 1.0)
        nodes = e.get("nodes") or [None, None]
        w = Way(P, nodes[0], nodes[-1], tg.get("oneway") == "yes")
        if tg.get("raceway") in ("pit_lane", "pitlane") or PIT.search(name):
            named.append(w)
        elif not name.strip():
            unnamed.append(w)
    return named, unnamed


def routes(ways):
    """Every route through the ways (each used once), following shared end nodes or ends within LINK_M."""
    # one-way ways only as drawn; two-way ways either way round
    opts = [w for w in ways] + [w.reversed() for w in ways if not w.oneway]
    ident = [i for i in range(len(ways))] + [i for i, w in enumerate(ways) if not w.oneway]

    def unit(v):
        return v / max(np.hypot(*v), 1e-9)

    def joins(a, b):
        if a.last is not None and a.last == b.first:
            return True
        gap = b.P[0] - a.P[-1]
        if np.hypot(*gap) < 3.0:
            return True
        if np.hypot(*gap) > LINK_M:
            return False
        g = unit(gap)
        end_a, start_b = unit(a.P[-1] - a.P[max(-6, -len(a.P))]), unit(b.P[min(5, len(b.P) - 1)] - b.P[0])
        return float(g @ end_a) > LINK_COS and float(g @ start_b) > LINK_COS

    nxt = [[j for j, b in enumerate(opts) if ident[j] != ident[i] and joins(a, b)] for i, a in enumerate(opts)]
    out = []

    def walk(path, used):
        out.append(list(path))
        if len(path) > 12:
            return
        for j in nxt[path[-1]]:
            if ident[j] not in used:
                walk(path + [j], used | {ident[j]})

    for i in range(len(opts)):
        walk([i], {ident[i]})
    return [np.concatenate([opts[j].P for j in p]) for p in out]


def beyond_edge(track, s, off):
    i = np.clip(np.searchsorted(track.s, s % track.length), 0, len(track.s) - 1)
    return np.abs(off) - np.where(off > 0, track.width_left[i], track.width_right[i])


def as_lane(track, P):
    """A route as a pit lane, or None: (s, offset, P) in the race direction, cut to where it is off the track."""
    s, off = pitlane.locate(track, P, NEAR_M)
    if s[-1] < s[0]:  # drawn against the race direction
        P = P[::-1]
        s, off = pitlane.locate(track, P, NEAR_M)
    L = track.length
    s = s - np.round(np.median(s) / L) * L  # the lap around the start line
    out = (beyond_edge(track, s, off) > OFF_M) & (np.abs(off) < NEAR_M)
    if out.sum() < 10:
        return None
    a, b = np.flatnonzero(out)[[0, -1]]
    a, b = max(a - 1, 0), min(b + 1, len(P) - 1)  # one point on the track: where it leaves / joins it
    if not (s[a] - SLACK_M <= 0.0 <= s[b] + SLACK_M):
        return None
    return P[a:b + 1], s[a], s[b]


def clear_of_track(track, lane):
    """Where a fitting error puts the lane on our tarmac (map-traced circuits: a few metres), move it out to
    WALL_M beyond the white line (a wall separates the real lane from the track), fading in over CLEAR_FADE_M
    from the ends and from where the lane crosses to the other side (Yas Marina's tunnel)."""
    side = np.sign(lane.offset)
    edge = np.abs(pitlane.edge_offset(track, 1, lane.s)) * (side > 0) + np.abs(pitlane.edge_offset(track, -1, lane.s)) * (side < 0)
    short = np.clip(edge + WALL_M - np.abs(lane.offset), 0, None)
    if not short.any():
        return lane
    marks = np.r_[0, np.flatnonzero(np.diff(side) != 0), len(side) - 1]
    dist = np.min(np.abs(np.arange(len(side))[:, None] - marks[None, :]), axis=1).astype(float)  # 1 m points
    fade = np.clip(dist / CLEAR_FADE_M, 0, 1)
    fade = 0.5 - 0.5 * np.cos(np.pi * fade)
    push = side * short * fade
    if not np.any(np.abs(push) > 0.01):
        return lane
    q = lane.s % track.length  # move those points straight out from the track, the rest stay as they are
    nx = -np.interp(q, track.s, np.sin(track.heading), period=track.length)
    ny = np.interp(q, track.s, np.cos(track.heading), period=track.length)
    x, y = lane.x + nx * push, lane.y + ny * push
    return pitlane.from_xy(track, lane.track, lane.source, x, y, lane.entry_s, lane.note)


def build_one(c, src: Path, check: bool, log=print):
    track = _get_track(f"f1:{c.key}")
    osm = json.loads((src / "osm" / f"{c.geo}.osm.json").read_text())
    local, R, t = place(track, c, src)
    A = np.c_[track.x, track.y]
    race = [densify(local(P)) for k, P, _ in shapes_of(osm) if k == "raceway"]
    if race:
        R, t, _, _ = refine(A, R, t, np.concatenate(race))
    named, unnamed = raceway_ways(osm, lambda ll: (local(ll) - t) @ R)  # inverse of A @ R.T + t
    how, ways = "named pit lane ways", named
    if not named:
        how = "unnamed raceway lines off the track"
        ways = []
        for w in unnamed:
            s, off = pitlane.locate(track, w.P, NEAR_M)
            s = (s + track.length / 2) % track.length - track.length / 2
            if np.median(np.abs(s)) > START_M or len(w.P) < 6:
                continue
            if np.mean((beyond_edge(track, s, off) > OFF_M) & (np.abs(off) < NEAR_M)) >= UNNAMED_OFF_SHARE:
                ways.append(w)
    if not ways:
        log(f"{c.key:12s} no pit lane mapped: carsim.pitlane estimates one")
        return None
    lanes = [x for x in (as_lane(track, P) for P in routes(ways)) if x is not None]
    if not lanes:
        log(f"{c.key:12s} pit lane ways found ({how}), but none alongside the start line: skipped")
        return None
    P, s0, s1 = max(lanes, key=lambda x: x[2] - x[1])
    x, y = pitlane.clean_line(P[:, 0], P[:, 1])
    lane = pitlane.from_xy(track, f"f1:{c.key}", "osm", x, y, s0, f"OpenStreetMap {c.geo}: {how}")
    lane = clear_of_track(track, lane)
    gap = beyond_edge(track, lane.s, lane.offset)
    log(f"{c.key:12s} {'left ' if lane.side > 0 else 'right'} entry {lane.entry_s:6.0f} m  exit {lane.exit_s:6.0f} m  "
        f"length {lane.length:5.0f} m  beyond the edge median {np.median(gap):4.1f} m (max {gap.max():4.1f})  "
        f"tightest radius {lane.min_radius():5.1f} m  [{how}]")
    if not check:
        pitlane.DATA.mkdir(parents=True, exist_ok=True)
        lane.save(pitlane.DATA / f"{c.key}.csv")
    return {"key": c.key, "side": lane.side, "entry_s": lane.entry_s, "exit_s": lane.exit_s, "length_m": lane.length,
            "median_gap_m": float(np.median(gap)), "min_radius_m": lane.min_radius(), "how": how}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--report", type=Path, help="also write the per-circuit summary as JSON")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    rows = []
    for c in CALENDAR:
        if only and c.key not in only:
            continue
        if not (a.sources / "osm" / f"{c.geo}.osm.json").exists():
            print(f"{c.key}: not fetched (tools/build_runoff.py --download)")
            continue
        r = build_one(c, a.sources, a.check)
        rows.append(r or {"key": c.key, "side": None})
    if a.report:
        a.report.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
