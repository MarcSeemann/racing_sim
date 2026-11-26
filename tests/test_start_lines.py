"""Every calendar circuit starts (s = 0) on its real start/finish line, which runs alongside the pit lane."""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from carsim.circuits import get_track
from carsim.f1_calendar import CALENDAR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_f1_2026 import R_EARTH, align, geo_line  # noqa: E402

# The raw OpenStreetMap fetch of tools/build_runoff.py (not in the repo: ODbL source data).
OSM = Path(os.environ.get("CARSIM_OSM_SOURCES", Path.home() / ".cache/carsim/osm-sources")) / "osm"
NO_PIT_LANE_IN_OSM = {"melbourne", "shanghai", "silverstone", "zandvoort"}  # none mapped near the track
NEAR_M = 60.0  # the part of a pit lane within this distance of the centreline runs alongside it
SLACK_M = 100.0  # the line may lie a little beyond the mapped pit lane's end


def _wrap(s, L):
    return (s + L / 2) % L - L / 2  # signed distance along the lap from the start


def _pit_lanes(osm: dict):
    for e in osm.get("elements", []):
        tg = e.get("tags", {})
        name = (tg.get("name:en") or tg.get("name") or "").lower()
        pit = tg.get("raceway") in ("pit_lane", "pitlane") or re.search(r"pit ?lane|stands|boxen|bokszutca", name)
        if not pit or e.get("type") != "way" or "geometry" not in e:
            continue
        if tg.get("sport") == "karting" or re.search(r"support|west circuit|national|entrada|sortida", name):
            continue  # other layouts' pit lanes, entry/exit roads
        yield np.array([[p["lon"], p["lat"]] for p in e["geometry"]])


def pit_lane_span(c):
    """(from, to): where the pit lane runs alongside the track, in m along the lap from the start (5-95 %)."""
    f = OSM / f"{c.geo}.geojson"
    tr = get_track("f1:" + c.key)
    T, L = np.c_[tr.x, tr.y], float(tr.length)
    tree = cKDTree(T)
    R, t = align(geo_line(f, c.reverse), T)  # GeoJSON frame -> our track
    g = np.asarray(json.loads(f.read_text())["features"][0]["geometry"]["coordinates"], float)[:, :2]
    if np.allclose(g[0], g[-1]):
        g = g[:-1]
    lon0, lat0 = g[:, 0].mean(), g[:, 1].mean()  # the frame of geo_line
    s = []
    for ll in _pit_lanes(json.loads((OSM / f"{c.geo}.osm.json").read_text())):
        P = np.c_[np.radians(ll[:, 0] - lon0) * R_EARTH * np.cos(np.radians(lat0)),
                  np.radians(ll[:, 1] - lat0) * R_EARTH]
        d, i = tree.query(P @ R.T + t)
        if np.sum(d < NEAR_M) >= 2:
            s.append(_wrap(tr.s[i[d < NEAR_M]], L))
    if not s:
        return None
    s = np.concatenate(s)
    return float(np.percentile(s, 5)), float(np.percentile(s, 95))


@pytest.mark.parametrize("c", [c for c in CALENDAR if c.key not in NO_PIT_LANE_IN_OSM], ids=lambda c: c.key)
def test_the_start_is_alongside_the_pit_lane(c):
    if not (OSM / f"{c.geo}.osm.json").exists():
        pytest.skip(f"no OpenStreetMap fetch in {OSM} (tools/build_runoff.py --download)")
    span = pit_lane_span(c)
    assert span is not None, "no pit lane found near the track"
    lo, hi = span
    assert lo - SLACK_M <= 0.0 <= hi + SLACK_M, f"start {min(abs(lo), abs(hi)):.0f} m from the pit lane ({lo:.0f}..{hi:.0f} m)"


def _corners(tr, max_radius):
    """(s at the tightest point, radius, 'left'/'right') of every corner tighter than max_radius."""
    h = np.unwrap(np.arctan2(np.gradient(tr.y), np.gradient(tr.x)))
    k = np.gradient(h) / np.gradient(tr.s)
    tight = np.abs(k) > 1 / max_radius
    out, i = [], 0
    while i < len(k):
        if tight[i]:
            j = i
            while j < len(k) and tight[j]:
                j += 1
            m = i + int(np.argmax(np.abs(k[i:j])))
            out.append((float(tr.s[m]), 1 / abs(k[m]), "left" if k[m] > 0 else "right"))
            i = j
        else:
            i += 1
    return out


def test_monaco_starts_on_the_pit_straight():
    """The line is ~150 m before Sainte Devote, the hairpin ~1.2 km into the lap (it was 650 m too far on, at Casino)."""
    tr = get_track("f1:monaco")
    first = _corners(tr, 50.0)[0]
    assert 100 < first[0] < 250 and first[2] == "right" and first[1] < 15  # Sainte Devote
    hairpin = min((c for c in _corners(tr, 50.0) if c[0] < 1500), key=lambda c: c[1])
    assert 1100 < hairpin[0] < 1300 and hairpin[2] == "left"  # Grand Hotel hairpin
