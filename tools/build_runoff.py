"""What lies beyond the white lines of the real circuits, from OpenStreetMap: data/tracks/runoff/<key>.npz.

    python tools/build_runoff.py --sources /some/dir --download     # fetch from OSM + build
    python tools/build_runoff.py --sources /some/dir                # rebuild from the fetch
    python tools/build_runoff.py --sources /some/dir --check        # only report, write nothing

How:
  1. Put the circuit on the map: our centreline is fitted onto its bacinger/f1-circuits GeoJSON trace
     (rigid ICP, as tools/build_elevation.py), then refined onto OSM's own `highway=raceway` lines
     (the mapped tarmac), so a few metres of trace error do not turn the kerb into the gravel trap.
  2. Fetch the circuit's surroundings from the Overpass API (one query per circuit, generic user
     agent, backoff): gravel/sand traps (`natural=sand`, `surface=gravel|sand`), grass (`landuse=grass`,
     `surface=grass`, ...), mapped asphalt areas, and barriers (`barrier=wall|guard_rail|fence|...`).
     Only the circuit's bounding box is sent. Raw answers are kept in <sources>/osm/.
  3. Walk out from the white line, metre by metre (PROFILE_M), at every centreline sample, on both
     sides: the mapped surface there, and the barrier from the first barrier line crossed.
     Unmapped ground: a gap of up to GAP_M before the first mapped surface takes that surface
     (mapping is not metre-exact); a wider unmapped band next to the track is paved run-off
     (asphalt). Samples where nothing at all is mapped keep carsim.runoff's estimate.
  4. A running mode along the lap (SMOOTH_M) removes single-sample speckle.

OSM data is (c) OpenStreetMap contributors, ODbL 1.0: the .npz files are a derived database under
the ODbL (see data/tracks/README.md).
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_f1_2026 import R_EARTH, SOURCES, align, geo_line  # noqa: E402
from carsim import runoff  # noqa: E402
from carsim.circuits import _get_track  # noqa: E402
from carsim.f1_calendar import CALENDAR  # noqa: E402

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = {"User-Agent": "carsim-track-builder/0.1"}
PROFILE_M = runoff.PROFILE_M
GAP_M = 3  # m of unmapped ground before a mapped surface still counts as that surface
MIN_BARRIER_M = 2.0  # m beyond the line: a barrier line closer than this is a fitting error (permanent circuits)
SMOOTH_M = 9  # samples (m) of the running mode along the lap
SNAP_M = 25.0  # m: raceway points this close to our centreline refine the fit
OUT = runoff.DATA
GRAVEL, GRASS, ASPHALT, BARRIER, NONE = (runoff.CODE["gravel"], runoff.CODE["grass"], runoff.CODE["asphalt"],
                                         runoff.CODE["barrier"], 255)


def query(bbox: str) -> str:
    b = bbox
    return (f'[out:json][timeout:120];(way["surface"]({b});way["natural"~"sand|scrub|grassland"]({b});'
            f'way["barrier"]({b});way["area:highway"]({b});way["highway"="raceway"]({b});'
            f'way["landuse"~"grass|meadow"]({b});relation["surface"]({b});relation["natural"="sand"]({b});'
            f'relation["landuse"~"grass|meadow"]({b}););out geom;')


def fetch(src: Path, c, log=print):
    """<sources>/osm/<geo>.geojson and <geo>.osm.json (downloaded once)."""
    d = src / "osm"
    d.mkdir(parents=True, exist_ok=True)
    gp = d / f"{c.geo}.geojson"
    if not gp.exists():
        gp.write_bytes(urllib.request.urlopen(urllib.request.Request(SOURCES["geo"].format(c.geo), headers=UA),
                                              timeout=60).read())
    dst = d / f"{c.geo}.osm.json"
    if dst.exists():
        return
    co = json.loads(gp.read_text())["features"][0]["geometry"]["coordinates"]
    lo, la = [p[0] for p in co], [p[1] for p in co]
    bbox = f"{min(la) - 0.003},{min(lo) - 0.005},{max(la) + 0.003},{max(lo) + 0.005}"
    wait = 10
    for k in range(10):
        ep = OVERPASS[k % len(OVERPASS)]
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": query(bbox)}).encode(), headers=UA)
            data = json.loads(urllib.request.urlopen(req, timeout=180).read())
            dst.write_text(json.dumps(data))
            log(f"{c.key}: {len(data['elements'])} OSM elements")
            time.sleep(5)
            return
        except Exception as e:  # busy server: back off
            log(f"{c.key}: retry {k} ({type(e).__name__})")
            time.sleep(wait)
            wait = min(2 * wait, 120)
    raise RuntimeError(f"{c.key}: Overpass gave up")


def kind_of(tags: dict):
    if tags.get("natural") == "sand" or tags.get("surface") in ("sand", "gravel", "fine_gravel", "pebblestone"):
        return "gravel"
    if tags.get("barrier") in ("wall", "guard_rail", "fence", "jersey_barrier", "retaining_wall", "block", "tyres"):
        return "barrier"
    if (tags.get("landuse") in ("grass", "meadow") or tags.get("surface") in ("grass", "artificial_turf")
            or tags.get("natural") in ("grassland", "scrub")):
        return "grass"
    if tags.get("highway") == "raceway":
        return "raceway"
    if tags.get("surface") in ("asphalt", "paved", "concrete") and (tags.get("area") == "yes" or "area:highway" in tags):
        return "asphalt"
    return None


def _rings(members):
    """Join a multipolygon's outer member ways into closed rings (lists of (lon, lat))."""
    parts = [[(p["lon"], p["lat"]) for p in m["geometry"]] for m in members
             if m.get("role") == "outer" and m.get("geometry")]
    rings = []
    while parts:
        cur = parts.pop(0)
        changed = True
        while cur[0] != cur[-1] and changed:
            changed = False
            for i, p in enumerate(parts):
                if p[0] == cur[-1]:
                    cur += p[1:]
                elif p[-1] == cur[-1]:
                    cur += p[::-1][1:]
                elif p[-1] == cur[0]:
                    cur = p[:-1] + cur
                elif p[0] == cur[0]:
                    cur = p[::-1][:-1] + cur
                else:
                    continue
                parts.pop(i)
                changed = True
                break
        if cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def shapes_of(osm: dict):
    """[(kind, [(lon, lat), ...], closed)] for every useful way / multipolygon ring."""
    out = []
    for e in osm["elements"]:
        k = kind_of(e.get("tags", {}))
        if not k:
            continue
        if e["type"] == "way" and e.get("geometry"):
            pts = [(p["lon"], p["lat"]) for p in e["geometry"]]
            out.append((k, pts, pts[0] == pts[-1] and len(pts) > 3))
        elif e["type"] == "relation":
            out += [(k, r, True) for r in _rings(e.get("members", []))]
    return out


def place(track, c, src: Path):
    """Track frame <- lon/lat: (to_track function, median misfit to OSM raceway lines after refining, m)."""
    gp = src / "osm" / f"{c.geo}.geojson"
    co = np.asarray(json.loads(gp.read_text())["features"][0]["geometry"]["coordinates"], float)[:, :2]
    if np.allclose(co[0], co[-1]):
        co = co[:-1]
    lon0, lat0 = co[:, 0].mean(), co[:, 1].mean()
    A = np.c_[track.x, track.y]
    R, t = align(A, geo_line(gp, c.reverse))

    def local(ll):
        ll = np.asarray(ll, float)
        return np.c_[np.radians(ll[:, 0] - lon0) * R_EARTH * np.cos(np.radians(lat0)), np.radians(ll[:, 1] - lat0) * R_EARTH]

    return local, R, t


def refine(A, R, t, raceway_pts):
    """Rigid ICP of our centreline onto nearby mapped raceway points (trimmed); (R, t, misfit before, after)."""
    if len(raceway_pts) < 50:
        return R, t, np.nan, np.nan
    tree = cKDTree(raceway_pts)
    d0 = np.median(tree.query(A @ R.T + t)[0])
    for _ in range(30):
        P = A @ R.T + t
        d, j = tree.query(P)
        m = d < SNAP_M
        if m.sum() < 0.5 * len(A):
            break
        pa, pb = A[m], raceway_pts[j[m]]
        ca, cb = pa.mean(0), pb.mean(0)
        U, _, Vt = np.linalg.svd((pa - ca).T @ (pb - cb))
        D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R2 = Vt.T @ D @ U.T
        R, t = R2, cb - ca @ R2.T
    d1 = np.median(tree.query(A @ R.T + t)[0])
    return R, t, float(d0), float(d1)


def densify(P, step=2.0):
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out)


def _mode(a, n):
    """Running mode over n samples along a closed lap, per column (codes < 256)."""
    out = a.copy()
    h = n // 2
    pad = np.concatenate([a[-h:], a, a[:h]])
    for j in range(a.shape[1]):
        cnt = np.zeros((len(a), 256), np.int16)
        for k in range(n):
            cnt[np.arange(len(a)), pad[k:k + len(a), j]] += 1
        out[:, j] = cnt.argmax(1)
    return out


def profiles(track, shapes_track, street: bool):
    """(left, right) uint8 code arrays (samples x PROFILE_M), plus the share of samples mapped per side."""
    n = len(track.s)
    nx, ny = -np.sin(track.heading), np.cos(track.heading)
    polys = [(k, P, MPath(P), P.min(0), P.max(0)) for k, P, closed in shapes_track
             if closed and k in ("gravel", "grass", "asphalt")]
    segs = [np.c_[P[:-1], P[1:]] for k, P, closed in shapes_track if k == "barrier" and len(P) > 1]
    segs = np.concatenate(segs) if segs else np.zeros((0, 4))
    out, mapped = [], []
    j = np.arange(PROFILE_M) + 0.5
    est = runoff.layout(track, street=street)
    for side, w, room, kind in ((1, track.width_left, est[0], est[2]), (-1, track.width_right, est[1], est[3])):
        px = track.x[:, None] + side * (w[:, None] + j[None, :]) * nx[:, None]
        py = track.y[:, None] + side * (w[:, None] + j[None, :]) * ny[:, None]
        pts = np.c_[px.ravel(), py.ravel()]
        lab = np.full(len(pts), NONE, np.uint8)
        for want in ("asphalt", "grass", "gravel"):  # later wins: a gravel trap mapped inside a grass area
            code = runoff.CODE[want]
            for k, P, path, lo, hi in polys:
                if k != want:
                    continue
                sel = np.flatnonzero((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
                if len(sel):
                    lab[sel[path.contains_points(pts[sel])]] = code
        lab = lab.reshape(n, PROFILE_M)
        # first barrier crossed walking out from the line
        a = np.c_[track.x + side * w * nx, track.y + side * w * ny]
        r = np.c_[side * PROFILE_M * nx, side * PROFILE_M * ny]
        bar = np.full(n, np.inf)
        if len(segs):
            p1, s_ = segs[:, :2], segs[:, 2:] - segs[:, :2]
            for i0 in range(0, n, 256):
                aa, rr = a[i0:i0 + 256, None, :], r[i0:i0 + 256, None, :]
                den = rr[..., 0] * s_[None, :, 1] - rr[..., 1] * s_[None, :, 0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    tt = ((p1[None, :, 0] - aa[..., 0]) * s_[None, :, 1] - (p1[None, :, 1] - aa[..., 1]) * s_[None, :, 0]) / den
                    uu = ((p1[None, :, 0] - aa[..., 0]) * rr[..., 1] - (p1[None, :, 1] - aa[..., 1]) * rr[..., 0]) / den
                d = np.where((tt > 0) & (tt <= 1) & (uu >= 0) & (uu <= 1), tt * PROFILE_M, np.inf)
                d[d < (0.5 if street else MIN_BARRIER_M)] = np.inf
                bar[i0:i0 + 256] = d.min(1)
        has = (lab != NONE).any(1) | np.isfinite(bar)
        prof = runoff.profile_from_layout(track, room, kind, track.kerb_width)
        for i in np.flatnonzero(has):
            row = lab[i].copy()
            m = np.flatnonzero(row != NONE)
            if len(m):
                first = m[0]
                row[:first] = row[first] if first <= GAP_M else ASPHALT
                last = row[first]
                for q in range(first, PROFILE_M):  # carry the last mapped surface outwards over gaps
                    if row[q] == NONE:
                        row[q] = last
                    else:
                        last = row[q]
            else:
                row[:] = ASPHALT if street else GRASS  # nothing mapped before the barrier
            if np.isfinite(bar[i]):
                row[int(bar[i]):] = BARRIER
            elif street:
                row[int(track.kerb_width + runoff.WALL_M):] = BARRIER
            prof[i] = row
        out.append(_mode(prof, SMOOTH_M))
        mapped.append(float(has.mean()))
    return out[0], out[1], mapped


def build_one(c, src: Path, check: bool, log=print):
    track = _get_track(f"f1:{c.key}")
    local, R, t = place(track, c, src)
    sh = shapes_of(json.loads((src / "osm" / f"{c.geo}.osm.json").read_text()))
    raw = [(k, local(pts), closed) for k, pts, closed in sh]
    A = np.c_[track.x, track.y]
    race = [densify(P) for k, P, _ in raw if k == "raceway"]
    race = np.concatenate(race) if race else np.zeros((0, 2))
    R, t, d0, d1 = refine(A, R, t, race)
    to_track = lambda P: (P - t) @ R  # inverse of A @ R.T + t
    shapes_track = [(k, to_track(P), closed) for k, P, closed in raw]
    left, right, mapped = profiles(track, shapes_track, street=f"f1:{c.key}" in runoff.STREET)
    share = {name: float(np.mean(np.r_[left[:, 2], right[:, 2]] == code)) for name, code in runoff.CODE.items()}
    log(f"{c.key:12s} fit to OSM raceway {d0:4.1f} -> {d1:4.1f} m, mapped L {mapped[0]:.0%} R {mapped[1]:.0%}, "
        "3 m out: " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    if not check:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"{c.key}.npz", left=left, right=right)
    return {"key": c.key, "fit_before_m": d0, "fit_after_m": d1, "mapped_left": mapped[0], "mapped_right": mapped[1],
            **{f"at3m_{k}": v for k, v in share.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--report", type=Path, help="also write the per-circuit summary as JSON")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    rows = []
    for c in CALENDAR:
        if only and c.key not in only:
            continue
        if a.download:
            fetch(a.sources, c)
        if not (a.sources / "osm" / f"{c.geo}.osm.json").exists():
            print(f"{c.key}: not fetched (use --download)")
            continue
        rows.append(build_one(c, a.sources, a.check))
    if a.report:
        a.report.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
