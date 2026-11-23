"""Build the 2026 F1 calendar circuit files (carsim/data/tracks/<key>.csv) from public data.

    python tools/build_f1_2026.py --sources /some/empty/dir --download   # fetch + build
    python tools/build_f1_2026.py --sources /some/dir                    # rebuild from a fetch
    python tools/build_f1_2026.py --sources /some/dir --check            # only report, write nothing

Sources (pinned commits, see SOURCES):
  TUM racetrack database (LGPL-3.0): surveyed centreline + widths, ~2019-2020 layouts
  bacinger/f1-circuits   (MIT):      centreline traced on maps, current layouts, no widths

Per circuit (carsim.f1_calendar.CALENDAR, field `source`):
  "tum"     the TUM file copied unchanged (its layout matches the current one: checked below)
  "geo"     the GeoJSON line -> local metres -> periodic cubic spline through the points ->
            Gaussian smoothing (sigma 2 m) -> a point every 5 m, constant nominal width.
            Tested on the 12 circuits both sources have: with these settings the GeoJSON
            version's lap time (quasi-steady estimate) is within ~3 % of the surveyed one, on average unbiased.
  "hybrid"  the "geo" line as the layout; the TUM survey (rigidly aligned onto it) replaces it
            wherever the two agree within 5 m, i.e. wherever the circuit was not rebuilt,
            blended over 40 m at the joins; widths from the survey (its median where rebuilt)

The written files use the TUM column layout (x_m, y_m, w_tr_right_m, w_tr_left_m), so
carsim.circuits.load_tum_csv reads them all the same way as Monza.

The script is deterministic: rebuilding from the same sources gives the same files.
"""
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carsim.f1_calendar import CALENDAR, DATA  # noqa: E402

SOURCES = {
    "tum": ("https://raw.githubusercontent.com/TUMFTM/racetrack-database/"
            "e59595d1f3573b30d1ded6a08984935b957688e0/tracks/{}.csv"),
    "geo": ("https://raw.githubusercontent.com/bacinger/f1-circuits/"
            "a78e29225481ccdc90ce93427a6cc11bab1313b8/circuits/{}.geojson"),
}
SIGMA_M = 2.0  # Gaussian smoothing of the traced GeoJSON line
OUT_DS = 5.0  # m between written points (like the TUM files)
MATCH_M = 5.0  # hybrid: survey and current layout "agree" within this distance
BLEND_M = 40.0  # hybrid: length of the blend between survey and traced line
R_EARTH = 6378137.0


def download(src: Path):
    for c in CALENDAR:
        for kind, name in (("tum", c.tum), ("geo", c.geo)):
            if name is None:
                continue
            dst = src / kind / (name + (".csv" if kind == "tum" else ".geojson"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with urllib.request.urlopen(SOURCES[kind].format(name), timeout=60) as r:
                    dst.write_bytes(r.read())
                print("fetched", dst)


# ---- geometry helpers (closed curves as (n, 2) arrays, last point != first) ----------------
def arc(P):
    Q = np.vstack([P, P[:1]])
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])


def resample(P, ds, *cols):
    """Points (and per-point columns) every `ds` m along the closed polyline P (linear)."""
    s = arc(P)
    t = np.arange(0.0, s[-1] - 1e-9, s[-1] / round(s[-1] / ds))
    close = lambda a: np.append(a, a[0])
    out = np.c_[np.interp(t, s, close(P[:, 0])), np.interp(t, s, close(P[:, 1]))]
    return (out,) + tuple(np.interp(t, s, close(c)) for c in cols)


def spline(P, ds=1.0, *cols):
    """Periodic cubic spline through the points (chord-length parameter), sampled every ds m."""
    s = arc(P)
    close = lambda a: np.append(a, a[0])
    t = np.linspace(0.0, s[-1], int(round(s[-1] / 0.25)), endpoint=False)
    sp = [CubicSpline(s, close(a), bc_type="periodic")(t) for a in (P[:, 0], P[:, 1]) + cols]
    return resample(np.c_[sp[0], sp[1]], ds, *sp[2:])


def gauss_closed(a, sigma):
    h = int(4 * sigma)
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.concatenate([a[-h:], a, a[:h]]), k, "valid")


def headings(P):
    d = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    return d / np.hypot(*d.T)[:, None]


def signed_area(P):
    x, y = P.T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---- sources -----------------------------------------------------------------------------
def geo_line(path: Path, reverse: bool):
    """The GeoJSON line in local metres (east, north) about its centroid, in race direction,
    from the start line, splined + smoothed, a point every 1 m."""
    g = json.loads(path.read_text())
    c = np.asarray(g["features"][0]["geometry"]["coordinates"], float)[:, :2]
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    lon0, lat0 = c[:, 0].mean(), c[:, 1].mean()
    P = np.c_[np.radians(c[:, 0] - lon0) * R_EARTH * np.cos(np.radians(lat0)),
              np.radians(c[:, 1] - lat0) * R_EARTH]
    if reverse:
        P = np.roll(P[::-1], 1, axis=0)  # same start point, the other way round
    (L,) = spline(P, 1.0)
    L = np.c_[gauss_closed(L[:, 0], SIGMA_M), gauss_closed(L[:, 1], SIGMA_M)]
    return resample(L, 1.0)[0]


def tum_line(path: Path):
    d = np.loadtxt(path, delimiter=",", comments="#")
    P, wr, wl = spline(d[:, :2], 1.0, d[:, 2], d[:, 3])
    return P, wr, wl


def align(A, B, iters=80):
    """Rigid transform (R, t) putting closed curve A onto B (both 1 m spaced): ICP from a few
    starting rotations, then trimmed to the points that agree (so rebuilt parts do not pull)."""
    tree = cKDTree(B)
    best = None
    for a0 in np.radians(np.arange(0, 360, 30)):
        c, s = np.cos(a0), np.sin(a0)
        R = np.array([[c, -s], [s, c]])
        t = B.mean(0) - A.mean(0) @ R.T
        for it in range(iters):
            P = A @ R.T + t
            d, j = tree.query(P)
            m = d < (np.inf if it < iters // 2 else max(3 * np.median(d), 2.0))
            pa, pb = A[m], B[j[m]]
            ca, cb = pa.mean(0), pb.mean(0)
            U, _, Vt = np.linalg.svd((pa - ca).T @ (pb - cb))
            D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            t = cb - ca @ R.T
        d, _ = tree.query(A @ R.T + t)
        score = np.median(d)
        if best is None or score < best[0]:
            best = (score, R, t)
    return best[1], best[2]


def runs_fix(mask, min_true, min_false):
    """Closed boolean series: drop True runs shorter than min_true, fill False gaps shorter than min_false."""
    def runs(m):
        n = len(m)
        if m.all() or not m.any():
            return [(0, n, bool(m[0]))]
        k = int(np.argmax(m != m[0]))  # rotate so a run starts at 0
        r = np.roll(m, -k)
        edges = np.flatnonzero(np.diff(r.astype(int))) + 1
        bounds = np.r_[0, edges, n]
        return [((b0 + k) % n, b1 - b0, bool(r[b0])) for b0, b1 in zip(bounds[:-1], bounds[1:])]
    m = mask.copy()
    for start, length, val in runs(m):
        if not val and length < min_false:
            m[(start + np.arange(length)) % len(m)] = True
    for start, length, val in runs(m):
        if val and length < min_true:
            m[(start + np.arange(length)) % len(m)] = False
    return m


def hybrid(G, T, wr, wl):
    """Current layout G with survey T (+ widths) wherever T agrees with it."""
    R, t = align(T, G)
    T = T @ R.T + t
    if np.linalg.det(R) < 0:
        raise ValueError("survey would have to be mirrored")
    d, j = cKDTree(T).query(G)
    same_way = np.sum(headings(G) * headings(T)[j], axis=1) > 0.9
    if np.mean(same_way[d < MATCH_M]) < 0.5:  # survey runs the other way round
        T, wr, wl = T[::-1], wl[::-1], wr[::-1]
        d, j = cKDTree(T).query(G)
        same_way = np.sum(headings(G) * headings(T)[j], axis=1) > 0.9
    m = runs_fix((d < MATCH_M) & same_way, min_true=60, min_false=15)
    # weight of the survey: 0 on the rebuilt parts, ramping up to 1 inside the agreeing parts
    # (the ramp lies entirely where the two agree, so it never pulls towards an old layout)
    w = np.clip(2 * gauss_closed(m.astype(float), BLEND_M / 2) - 1, 0, 1)
    P = w[:, None] * T[j] + (1 - w[:, None]) * G
    w_r = w * wr[j] + (1 - w) * np.median(wr)
    w_l = w * wl[j] + (1 - w) * np.median(wl)
    return resample(P, 1.0, w_r, w_l), float(np.mean(m)), float(d[m].mean())


def header(c, note):
    lines = [f"{c.name} ({c.grand_prix}, round {c.round} of 2026); built by tools/build_f1_2026.py",
             f"source: {c.source}; {note}", "x_m,y_m,w_tr_right_m,w_tr_left_m"]
    return "\n".join(lines)


def build(src: Path, check: bool):
    for c in CALENDAR:
        out = DATA / f"{c.key}.csv"
        if c.key == "monza":
            print(f"{c.key:12s} tum     kept: monza.csv (the existing file, unchanged)")
            continue
        if c.source == "tum":
            f = src / "tum" / f"{c.tum}.csv"
            note = "copied unchanged"
            if not check:
                shutil.copyfile(f, out)
            d = np.loadtxt(f, delimiter=",", comments="#")
            L = arc(d[:, :2])[-1]
        else:
            G = geo_line(src / "geo" / f"{c.geo}.geojson", c.reverse)
            if c.source == "geo":
                P, wr, wl = resample(G, OUT_DS)[0], None, None
                wr = wl = np.full(len(P), c.width_m / 2)
                note = f"f1-circuits {c.geo}.geojson, nominal width {c.width_m:.0f} m"
            else:
                (P1, w_r, w_l), share, dev = hybrid(G, *tum_line(src / "tum" / f"{c.tum}.csv"))
                P, wr, wl = resample(P1, OUT_DS, w_r, w_l)
                note = (f"TUM {c.tum}.csv on {share:.0%} of the lap (mean offset {dev:.1f} m), "
                        f"f1-circuits {c.geo}.geojson where rebuilt")
            L = arc(P)[-1]
            if not check:
                np.savetxt(out, np.c_[P, wr, wl], fmt="%.4f", delimiter=",", header=header(c, note))
        err = L / c.length_m - 1
        flag = "  <-- length off by more than 2 %" if abs(err) > 0.02 else ""
        print(f"{c.key:12s} {c.source:7s} {L:6.0f} m (official {c.length_m}, {err:+.1%})  {note}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True, help="directory with tum/*.csv and geo/*.geojson")
    ap.add_argument("--download", action="store_true", help="fetch the pinned source files first")
    ap.add_argument("--check", action="store_true", help="report only, do not write the circuit files")
    a = ap.parse_args()
    if a.download:
        download(a.sources)
    build(a.sources, a.check)


if __name__ == "__main__":
    main()
