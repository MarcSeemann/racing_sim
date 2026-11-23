"""Add elevation to the real circuits: carsim/data/tracks/elevation/<key>.csv (s_m, z_m).

    python tools/build_elevation.py --sources /some/dir --download   # fetch heights + build
    python tools/build_elevation.py --sources /some/dir              # rebuild from the fetch
    python tools/build_elevation.py --sources /some/dir --check      # only report, write nothing

How:
  1. Put the circuit on the map. Every circuit has a GeoJSON trace in bacinger/f1-circuits (lon/lat,
     same pinned commit as tools/build_f1_2026.py). Our centreline (1 m samples) is fitted onto that
     trace with the same rigid ICP as the calendar build (`align`), then turned into lon/lat.
  2. Look up the ground height every SAMPLE_M metres from OpenTopoData (public API, 1 call/s):
     EU-DEM 25 m in Europe, SRTM 30 m elsewhere ("eudem25m,srtm30m": the first that has data).
     Only circuit coordinates are sent. The raw answers are cached in <sources>/elevation/.
  3. Terrain models are noisy (trees, buildings, cuttings: a few metres), so the profile is smoothed
     along the lap (Gaussian, SIGMA_M) and checked: the largest gradient and the total climb are
     printed next to published figures where we know them (KNOWN).

The written heights are metres above sea level: the mean also sets the air pressure at the circuit
(Mexico City at ~2240 m has ~22 % thinner air). The script is deterministic for the same fetch.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_f1_2026 import R_EARTH, SOURCES, align, gauss_closed, geo_line  # noqa: E402
from carsim.f1_calendar import CALENDAR, DATA  # noqa: E402

API = "https://api.opentopodata.org/v1/eudem25m,srtm30m?locations={}"
SAMPLE_M = 10.0  # m between height lookups
MEDIAN_M = 150.0  # m, running median first: removes buildings, bridges and trees (short spikes)
SIGMA_M = 30.0  # m, then Gaussian smoothing of the profile along the lap
OUT_DS = 5.0  # m between written points
BATCH = 100  # OpenTopoData's limit per request
OUT = DATA / "elevation"

# Where the terrain model is wrong (it measures the SURFACE: buildings, stands, tunnels, overpasses),
# the profile is not used: these circuits stay flat at their mean height (the air pressure is still right).
FLAT = {
    "singapore": "street circuit: SRTM sees the city (31 m range, 21 % grades on a flat circuit)",
    "monaco": "tunnel under the hotel and buildings in the terrain model (21 % grades)",
    "las_vegas": "the Strip's hotels and overpasses in SRTM",
    "mexico": "15 % grades on a flat circuit (the stadium section's stands in SRTM)",
}

# Widely quoted height ranges of the lap (m, rounded), as a rough check only.
KNOWN = {"spa": 102, "austin": 41, "spielberg": 65, "monaco": 42}


def _download_geo(src: Path, c):
    dst = src / "geo" / f"{c.geo}.geojson"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(SOURCES["geo"].format(c.geo), timeout=60).read())
    return dst


def lonlat_of_track(track, geo_path: Path, reverse: bool):
    """lon, lat (deg) of every track sample: the track rigidly fitted onto the GeoJSON trace."""
    g = json.loads(geo_path.read_text())
    c = np.asarray(g["features"][0]["geometry"]["coordinates"], float)[:, :2]
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    lon0, lat0 = c[:, 0].mean(), c[:, 1].mean()  # the same frame as geo_line
    G = geo_line(geo_path, reverse)
    A = np.c_[track.x, track.y]
    R, t = align(A, G)
    P = A @ R.T + t
    d = np.hypot(*(P[:, None, :] - G[None, ::25, :]).transpose(2, 0, 1)).min(axis=1)
    lon = lon0 + np.degrees(P[:, 0] / (R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(P[:, 1] / R_EARTH)
    return lon, lat, float(np.median(d))


def fetch_heights(cache: Path, lon, lat, download: bool, log=print):
    """Ground height (m) at each (lon, lat), from the cache or OpenTopoData."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    got = json.loads(cache.read_text()) if cache.exists() else {}
    keys = [f"{a:.6f},{o:.6f}" for o, a in zip(lon, lat)]
    missing = [k for k in dict.fromkeys(keys) if k not in got]
    if missing and not download:
        raise SystemExit(f"{cache.name}: {len(missing)} heights not fetched yet (use --download)")
    for b in range(0, len(missing), BATCH):
        part = missing[b:b + BATCH]
        req = urllib.request.Request(API.format("|".join(part)), headers={"User-Agent": "carsim-elevation"})
        for attempt in range(5):
            try:
                ans = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception as e:  # rate limit or a hiccup: wait and retry
                log(f"   retry ({e})")
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit("OpenTopoData did not answer")
        if ans.get("status") != "OK":
            raise SystemExit(f"OpenTopoData: {ans}")
        for k, r in zip(part, ans["results"]):
            got[k] = [r["elevation"], r["dataset"]]
        cache.write_text(json.dumps(got))
        time.sleep(1.1)  # the public API allows one call per second
    z = np.array([np.nan if got[k][0] is None else got[k][0] for k in keys], float)
    sets = sorted({got[k][1] for k in keys})
    return z, sets


def profile(track, z_samples, s_samples):
    """Smoothed, closed height profile on every track sample: a running median (spikes from
    buildings and trees are shorter than MEDIAN_M; real hills are longer), then a Gaussian."""
    from scipy.ndimage import median_filter
    L = track.length
    step = np.median(np.diff(s_samples))
    z_samples = median_filter(z_samples, size=max(int(round(MEDIAN_M / step)) | 1, 1), mode="wrap")
    z = np.interp(track.s, s_samples, z_samples, period=L)
    ds = L / len(track.s)
    return gauss_closed(z, SIGMA_M / ds)


def build(src: Path, download: bool, check: bool, only=None, log=print):
    from carsim.f1_calendar import load
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in CALENDAR:
        if only and c.key not in only:
            continue
        track = load(c.key, elevation=False)
        lon, lat, fit = lonlat_of_track(track, _download_geo(src, c), c.reverse)
        step = max(int(round(SAMPLE_M / (track.length / len(track.s)))), 1)
        idx = np.arange(0, len(track.s), step)
        raw, sets = fetch_heights(src / "elevation" / f"{c.key}.json", lon[idx], lat[idx], download, log)
        good = np.isfinite(raw)
        if good.mean() < 0.9:
            log(f"{c.key:12s} only {good.mean():.0%} of the heights exist: skipped")
            continue
        z = profile(track, raw[good], track.s[idx][good])
        if c.key in FLAT:
            log(f"{c.key:12s} kept flat at {z.mean():.0f} m: {FLAT[c.key]}")
            z = np.full_like(z, z.mean())
        grade = np.gradient(z, track.s)
        climb = float(np.sum(np.clip(np.diff(np.r_[z, z[0]]), 0, None)))
        rng = float(z.max() - z.min())
        known = KNOWN.get(c.key)
        flag = "" if known is None or abs(rng - known) <= max(0.35 * known, 6) else "  <-- check"
        log(f"{c.key:12s} {'/'.join(sets):18s} fit {fit:4.1f} m  altitude {z.mean():6.0f} m  range {rng:5.1f} m"
            f"{'' if known is None else f' (published ~{known})':18s}  max grade {np.abs(grade).max():5.1%}"
            f"  climb {climb:5.0f} m{flag}")
        rows.append((c.key, rng, known))
        if not check:
            keep = np.arange(0, len(track.s), max(int(round(OUT_DS / (track.length / len(track.s)))), 1))
            how = (f"FLAT at the mean height: {FLAT[c.key]}" if c.key in FLAT else
                   f"running median {MEDIAN_M:.0f} m + Gaussian {SIGMA_M:.0f} m")
            hdr = (f"{c.name}: height above sea level along the lap; built by tools/build_elevation.py\n"
                   f"source: OpenTopoData {'/'.join(sets)}, sampled every {SAMPLE_M:.0f} m, {how}; "
                   f"centreline fitted to f1-circuits {c.geo} (median {fit:.1f} m)\n"
                   "s_m,z_m")
            np.savetxt(OUT / f"{c.key}.csv", np.c_[track.s[keep], z[keep]], fmt="%.2f", delimiter=",", header=hdr)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True, help="directory for the GeoJSON and height cache")
    ap.add_argument("--download", action="store_true", help="fetch what is not in the cache yet")
    ap.add_argument("--check", action="store_true", help="only report, write nothing")
    ap.add_argument("--only", help="comma-separated circuit keys")
    a = ap.parse_args()
    build(a.sources, a.download, a.check, a.only.split(",") if a.only else None)


if __name__ == "__main__":
    main()
