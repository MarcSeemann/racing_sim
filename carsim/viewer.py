"""Interactive lap viewer — roadmap phase 12.

Writes one self-contained HTML file (no internet needed) that compares laps:
track map with both cars' lines, speed, throttle/brake and the time delta along
the lap. Hovering any chart moves a marker on all of them and on the map.

    from carsim.viewer import write_viewer
    write_viewer("compare.html", track, [("Baseline", log_a, laps_a), ("+10% downforce", log_b, laps_b)])
"""
import json

import numpy as np

from .compare import lap_trace, lap_window


def _lap_data(name, log, laps, track, lap=None):
    lap = len(laps) - 1 if lap is None else lap
    tr = lap_trace(log, laps, lap, track, grid_m=5.0)
    m = lap_window(log, laps, lap)
    step = max(int(m.sum() // 1500), 1)
    xy = np.c_[log["x"][m], log["y"][m]][::step]
    return {"name": name, "lap_time": round(float(laps[lap]), 3),
            "s": np.round(tr["s"], 1).tolist(), "t": np.round(tr["t"], 3).tolist(),
            "v": np.round(tr["v"] * 3.6, 1).tolist(), "throttle": np.round(tr["throttle"], 3).tolist(),
            "brake": np.round(tr["brake"], 3).tolist(), "xy": np.round(xy, 1).tolist()}


def write_viewer(path: str, track, runs, title: str = "Lap comparison"):
    """runs: [(name, log, lap_times)] — the last lap of each is shown; the first run is the reference."""
    data = {"title": title, "track": {"x": np.round(track.x[::2], 1).tolist(), "y": np.round(track.y[::2], 1).tolist(),
                                      "width": track.width},
            "laps": [_lap_data(n, lg, lp, track) for n, lg, lp in runs]}
    html = _TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__TITLE__", title)
    with open(path, "w") as fh:
        fh.write(html)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfaf7;--panel:#ffffff;--ink:#1f1f1e;--muted:#6b6a63;--grid:#e6e5e0;--c0:#2a78d6;--c1:#eb6834;--c2:#1baf7a;--c3:#eda100}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--panel:#232322;--ink:#f0efec;--muted:#a9a89f;--grid:#3a3a37;--c0:#3987e5;--c1:#d95926;--c2:#199e70;--c3:#c98500}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:20px 24px 8px}h1{font-size:20px;margin:0 0 6px}.laps{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted)}
.key{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
main{display:grid;grid-template-columns:minmax(260px,1fr) 2fr;gap:16px;padding:12px 24px 24px}
@media (max-width:820px){main{grid-template-columns:1fr;padding:12px 16px}}
.card{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:12px}
.card h2{font-size:13px;margin:0 0 6px;font-weight:600}canvas{width:100%;display:block}
#readout{font-variant-numeric:tabular-nums;color:var(--muted);margin-top:8px;min-height:3em}
.charts{display:flex;flex-direction:column;gap:12px}
</style></head><body>
<header><h1 id="title"></h1><div class="laps" id="laps"></div></header>
<main><div class="card"><h2>Track (hover a chart)</h2><canvas id="map" height="300"></canvas><div id="readout"></div></div>
<div class="charts">
<div class="card"><h2>Speed (km/h)</h2><canvas id="speed" height="190"></canvas></div>
<div class="card"><h2>Throttle (up) and brake (down)</h2><canvas id="pedals" height="150"></canvas></div>
<div class="card"><h2 id="dtitle">Time delta (s), negative = ahead of the reference</h2><canvas id="delta" height="150"></canvas></div>
</div></main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const colors = () => [css('--c0'), css('--c1'), css('--c2'), css('--c3')];
document.getElementById('title').textContent = D.title;
const ref = D.laps[0];
D.laps.forEach((l, i) => {
  const d = (l.lap_time - ref.lap_time);
  const el = document.createElement('div');
  el.innerHTML = `<span class="key" style="background:${colors()[i % 4]}"></span>${l.name}: <b style="color:var(--ink)">${l.lap_time.toFixed(3)} s</b>` + (i ? ` (${d >= 0 ? '+' : ''}${d.toFixed(3)})` : ' (reference)');
  document.getElementById('laps').appendChild(el);
});
function deltaOf(l) {  // time of l minus time of ref, at ref's distances
  return ref.s.map((s, k) => { let j = Math.min(k, l.s.length - 1); return l.t[j] - ref.t[k]; });
}
let hover = null;
function setup(cv) { const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = r.width * dpr; cv.height = cv.getAttribute('height') * dpr; const g = cv.getContext('2d'); g.scale(dpr, dpr);
  return [g, r.width, +cv.getAttribute('height')]; }
function chart(id, series, yMin, yMax, opts = {}) {
  const cv = document.getElementById(id), [g, W, H] = setup(cv), L = 44, R = 10, T = 8, B = 22;
  const sMax = ref.s[ref.s.length - 1], X = s => L + (W - L - R) * s / sMax, Y = v => T + (H - T - B) * (1 - (v - yMin) / (yMax - yMin));
  g.strokeStyle = css('--grid'); g.fillStyle = css('--muted'); g.font = '11px system-ui'; g.lineWidth = 1;
  const ticks = opts.ticks || 4;
  for (let i = 0; i <= ticks; i++) { const v = yMin + (yMax - yMin) * i / ticks; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke();
    g.fillText(opts.fmt ? opts.fmt(v) : v.toFixed(0), 4, Y(v) + 4); }
  for (let s = 0; s <= sMax; s += 500) g.fillText(s + ' m', X(s) - 12, H - 6);
  series.forEach(([xs, ys, col]) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    ys.forEach((v, k) => { const x = X(xs[k]), y = Y(v); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); });
  if (hover !== null) { g.strokeStyle = css('--muted'); g.beginPath(); g.moveTo(X(hover), T); g.lineTo(X(hover), H - B); g.stroke(); }
  cv.onmousemove = e => { const r = cv.getBoundingClientRect(); hover = Math.max(0, Math.min(sMax, (e.clientX - r.left - L) / (W - L - R) * sMax)); draw(); };
  cv.onmouseleave = () => { hover = null; draw(); };
  cv.ontouchmove = e => { const r = cv.getBoundingClientRect(), x = e.touches[0].clientX;
    hover = Math.max(0, Math.min(sMax, (x - r.left - L) / (W - L - R) * sMax)); draw(); e.preventDefault(); };
}
function draw() {
  const c = colors();
  const vAll = D.laps.flatMap(l => l.v), vMin = Math.floor(Math.min(...vAll) / 50) * 50, vMax = Math.ceil(Math.max(...vAll) / 50) * 50;
  chart('speed', D.laps.map((l, i) => [l.s, l.v, c[i % 4]]), vMin, vMax, {ticks: Math.round((vMax - vMin) / 50)});
  chart('pedals', D.laps.flatMap((l, i) => [[l.s, l.throttle, c[i % 4]], [l.s, l.brake.map(b => -b), c[i % 4]]]), -1, 1, {ticks: 2, fmt: v => v > 0 ? 'thr' : v < 0 ? 'brk' : '0'});
  const ds = D.laps.slice(1).map(deltaOf), dAll = ds.flat().concat([0]);
  const dMin = Math.min(...dAll), dMax = Math.max(...dAll), pad = Math.max(0.05, (dMax - dMin) * 0.1);
  chart('delta', [[ref.s, ref.s.map(() => 0), c[0]]].concat(ds.map((d, i) => [ref.s, d, c[(i + 1) % 4]])), dMin - pad, dMax + pad, {fmt: v => v.toFixed(2)});
  // map
  const cv = document.getElementById('map'), [g, W, H] = setup(cv), xs = D.track.x, ys = D.track.y;
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const k = Math.min((W - 20) / (x1 - x0), (H - 20) / (y1 - y0)), ox = (W - k * (x1 - x0)) / 2, oy = (H - k * (y1 - y0)) / 2;
  const P = (x, y) => [ox + (x - x0) * k, H - oy - (y - y0) * k];
  g.strokeStyle = css('--grid'); g.lineWidth = Math.max(D.track.width * k, 6); g.lineJoin = 'round'; g.beginPath();
  xs.forEach((x, i) => { const [a, b] = P(x, ys[i]); i ? g.lineTo(a, b) : g.moveTo(a, b); }); g.closePath(); g.stroke();
  D.laps.forEach((l, i) => { g.strokeStyle = c[i % 4]; g.lineWidth = 1.5; g.beginPath();
    l.xy.forEach(([x, y], j) => { const [a, b] = P(x, y); j ? g.lineTo(a, b) : g.moveTo(a, b); }); g.stroke(); });
  const ro = document.getElementById('readout');
  if (hover !== null) {
    let txt = `at ${hover.toFixed(0)} m: `;
    D.laps.forEach((l, i) => { const j = Math.min(Math.round(hover / 5), l.s.length - 1);
      const f = l.s.length > 1 ? j / (l.s.length - 1) : 0, p = l.xy[Math.min(Math.round(f * (l.xy.length - 1)), l.xy.length - 1)];
      const [a, b] = P(p[0], p[1]); g.fillStyle = c[i % 4]; g.beginPath(); g.arc(a, b, 5, 0, 7); g.fill();
      txt += `${l.name} ${l.v[j].toFixed(0)} km/h${i ? ` (${(l.t[j] - ref.t[Math.min(j, ref.t.length - 1)] >= 0 ? '+' : '')}${(l.t[j] - ref.t[Math.min(j, ref.t.length - 1)]).toFixed(2)} s)` : ''}   `; });
    ro.textContent = txt;
  } else ro.textContent = 'Hover the charts to see each car at that point of the lap.';
}
draw(); window.addEventListener('resize', draw);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script></body></html>
"""
