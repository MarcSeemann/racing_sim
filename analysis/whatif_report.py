"""What-if report: every preset study against one baseline, as lap-time deltas.

Usage: python analysis/whatif_report.py [out.png] [track]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, GRID, INK, MUTED, ORANGE, style
from carsim.whatif import PRESETS, Scenario, results_table, run_study


def main(out="whatif_report.png", track="short"):
    base = Scenario(track=track)
    changes = {}
    for study in PRESETS.values():
        changes.update(study)
    res = run_study(base, changes)
    table = results_table(res)
    print(table)
    Path(out).with_suffix(".md").write_text(table + "\n")
    b = res[0][1]["lap_time"]
    rows = [(n, r["lap_time"] - b) for n, r in res[1:] if not r.get("off_track")]
    rows.sort(key=lambda x: x[1])
    names, d = zip(*rows)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.6), dpi=150)
    y = np.arange(len(rows))
    ax.barh(y, d, color=[BLUE if v < 0 else ORANGE for v in d], height=0.62)
    for yi, v in zip(y, d):
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:+.2f} s", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.set_yticks(y, names, fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    lo, hi = min(d), max(d)
    ax.set_xlim(lo - 0.25 * (hi - lo) - 0.1, hi + 0.25 * (hi - lo) + 0.1)
    style(ax, f"What-if: change in lap time vs baseline ({b:.2f} s, {track} circuit, last of 3 laps)",
          "Lap time change (s): blue = faster, orange = slower", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
