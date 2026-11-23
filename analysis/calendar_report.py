"""Calendar report: the 24 circuits of the 2026 F1 calendar (the held-out test set).

Each map shows the track edges, the start (dot) and the race direction (arrow); the colour
gives the data source (surveyed, surveyed + traced where rebuilt, traced only).

Usage: python analysis/calendar_report.py [out.png]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import AQUA, BLUE, INK, MUTED, ORANGE
from carsim.circuits import get_track
from carsim.f1_calendar import CALENDAR, PREFIX
from carsim.trackgen import track_stats

COLOUR = {"tum": BLUE, "hybrid": AQUA, "geo": ORANGE}


def draw(ax, c):
    tr = get_track(PREFIX + c.key)
    col = COLOUR[c.source]
    for w in (tr.width_left, -tr.width_right):
        ax.plot(tr.x - w * np.sin(tr.heading), tr.y + w * np.cos(tr.heading), color=col, lw=0.7)
    ax.plot(tr.x[0], tr.y[0], "o", color=INK, ms=4, zorder=3)
    i = int(len(tr.s) * 0.04)  # the arrow: a little after the start, pointing along the lap
    ax.annotate("", xy=(tr.x[i + 40], tr.y[i + 40]), xytext=(tr.x[i], tr.y[i]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    st = track_stats(tr)
    ax.set_title(f"{c.round}. {c.key}" + (" (cancelled)" if c.cancelled else ""), loc="left",
                 fontsize=9, color=INK, fontweight="bold")
    ax.text(0.0, -0.02, f"{tr.length / 1000:.2f} km · {tr.width:.1f} m · R min {st['min_radius_m']:.0f} m",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    ax.set_aspect("equal")
    ax.axis("off")


def main(out="docs/images/calendar_report.png"):
    fig, axes = plt.subplots(4, 6, figsize=(18, 12.5), dpi=130)
    for ax, c in zip(axes.flat, CALENDAR):
        draw(ax, c)
    fig.suptitle("2026 F1 calendar: held-out test circuits (f1:<key>). Blue = surveyed (TUM), "
                 "green = surveyed + traced where rebuilt, orange = traced only (nominal width); dot = start",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
