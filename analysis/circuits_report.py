"""Circuits report: Monza (real data) next to random circuits from the generator.

Usage: python analysis/circuits_report.py [out.png]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, INK, MUTED, ORANGE
from carsim.circuits import monza
from carsim.trackgen import random_track, track_stats


def draw(ax, tr, color):
    for w in (tr.width_left, -tr.width_right):
        ax.plot(tr.x - w * np.sin(tr.heading), tr.y + w * np.cos(tr.heading), color=color, lw=0.8)
    ax.plot(tr.x[0], tr.y[0], "o", color=INK, ms=4)
    st = track_stats(tr)
    ax.set_title(f"{tr.name}", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.text(0.0, -0.02, f"{st['length_m'] / 1000:.2f} km · {tr.width:.1f} m wide · tightest R {st['min_radius_m']:.0f} m",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
    ax.set_aspect("equal")
    ax.axis("off")


def main(out="docs/images/circuits_report.png"):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), dpi=150)
    draw(axes.flat[0], monza(), ORANGE)
    for ax, seed in zip(axes.flat[1:], (0, 3, 5, 7, 8, 10, 2)):
        draw(ax, random_track(seed), BLUE)
    fig.suptitle("Monza (surveyed data, orange) and random circuits from carsim.trackgen (blue); dot = start line",
                 x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
