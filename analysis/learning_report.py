"""Learning report: how far the self-taught driver gets round the lap, generation by generation.

Usage: python analysis/learning_report.py [track] [out.png]
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, style
from carsim.learn import TRACKS, run_dir


def main(track="short", out="learning_report.png"):
    rows = list(csv.DictReader(open(run_dir(track) / "progress.csv")))
    gen = np.array([int(r["generation"]) for r in rows])
    prog = np.array([float(r["test_progress_m"]) for r in rows])
    lap = np.array([float(r["test_lap_s"]) for r in rows])
    done = np.array([r["test_lap_done"] == "1" for r in rows])
    mins = float(rows[-1]["minutes"])
    L = TRACKS[track]().length
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)
    first = gen[done][0] if done.any() else None
    fig.suptitle(f"Self-taught driver, {track} circuit: " +
                 (f"first full lap at generation {first}, best {np.nanmin(lap):.2f} s" if first else "no full lap yet"),
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax = axes[0]
    ax.plot(gen, np.minimum(prog, L), color=BLUE, lw=1.5, marker="o", ms=3)
    ax.axhline(L, color=MUTED, lw=1, ls="--")
    ax.text(gen[0], L * 1.01, f"full lap ({L:.0f} m)", fontsize=8, color=MUTED, va="bottom")
    style(ax, "Test drive from the start line: distance before crashing", "Generation", "Distance (m)")
    ax.set_ylim(0, L * 1.1)
    ax = axes[1]
    if done.any():
        ax.plot(gen[done], lap[done], color=ORANGE, lw=0, marker="o", ms=6, mec="white")
    style(ax, "Lap time when it finishes", "Generation", "Lap time (s)")
    ax.set_xlim(axes[0].get_xlim())
    fig.text(0.99, 0.01, f"{len(rows)} generations, {mins:.0f} min of training", ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
