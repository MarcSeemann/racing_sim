"""Learning report: how far the self-taught driver gets round the lap, generation by generation.

Usage: python analysis/learning_report.py [track] [out.png]
       python analysis/learning_report.py random [out.png]     # the all-rounder (random circuits)
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


def main_random(rows, out):
    """Training on random circuits: held-out circuits (pick the best) and the test circuit."""
    gen = np.array([int(r["generation"]) for r in rows])
    frac = np.array([float(r["val_lap_fraction"]) for r in rows])
    done = np.array([int(r["val_laps_done"]) for r in rows])
    n_val = int(rows[-1]["val_tracks"])
    t_prog = np.array([float(r["test_progress_m"]) for r in rows])
    t_lap = np.array([float(r["test_lap_s"]) for r in rows])
    mins = float(rows[-1]["minutes"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)
    best = np.nanmin(t_lap) if np.isfinite(t_lap).any() else None
    fig.suptitle("Self-taught driver, trained on random circuits: " +
                 (f"best test-track lap {best:.2f} s" if best else f"held-out laps {done.max()}/{n_val} at best"),
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax = axes[0]
    ax.plot(gen, 100 * frac, color=BLUE, lw=1.5, label="Share of the lap driven (mean)")
    ax.plot(gen, 100 * done / n_val, color=ORANGE, lw=1, alpha=0.8, label="Held-out laps finished")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    style(ax, f"{n_val} held-out random circuits (never trained on)", "Generation", "%")
    ax = axes[1]
    ok = np.isfinite(t_prog)
    ax.plot(gen[ok], t_prog[ok], color=BLUE, lw=1.5, marker="o", ms=3)
    style(ax, "Test circuit (never trained on): distance from the line", "Generation", "Distance (m)")
    fig.text(0.99, 0.01, f"{len(rows)} generations, {mins:.0f} min of training", ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(out, facecolor="white")


def main(track="short", out="learning_report.png"):
    rows = list(csv.DictReader(open(run_dir(track) / "progress.csv")))
    if rows and "val_lap_fraction" in rows[0]:
        return main_random(rows, out)
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
