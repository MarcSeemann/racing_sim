"""Stint report: lap times, tyre temperatures, wear and pressure over a long run.

Usage: python analysis/stint_report.py [out.png] [laps]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import AQUA, BLUE, INK, MUTED, ORANGE, YELLOW, style
from carsim import Car2D, CarParams, run_laps
from carsim.drivers import LapDriver
from carsim.track import test_circuit

TYRES = (("fl", "Front-left", BLUE), ("fr", "Front-right", ORANGE), ("rl", "Rear-left", AQUA), ("rr", "Rear-right", YELLOW))


def main(out="stint_report.png", laps="30"):
    laps = int(laps)
    car = Car2D(CarParams())
    car.set_speed(30.0)
    driver = LapDriver(test_circuit())
    log, times = run_laps(car, driver, laps)
    ends = np.cumsum(times)
    idx = [int(np.searchsorted(log["t"], e)) - 1 for e in ends]
    n = np.arange(1, len(times) + 1)
    best = int(np.argmin(times))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=150)
    fig.suptitle(f"{len(times)}-lap stint on the test circuit: tyres warm up by lap {best + 1}, "
                 f"then wear costs {times[-1] - times[best]:.2f} s a lap by the end",
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")

    ax = axes[0]
    ax.plot(n, times, color=BLUE, lw=2, marker="o", ms=5, mec="white", mew=1.2)
    ax.annotate(f"best {times[best]:.2f} s", (n[best], times[best]), xytext=(6, -14),
                textcoords="offset points", fontsize=8, color=INK)
    ax.annotate(f"out-lap {times[0]:.2f} s (tyres from 70 °C blankets)", (1, times[0]), xytext=(10, -2),
                textcoords="offset points", fontsize=8, color=INK)
    style(ax, "Lap time", "Lap", "Lap time (s)")

    ax = axes[1]
    for key, name, col in TYRES:
        ax.plot(n, log[f"tsurf_{key}"][idx], color=col, lw=2, label=name)
    ax.axhline(100, color=MUTED, lw=1, ls="--")
    ax.text(n[-1], 99, "grip optimum 100 °C", fontsize=8, color=MUTED, ha="right", va="top")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right", bbox_to_anchor=(1.0, 0.32))
    style(ax, "Tread temperature at the end of each lap", "Lap", "Surface temperature (°C)")

    ax = axes[2]
    for key, name, col in TYRES:
        ax.plot(n, 100 * log[f"wear_{key}"][idx], color=col, lw=2, label=name)
    ax.axhline(75, color=MUTED, lw=1, ls="--")
    ax.text(10, 71, "wear cliff (75%): grip falls away", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    style(ax, "Tread wear", "Lap", "Tread used (%)")
    ax.set_ylim(0, 110)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor="white")
    print("lap times", [round(t, 2) for t in times])
    for key, name, _ in TYRES:
        i = idx[-1]
        print(f"{name:12s} tread {log[f'tsurf_{key}'][i]:.0f} C  core {log[f'tcore_{key}'][i]:.0f} C  "
              f"pressure {log[f'press_{key}'][i]:.2f} bar  wear {100 * log[f'wear_{key}'][i]:.0f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
