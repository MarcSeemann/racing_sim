"""Driver report: centreline vs racing line — where the line is, and where it gains time.

Usage: python analysis/driver_report.py [out.png]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, style
from carsim import Car2D, CarParams, run_laps
from carsim.compare import delta, lap_trace
from carsim.conditions import Conditions, Weather
from carsim.drivers import LapDriver
from carsim.track import test_circuit


def main(out="driver_report.png"):
    tr = test_circuit()
    runs = {}
    for name in ("centre", "racing"):
        car = Car2D(CarParams(tyre_thermals=False), conditions=Conditions(tr, Weather.dry(), rubber=1e9))
        car.set_speed(30.0)
        drv = LapDriver(tr, line=name)
        log, laps = run_laps(car, drv, 2, log_every=5)
        runs[name] = (log, laps, drv)
        print(name, laps, flush=True)
    a = lap_trace(runs["centre"][0], runs["centre"][1], 1, tr)
    b = lap_trace(runs["racing"][0], runs["racing"][1], 1, tr)
    s, d = delta(a, b)
    path = runs["racing"][2].path

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
    fig.suptitle(f"Racing line: using the full width is worth {runs['centre'][1][1] - runs['racing'][1][1]:.2f} s "
                 f"a lap on the 2.5 km test circuit",
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")

    ax = axes[0]
    half = tr.width / 2
    for sign in (1, -1):
        ax.plot(tr.x - sign * half * np.sin(tr.heading), tr.y + sign * half * np.cos(tr.heading), color=MUTED, lw=1)
    ax.plot(tr.x, tr.y, color=ORANGE, lw=1.2, ls="--", label="Centreline")
    ax.plot(path.x, path.y, color=BLUE, lw=2, label="Racing line")
    ax.set_xlim(220, 760)
    ax.set_ylim(-25, 70)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    style(ax, "Chicane and hairpin: outside, apex, outside", "x (m)", "y (m)")

    ax = axes[1]
    ax.plot(a["s"], a["v"] * 3.6, color=ORANGE, lw=1.5, label="Centreline")
    ax.plot(b["s"], b["v"] * 3.6, color=BLUE, lw=1.5, label="Racing line")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    style(ax, "Speed along the lap", "Distance (m)", "Speed (km/h)")

    ax = axes[2]
    ax.plot(s, d, color=BLUE, lw=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.text(s[-1], d[-1] + 0.15, f"{d[-1]:+.2f} s", fontsize=9, color=INK, ha="right", va="bottom")
    style(ax, "Time delta, racing line minus centreline", "Distance (m)", "Delta (s), negative = racing line ahead")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
