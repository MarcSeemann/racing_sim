"""Track report: hill circuit elevation, crest/compression loads, rubbering-in.

Usage: python analysis/track_report.py [out.png]
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
from carsim.conditions import Conditions, Weather
from carsim.drivers import LapDriver
from carsim.params import G
from carsim.track import hill_circuit


def main(out="track_report.png", laps="8"):
    laps = int(laps)
    tr = hill_circuit()
    cond = Conditions(tr, Weather.dry(), cars_on_track=20)
    car = Car2D(CarParams(tyre_thermals=False), conditions=cond)
    car.set_speed(30.0)
    driver = LapDriver(tr)
    log, times = run_laps(car, driver, laps, log_every=5)
    total = log["fz_fl"] + log["fz_fr"] + log["fz_rl"] + log["fz_rr"]
    static = car.p.chassis.mass * G * np.cos(log["grade"]) + log["downforce"]
    last = log["t"] > sum(times[:-1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
    fig.suptitle(f"Track: {tr.z.max() - tr.z.min():.0f} m of elevation, the car goes light over the crest, "
                 f"and rubber laid by the field makes it {times[1] - times[-1]:.2f} s faster in {laps - 2} laps",
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")

    ax = axes[0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.35, 1.0, 256)))  # lowest point still visible
    sc = ax.scatter(tr.x, tr.y, c=tr.z, s=4, cmap=cmap)
    for s0, name in ((0, "start"), (1600, "crest"), (2300, "compression"), (720, "hairpin")):
        i = int(np.argmin(np.abs(tr.s - s0)))
        ax.plot(tr.x[i], tr.y[i], "o", color=ORANGE if name in ("crest", "compression") else INK, ms=6)
        ax.annotate(name, (tr.x[i], tr.y[i]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04)
    cb.set_label("Elevation (m)", color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_aspect("equal")
    style(ax, "Hill circuit (2.5 km, anticlockwise)", "x (m)", "y (m)")

    ax = axes[1]
    s = log["s_track"][last]
    order = np.argsort(s)
    ax.plot(s[order], (total / static)[last][order], color=BLUE, lw=1.5)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    for s0, name in ((1600, "crest: light"), (2300, "compression: heavy")):
        ax.annotate(name, (s0, 1.0), xytext=(s0 - 300, 1.12 if "heavy" in name else 0.86), fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    style(ax, "Tyre load vs flat ground (weight + downforce)", "Distance along the lap (m)", "Load ratio")
    ax.set_ylim(0.75, 1.25)

    ax = axes[2]
    n = np.arange(1, len(times) + 1)
    ax.plot(n, times, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.2)
    top = times[1] + 0.4
    ax.text(2.2, top - 0.05, f"out-lap {times[0]:.2f} s (off the scale)", fontsize=8, color=INK, va="top")
    ax.annotate(f"{times[-1]:.2f} s", (n[-1], times[-1]), xytext=(-10, -14), textcoords="offset points", fontsize=8, color=INK)
    style(ax, "Lap time on a green track (tyres held constant)", "Lap", "Lap time (s)")
    ax.set_ylim(min(times[1:]) - 0.3, top)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor="white")
    print("lap times", [round(t, 2) for t in times])


if __name__ == "__main__":
    main(*sys.argv[1:])
