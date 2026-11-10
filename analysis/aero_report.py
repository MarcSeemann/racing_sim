"""Aero report: ride height and downforce vs speed, DRS effect.

Usage: python analysis/aero_report.py [out.png]
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
from carsim import Inputs, LongitudinalCar, run
from carsim.drivers import FullThrottleDriver


def main(out="aero_report.png"):
    log = run(LongitudinalCar(), FullThrottleDriver(), t_end=40.0, log_every=5)
    drs = lambda s, c: Inputs(throttle=1.0, gear=FullThrottleDriver()(s, c).gear, drs=True)
    log_drs = run(LongitudinalCar(), drs, t_end=40.0, log_every=5)
    kmh = log["v"] * 3.6
    order = np.argsort(kmh)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=150)
    fig.suptitle(f"Aero: the car squats {1000 * (0.035 - log['ride_front'].min()):.0f} mm at the nose, "
                 f"ground effect adds downforce, DRS is worth +{(log_drs['v'][-1] - log['v'][-1]) * 3.6:.0f} km/h",
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")

    ax = axes[0]
    ax.plot(kmh[order], 1000 * log["ride_front"][order], color=BLUE, lw=2, label="Front ride height")
    ax.plot(kmh[order], 1000 * log["ride_rear"][order], color=ORANGE, lw=2, label="Rear ride height")
    ax.axhline(8, color=MUTED, lw=1, ls="--")
    ax.text(10, 9, "floor stalls below 8 mm (front)", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    style(ax, "Ride height vs speed", "Speed (km/h)", "Ride height (mm)")
    ax.set_ylim(0, 90)

    ax = axes[1]
    ax.plot(kmh[order], log["ClA"][order], color=BLUE, lw=2)
    ax.axhline(4.0, color=MUTED, lw=1, ls="--")
    ax.text(10, 4.03, "reference ClA 4.0 (at 20 mm)", fontsize=8, color=MUTED)
    style(ax, "Downforce coefficient vs speed", "Speed (km/h)", "ClA (m²)")

    ax = axes[2]
    ax.plot(log["t"], log["v"] * 3.6, color=BLUE, lw=2, label="DRS closed")
    ax.plot(log_drs["t"], log_drs["v"] * 3.6, color=ORANGE, lw=2, label="DRS open")
    for lg, va in ((log, "top"), (log_drs, "bottom")):
        ax.text(39.5, lg["v"][-1] * 3.6 + (-3 if va == "top" else 3), f"{lg['v'][-1] * 3.6:.0f} km/h",
                fontsize=8, color=INK, va=va, ha="right")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    style(ax, "Flat-out run, DRS open vs closed", "Time (s)", "Speed (km/h)")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
