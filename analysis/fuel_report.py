"""Fuel report: what fuel load costs, and a race stint where fuel burn hides tyre wear.

Usage: python analysis/fuel_report.py [out.png] [laps]
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
from carsim import Car2D, race_car, run_laps
from carsim.conditions import Conditions, Weather
from carsim.drivers import LapDriver
from carsim.track import test_circuit


def lap_at_fuel(tr, fuel):
    car = Car2D(race_car(fuel, tyre_thermals=False), conditions=Conditions(tr, Weather.dry(), rubber=1e9))
    car.set_speed(30.0)
    _, laps = run_laps(car, LapDriver(tr, line="racing"), 2)
    return laps[1]


def main(out="fuel_report.png", laps="25"):
    laps = int(laps)
    tr = test_circuit()
    loads = [0.0, 25.0, 50.0, 75.0, 100.0]
    times = [lap_at_fuel(tr, f) for f in loads]
    slope = np.polyfit(loads, times, 1)[0] * 10  # s per 10 kg
    print("fuel sweep", list(zip(loads, [round(t, 3) for t in times])), "s/10kg", round(slope, 3), flush=True)

    car = Car2D(race_car(100.0), conditions=Conditions(tr, Weather.dry(), rubber=1e9))
    car.set_speed(30.0)
    log, stint = run_laps(car, LapDriver(tr, line="racing"), laps)
    ends = np.cumsum(stint)
    fuel_at = np.array([100.0] + [log["fuel"][int(np.searchsorted(log["t"], e)) - 1] for e in ends])
    fuel_mid = (fuel_at[:-1] + fuel_at[1:]) / 2
    corrected = np.array(stint) - slope / 10 * (fuel_mid - fuel_mid[-1])  # as if every lap ran on the last lap's fuel
    n = np.arange(1, len(stint) + 1)
    print("stint", [round(t, 2) for t in stint], flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=150)
    hidden = slope / 10 * (fuel_mid[1] - fuel_mid[-1])
    fig.suptitle(f"Fuel: every 10 kg costs {slope:.2f} s a lap; over the stint the lighter car hides "
                 f"{hidden:.2f} s of the {corrected[-1] - corrected[1]:.2f} s the tyres lose",
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")

    ax = axes[0]
    ax.plot(loads, times, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.2)
    style(ax, "Lap time vs fuel load (tyres at optimum)", "Fuel on board (kg)", "Lap time (s)")

    ax = axes[1]
    ax.plot(n, stint, color=BLUE, lw=2, marker="o", ms=4, mec="white", mew=0.8, label="Measured lap time")
    ax.plot(n, corrected, color=ORANGE, lw=2, marker="o", ms=4, mec="white", mew=0.8, label="Fuel-corrected (tyres alone)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    style(ax, f"{len(stint)}-lap stint from 100 kg of fuel", "Lap", "Lap time (s)")
    ax.set_ylim(min(stint[1:] + list(corrected[1:])) - 0.3, max(stint[1:] + list(corrected[1:])) + 0.3)
    ax.text(3.0, ax.get_ylim()[1] - 0.05, f"out-lap {stint[0]:.2f} s (off the scale)", fontsize=8, color=MUTED, va="top")

    ax = axes[2]
    ax.plot(np.arange(0, len(stint) + 1), fuel_at, color=BLUE, lw=2)
    per_lap = (fuel_at[0] - fuel_at[-1]) / len(stint)
    ax.text(len(stint) * 0.45, fuel_at[0] - 8, f"{per_lap:.2f} kg per lap", fontsize=9, color=INK)
    style(ax, "Fuel remaining", "Lap", "Fuel (kg)")
    ax.set_ylim(0, 105)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
