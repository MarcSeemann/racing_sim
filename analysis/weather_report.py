"""Weather report: tyre crossover vs water depth, a shower's water and track temperature,
and lap times through the shower on each compound.

Usage: python analysis/weather_report.py [out.png]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _style import AQUA, BLUE, INK, MUTED, ORANGE, style
from carsim import Car2D, CarParams, run_laps
from carsim.conditions import Conditions, SurfaceParams, Weather
from carsim.drivers import LapDriver
from carsim.track import short_circuit

COMPOUNDS = (("slick", "Slick", ORANGE), ("intermediate", "Intermediate", AQUA), ("wet", "Full wet", BLUE))
FROZEN = SurfaceParams(drain_tau=1e12, evap_mm_h=0.0, line_clear_per_lap=0.0)
SHOWER = dict(start=120.0, peak=300.0, end=480.0, rain_mm_h=8.0, air_temp_c=20.0)
SESSION = 900.0


def crossover(depths):
    tr = short_circuit()
    out = {}
    for key, _, _ in COMPOUNDS:
        out[key] = []
        for d in depths:
            cond = Conditions(tr, Weather.constant(air_temp_c=18.0, cloud=1.0), surface=FROZEN, water_mm=d, rubber=2e6)
            car = Car2D(CarParams(tyre_thermals=False), conditions=cond, compound=key)
            car.set_speed(30.0)
            drv = LapDriver(tr)
            _, laps = run_laps(car, drv, 2)
            out[key].append(laps[-1] if len(laps) == 2 and not drv.off_track else np.nan)
            print(key, d, out[key][-1], flush=True)
    return out


def session(compound):
    tr = short_circuit()
    cond = Conditions(tr, Weather.shower(**SHOWER), cars_on_track=20, rubber=2e6)
    car = Car2D(CarParams(), conditions=cond, compound=compound)
    car.set_speed(30.0)
    drv = LapDriver(tr)
    log, laps = run_laps(car, drv, 999, log_every=200, t_max=SESSION)
    print(compound, "laps", len(laps), "off", drv.off_track, flush=True)
    return log, laps, drv.off_track


def main(out="weather_report.png"):
    depths = [0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0]
    xo = crossover(depths)
    runs = {key: session(key) for key, _, _ in COMPOUNDS}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
    fig.suptitle("Weather: slicks win until the track is damp, intermediates for a thin film, full wets in standing water",
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")

    ax = axes[0]
    for key, name, col in COMPOUNDS:
        ax.plot(depths, xo[key], color=col, lw=2, marker="o", ms=6, mec="white", mew=1.2, label=name)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    style(ax, "Lap time vs standing water (1.1 km circuit)", "Water depth on the racing line (mm)", "Lap time (s)")

    ax = axes[1]
    log = runs["intermediate"][0]
    t = log["t"] / 60
    ax.plot(t, log["water"], color=BLUE, lw=2, label="Water on the racing line (mm)")
    ax.plot(t, log["rain"] / 10, color=MUTED, lw=1.2, ls="--", label="Rain (mm/h ÷ 10)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    style(ax, "A shower: rain, standing water and the drying line", "Session time (min)", "mm")
    ax2 = ax.inset_axes([0.55, 0.38, 0.42, 0.3])
    ax2.plot(t, log["track_temp"], color=ORANGE, lw=1.5)
    ax2.set_title("Track temperature (°C)", fontsize=7, color=INK, loc="left")
    ax2.tick_params(labelsize=6, colors=MUTED)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    ax = axes[2]
    for key, name, col in COMPOUNDS:
        lg, laps, off = runs[key]
        ends = np.cumsum(laps) / 60
        ax.plot(ends, laps, color=col, lw=2, marker="o", ms=4, mec="white", mew=0.8,
                label=name + (" (went off)" if off else ""))
    ax.axvspan(SHOWER["start"] / 60, SHOWER["end"] / 60, color="#e6e5e0", alpha=0.5, lw=0)
    ax.text(SHOWER["start"] / 60 + 0.1, ax.get_ylim()[1] if False else 0, "", fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    style(ax, "Lap times through the shower (shaded = raining)", "Session time (min)", "Lap time (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
