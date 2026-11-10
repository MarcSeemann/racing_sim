"""Cornering report: understeer balance, grip vs corner speed, brake-bias stability.

Usage: python analysis/cornering_report.py [out.png]
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from carsim import Car2D, CarParams, Inputs, run
from carsim.drivers import SkidpadDriver
from carsim.params import G

from _style import AQUA, BLUE, GRID, INK, MUTED, ORANGE, style  # noqa: E402


def skidpad(radius, params=None, v_start=10.0):
    d = SkidpadDriver(radius, v_start=v_start)
    car = Car2D(params or CarParams(tyre_thermals=False))
    car.set_speed(v_start)
    log = run(car, d, t_end=150.0, stop=lambda s: abs(d.radius_error(s)) > 3.0)
    err = np.hypot(log["x"], log["y"] - radius) - radius
    held = (np.abs(err) < 0.5) & (log["t"] > 2.0)
    return log, held


def main(out="cornering_report.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
    fig.suptitle("Cornering (tyres at optimum): gentle understeer, downforce adds grip with speed, rear brake bias spins it",
                 x=0.04, ha="left", fontsize=13, color=INK, fontweight="bold")

    # A: understeer curve for three roll-stiffness splits
    ax = axes[0]
    for share, col in zip((0.45, 0.55, 0.65), (BLUE, ORANGE, AQUA)):
        p = CarParams(tyre_thermals=False)
        p.chassis.roll_stiffness_front = share
        log, held = skidpad(50.0, p)
        ay, st = log["ay"][held] / G, np.degrees(log["steer"][held])
        order = np.argsort(ay)
        ay, st = ay[order], st[order]
        bins = np.linspace(0.3, ay.max(), 40)
        idx = np.digitize(ay, bins)
        xs = [ay[idx == k].mean() for k in range(1, len(bins)) if (idx == k).any()]
        ys = [st[idx == k].mean() for k in range(1, len(bins)) if (idx == k).any()]
        ax.plot(xs, ys, color=col, lw=2, label=f"{share:.0%} of roll stiffness at the front")
    ax.axhline(math.degrees(math.atan(3.6 / 50)), color=MUTED, lw=1, ls="--")
    ax.text(1.2, math.degrees(math.atan(3.6 / 50)) + 0.01, "geometric steer for a 50 m circle", fontsize=8, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    style(ax, "Steering needed on a 50 m circle", "Lateral acceleration (g)", "Road-wheel steer angle (deg)")

    # B: max lateral g vs corner speed, with and without downforce
    ax = axes[1]
    radii = (30, 60, 100, 150, 220)
    for cla, col, name in ((4.0, BLUE, "With downforce (ClA 4.0)"), (0.0, ORANGE, "No downforce")):
        pts = []
        for R in radii:
            p = CarParams(tyre_thermals=False)
            p.aero.ClA = cla
            log, held = skidpad(R, p, v_start=math.sqrt(1.0 * G * R))
            i = np.argmax(np.where(held, log["ay"], -np.inf))
            pts.append((log["v"][i] * 3.6, log["ay"][i] / G, R))
        xs, ys, rs = zip(*pts)
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=name)
        if cla > 0:
            for x, y, R in pts:
                ax.annotate(f"R {R} m", (x, y), xytext=(0, 9), textcoords="offset points",
                            fontsize=8, color=MUTED, ha="center")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    style(ax, "Cornering grip vs corner speed", "Corner speed at the limit (km/h)", "Max lateral acceleration (g)")
    ax.set_ylim(1.0, 3.6)

    # C: brake bias
    ax = axes[2]
    for bias, col in zip((0.60, 0.40, 0.25), (BLUE, ORANGE, AQUA)):
        p = CarParams(tyre_thermals=False)
        p.chassis.brake_front_bias = bias
        car = Car2D(p)
        car.set_speed(150 / 3.6)
        log = run(car, lambda s, c: Inputs(brake=1.0, steer=0.01, gear=s.gear), t_end=6.0,
                  stop=lambda s: s.vx <= 0.5 and s.t > 0.5)
        ax.plot(log["t"], np.degrees(log["psi"]), color=col, lw=2, label=f"{bias:.0%} front brake bias")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    style(ax, "Full braking from 150 km/h: heading change", "Time (s)", "Heading change (deg)")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
