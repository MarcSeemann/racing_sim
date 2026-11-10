"""Straight-line telemetry report: one flat-out run, plus an aero what-if.

Usage: python analysis/straight_line_report.py [out.png]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from carsim import CarParams, LongitudinalCar, run
from carsim.drivers import FullThrottleDriver
from carsim.params import G
from carsim.powertrain import RPM_TO_RADS, full_throttle_torque, overall_ratio

from _style import AQUA, BLUE, GRID, INK, MUTED, ORANGE, style  # noqa: E402

AERO_CONFIGS = {  # name: (CdA, ClA) at the reference ride height
    "Low downforce (Monza-style)": (0.85, 3.0),
    "Medium downforce": (1.10, 4.0),
    "High downforce (Monaco-style)": (1.40, 5.2),
}


def main(out="straight_line_report.png"):
    car = LongitudinalCar()
    log = run(car, FullThrottleDriver(), t_end=40.0)
    t, kmh = log["t"], log["v"] * 3.6

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle("Single-seater flat-out run — 353 km/h top speed, 0–200 km/h in 4.7 s",
                 x=0.06, ha="left", fontsize=13, color=INK, fontweight="bold")

    # A: speed vs time with milestones
    ax = axes[0, 0]
    ax.plot(t, kmh, color=BLUE, lw=2)
    for target in (100, 200, 300):
        tt = log.first_time("v", target / 3.6)
        ax.plot(tt, target, "o", color=BLUE, ms=6, mec="white", mew=1.5)
        ax.annotate(f"{target} km/h in {tt:.1f} s", (tt, target), xytext=(8, -12),
                    textcoords="offset points", fontsize=8, color=INK)
    style(ax, "Speed", "Time (s)", "Speed (km/h)")

    # B: rpm vs time, gear numbers annotated
    ax = axes[0, 1]
    mask = t < 15
    ax.plot(t[mask], log["rpm"][mask], color=BLUE, lw=1.5)
    gears = log["gear"]
    for g in range(1, 9):
        idx = np.nonzero(gears == g)[0]
        idx = idx[idx < mask.sum()]
        if len(idx):
            mid = idx[len(idx) // 2]
            ax.text(t[mid], 13700, str(g), fontsize=8, color=MUTED, ha="center")
    ax.text(0.0, 14300, "gear", ha="left", fontsize=8, color=MUTED)
    style(ax, "Engine speed and gear (first 15 s)", "Time (s)", "Engine speed (rpm)")
    ax.set_ylim(8000, 15500)

    # C: wheel force available vs resistance -> intersection is top speed
    ax = axes[1, 0]
    p, c, rho = car.p, car.p.chassis, car.env.air_density
    v = np.linspace(1, 110, 400)
    best = []
    for vi in v:
        f = 0.0
        for g in range(1, len(p.gearbox.ratios) + 1):
            rpm = vi / c.wheel_radius * overall_ratio(p.gearbox, g) / RPM_TO_RADS
            if g == 1:
                rpm = max(rpm, car.launch_rpm)  # clutch slip at launch
            if rpm >= car.launch_rpm:
                f = max(f, full_throttle_torque(p.engine, rpm) * overall_ratio(p.gearbox, g)
                        * p.gearbox.efficiency / c.wheel_radius)
        best.append(f)
    best = np.array(best)
    grip = p.tyres.longitudinal.mu0 * (c.mass * G * (1 - c.front_weight) + 0.5 * rho * p.aero.ClA * v ** 2 * (1 - p.aero.front_balance))
    resist = 0.5 * rho * p.aero.CdA * v ** 2 + c.crr * (c.mass * G + 0.5 * rho * p.aero.ClA * v ** 2)
    ax.plot(v * 3.6, best / 1000, color=BLUE, lw=2, label="Engine force at wheels (best gear)")
    ax.plot(v * 3.6, resist / 1000, color=ORANGE, lw=2, label="Drag + rolling resistance")
    ax.plot(v * 3.6, grip / 1000, color=MUTED, lw=1.2, ls="--", label="Rear grip limit (static load)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style(ax, "Where top speed comes from", "Speed (km/h)", "Force (kN)")
    ax.set_ylim(0, 36)

    # D: aero what-if
    ax = axes[1, 1]
    rows = []
    for (name, (cda, cla)), col in zip(AERO_CONFIGS.items(), (BLUE, ORANGE, AQUA)):
        pp = CarParams()
        pp.aero.CdA, pp.aero.ClA = cda, cla
        lg = run(LongitudinalCar(pp), FullThrottleDriver(), t_end=40.0)
        ax.plot(lg["t"], lg["v"] * 3.6, color=col, lw=2, label=name)
        ax.text(40.3, lg["v"][-1] * 3.6, f"{lg['v'][-1] * 3.6:.0f}", fontsize=8, color=INK, va="center")
        rows.append((name, lg["v"][-1] * 3.6, lg.first_time("v", 200 / 3.6), lg.first_time("v", 300 / 3.6)))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    style(ax, "What-if: aero package vs straight-line speed", "Time (s)", "Speed (km/h)")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, facecolor="white")
    log.to_csv(str(Path(out).with_suffix(".csv")))
    for r in rows:
        print(f"{r[0]:32s} top {r[1]:6.1f} km/h   0-200 {r[2]:.2f} s   0-300 {r[3]:.2f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
