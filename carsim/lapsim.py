"""What the CAR can do on a circuit: a quasi-steady-state lap time (no driver, no safety margin).

The classic lap-time-simulation shape, using the car's own numbers instead of a driver's plan:

  * corner speed where lateral grip (with downforce, load-sensitive tyres) equals v^2 * curvature
  * backward pass: braking on the longitudinal grip left over by cornering (friction ellipse), plus
    drag and gradient
  * forward pass: the smaller of engine power (the real full-throttle torque curve through the best
    gear, ERS deployment included) and rear-axle traction, minus drag and gradient

on the minimum-curvature racing line. It ignores transients (the car is assumed to settle at every
point), so it is optimistic by a few per cent compared with a perfect driver; it is the yardstick
for "is the car fast enough?", separate from "is the driver good enough?".

    from carsim.lapsim import lap
    r = lap(get_track("monza"), race_car(10, downforce="low"))
    r["time"], r["top_speed_kmh"]
"""
import math

import numpy as np

from .params import G, CarParams, Environment, pressure_at_altitude
from .powertrain import RPM_TO_RADS, full_throttle_torque, overall_ratio


def wheel_power_curve(p: CarParams, v_max: float = 130.0, n: int = 260):
    """(speeds m/s, power at the wheels W, top-gear rev-limit speed m/s): best gear at each speed."""
    v = np.linspace(1.0, v_max, n)
    r = p.chassis.wheel_radius
    best = np.zeros(n)
    for g in range(1, len(p.gearbox.ratios) + 1):
        ratio = overall_ratio(p.gearbox, g)
        rpm = v / r * ratio / RPM_TO_RADS
        torque = np.array([full_throttle_torque(p.engine, x) for x in rpm])
        best = np.maximum(best, torque * ratio * p.gearbox.efficiency / r * v)
    top = p.engine.rev_limit_rpm * RPM_TO_RADS * r / overall_ratio(p.gearbox, len(p.gearbox.ratios))
    return v, best, top


def lap(track, params: CarParams, env: Environment | None = None, fuel_kg: float = 0.0, line="racing",
        lat_eff: float = 0.86, long_eff: float = 0.95, grip: float = 1.0) -> dict:
    """Quasi-steady lap of `track` with car `params`.

    lat_eff / long_eff: share of the one-big-tyre grip the whole car can use in a corner / under
    braking (load transfer, balance; 0.86 is what the skidpad tests give for this car model).
    grip: track/tyre grip factor (1 = rubbered dry track, tyres in the window).
    Returns time (s), s and v (m/s) along the line, top_speed_kmh, and the minimum speeds."""
    from .racing_line import cached_racing_line
    path = cached_racing_line(track)[0] if line == "racing" else track
    if env is None:
        env = Environment()
        if getattr(track, "altitude_m", None) is not None:
            env.pressure_pa = pressure_at_altitude(track.altitude_m)
    rho = env.air_density
    p, tp = params, params.tyres
    m = p.chassis.mass + fuel_kg
    down = 0.5 * rho * p.aero.ClA / m  # downforce per kg per v^2
    drag = 0.5 * rho * p.aero.CdA / m
    rear = (1 - p.chassis.front_weight, 1 - p.aero.front_balance)
    vv, pw, v_rev = wheel_power_curve(p)
    power = lambda v: float(np.interp(v, vv, pw))
    slope = np.sin(np.asarray(track.grade)) if len(track.grade) == len(path.s) else np.zeros(len(path.s))

    def mu(curve, v2):
        fz = m * (G + down * v2) / 4
        return curve.mu0 * max(1 + tp.load_sensitivity * (fz - tp.nominal_load) / tp.nominal_load, 0.5) * \
            tp.grip_scale * grip

    k = np.abs(np.asarray(path.curvature, float))
    n = len(k)
    ds = np.diff(np.append(path.s, path.length))
    # top speed on a straight: where power = drag (or the rev limiter)
    v_top = min(v_rev, float(vv[np.argmin(np.abs(pw / (m * vv) - drag * vv * vv))]) if pw.max() > 0 else v_rev)
    def mu_arr(curve, v2):
        fz = m * (G + down * v2) / 4
        return curve.mu0 * np.maximum(1 + tp.load_sensitivity * (fz - tp.nominal_load) / tp.nominal_load,
                                      0.5) * tp.grip_scale * grip

    v = np.full(n, v_top)
    for _ in range(60):  # corner speed, all points at once: v^2 k = lat_eff mu(v) (g + down v^2)
        cap = lat_eff * mu_arr(tp.lateral, v * v) * (G + down * v * v)
        with np.errstate(divide="ignore"):
            v_new = np.minimum(np.where(k > 1e-7, np.sqrt(cap / np.maximum(k, 1e-12)), v_top), v_top)
        v = 0.5 * v + 0.5 * v_new
    lat_cap = lambda x: lat_eff * mu(tp.lateral, x * x) * (G + down * x * x)
    long_cap = lambda x: long_eff * mu(tp.longitudinal, x * x) * (G + down * x * x)
    for _ in range(2):  # twice round, so the profile closes on itself
        for j in range(n - 1, -n, -1):  # braking
            a, b = j % n, (j + 1) % n
            vb = v[b]
            lat = min(vb * vb * k[a] / lat_cap(vb), 1.0)
            decel = long_cap(vb) * math.sqrt(1 - lat * lat) + drag * vb * vb + G * slope[a]
            v[a] = min(v[a], math.sqrt(vb * vb + 2 * max(decel, 0.0) * ds[a]))
        for j in range(0, 2 * n):  # acceleration
            a, b = j % n, (j + 1) % n
            va = max(v[a], 3.0)
            lat = min(va * va * k[a] / lat_cap(va), 1.0)
            trac = mu(tp.longitudinal, va * va) * (G * rear[0] + down * va * va * rear[1]) * grip
            acc = min(power(va) / (m * va), trac) * math.sqrt(1 - lat * lat) - drag * va * va - G * slope[a]
            v[b] = min(v[b], math.sqrt(max(va * va + 2 * acc * ds[a], 0.0)))
    t = float(np.sum(ds / np.maximum(0.5 * (v + np.roll(v, -1)), 0.5)))
    return {"time": t, "s": np.asarray(path.s), "v": v, "top_speed_kmh": float(v.max() * 3.6),
            "min_speed_kmh": float(v.min() * 3.6), "air_density": rho, "mass": m}
