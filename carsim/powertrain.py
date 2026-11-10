"""Engine and gearbox models."""
import numpy as np

from .params import Engine, Gearbox

RPM_TO_RADS = 2 * np.pi / 60


def full_throttle_torque(engine: Engine, rpm: float) -> float:
    """Full-throttle torque in N m; zero above the rev limiter."""
    if rpm >= engine.rev_limit_rpm:
        return 0.0
    return float(np.interp(rpm, engine.rpm_points, engine.torque_points))


def density_factor(engine: Engine, air_density: float) -> float:
    return (air_density / engine.ref_density) ** engine.density_exponent


def engine_torque(engine: Engine, rpm: float, throttle: float, air_density: float | None = None) -> float:
    """Net crank torque: throttle blends full torque with closed-throttle engine braking."""
    throttle = min(max(throttle, 0.0), 1.0)
    k = 1.0 if air_density is None else density_factor(engine, air_density)
    drive = throttle * full_throttle_torque(engine, rpm) * k
    braking = (1 - throttle) * engine.engine_brake_torque * min(rpm / engine.rev_limit_rpm, 1.0)
    return drive - braking


def overall_ratio(gearbox: Gearbox, gear: int) -> float:
    """Crank-to-wheel ratio; 0 for neutral."""
    if gear <= 0:
        return 0.0
    return gearbox.ratios[gear - 1] * gearbox.final_drive


def peak_power(engine: Engine, n: int = 2000):
    """(power W, rpm) at the peak of the full-throttle curve."""
    rpm = np.linspace(engine.rpm_points[0], engine.rev_limit_rpm - 1, n)
    power = np.array([full_throttle_torque(engine, r) for r in rpm]) * rpm * RPM_TO_RADS
    i = int(np.argmax(power))
    return float(power[i]), float(rpm[i])
