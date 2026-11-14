"""Tyre model — roadmap phase 3.

Pacejka "Magic Formula" for pure longitudinal and lateral slip, a normalised-slip
combination for combined slip (friction ellipse), and load sensitivity
(grip coefficient falls as vertical load rises).

    F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))

D = mu(Fz) * Fz is the peak force. B is solved so the peak sits at the
requested slip, so parameters stay physical: peak slip, shape, sliding grip.
Sign convention: positive slip ratio = driving (wheel faster than road);
positive slip angle gives positive lateral force.
"""
import math
from dataclasses import dataclass, field
from functools import lru_cache


@lru_cache(maxsize=None)
def _peak_input(C: float, E: float) -> float:
    """Value u = B*x at which sin(C*atan(u - E(u - atan u))) peaks."""
    best_u, best_f = 0.0, -1.0
    u = 0.0
    while u < 20.0:
        f = math.sin(C * math.atan(u - E * (u - math.atan(u))))
        if f > best_f:
            best_u, best_f = u, f
        u += 0.0005
    return best_u


def magic_formula(x: float, B: float, C: float, D: float, E: float) -> float:
    bx = B * x
    return D * math.sin(C * math.atan(bx - E * (bx - math.atan(bx))))


@dataclass
class TyreCurve:
    """One direction (longitudinal or lateral) of the tyre."""
    mu0: float  # grip coefficient at nominal load
    peak_slip: float  # slip ratio (longitudinal) or slip angle in rad (lateral) at peak force
    C: float  # shape factor: higher = sharper drop after the peak
    E: float  # curvature factor: shapes the region around the peak

    @property
    def B(self) -> float:
        return _peak_input(self.C, self.E) / self.peak_slip


@dataclass
class TyreParams:
    # Race slick on a dry, rubbered track. Values are generic, not from a real tyre.
    longitudinal: TyreCurve = field(default_factory=lambda: TyreCurve(mu0=1.75, peak_slip=0.11, C=1.65, E=0.10))
    lateral: TyreCurve = field(default_factory=lambda: TyreCurve(mu0=1.70, peak_slip=math.radians(7.0), C=1.45, E=0.20))
    nominal_load: float = 4000.0  # N per tyre
    load_sensitivity: float = -0.08  # d(mu)/mu per unit (Fz - Fz0)/Fz0
    grip_scale: float = 1.0  # multiplier for track/weather/temperature effects (later phases)

    def mu(self, curve: TyreCurve, fz: float, scale: float = 1.0) -> float:
        """Grip coefficient at load fz. `scale` = this tyre's own grip factor (temperature, wear...)."""
        dfz = (fz - self.nominal_load) / self.nominal_load
        return max(curve.mu0 * (1 + self.load_sensitivity * dfz), 0.5 * curve.mu0) * self.grip_scale * scale


def fx_pure(p: TyreParams, kappa: float, fz: float, scale: float = 1.0) -> float:
    """Longitudinal force (N) of one tyre at slip ratio kappa, load fz."""
    if fz <= 0:
        return 0.0
    c = p.longitudinal
    return magic_formula(kappa, c.B, c.C, p.mu(c, fz, scale) * fz, c.E)


def fy_pure(p: TyreParams, alpha: float, fz: float, scale: float = 1.0) -> float:
    """Lateral force (N) of one tyre at slip angle alpha (rad), load fz."""
    if fz <= 0:
        return 0.0
    c = p.lateral
    return magic_formula(alpha, c.B, c.C, p.mu(c, fz, scale) * fz, c.E)


def combined(p: TyreParams, kappa: float, alpha: float, fz: float, scale: float = 1.0):
    """(Fx, Fy) for combined slip.

    Each slip is normalised by its peak; the resultant slip drives both curves and
    the force is split along the slip direction. At the peak of one direction
    there is little left for the other — the friction ellipse.
    """
    sx = kappa / p.longitudinal.peak_slip
    sy = math.tan(alpha) / math.tan(p.lateral.peak_slip)
    rho = math.hypot(sx, sy)
    if rho < 1e-9:
        return 0.0, 0.0
    fx = fx_pure(p, rho * p.longitudinal.peak_slip, fz, scale) * sx / rho
    fy = fy_pure(p, math.atan(rho * math.tan(p.lateral.peak_slip)), fz, scale) * sy / rho
    return fx, fy


def dfx_dkappa(p: TyreParams, kappa: float, fz: float, eps: float = 1e-4) -> float:
    return (fx_pure(p, kappa + eps, fz) - fx_pure(p, kappa - eps, fz)) / (2 * eps)


# ---- fast path for the vehicle model (phase 14) ---------------------------------------------
class TyreConsts:
    """Per-step constants of a TyreParams (so the hot loop reads plain floats)."""
    __slots__ = ("kp", "Bx", "Cx", "Ex", "mux", "tap", "By", "Cy", "Ey", "muy", "fz0", "ls", "gs")

    def __init__(self, p: TyreParams):
        lo, la = p.longitudinal, p.lateral
        self.kp, self.Bx, self.Cx, self.Ex, self.mux = lo.peak_slip, lo.B, lo.C, lo.E, lo.mu0
        self.tap, self.By, self.Cy, self.Ey, self.muy = math.tan(la.peak_slip), la.B, la.C, la.E, la.mu0
        self.fz0, self.ls, self.gs = p.nominal_load, p.load_sensitivity, p.grip_scale


def _mf_and_slope(x, B, C, D, E):
    """Magic formula value and its derivative dF/dx."""
    bx = B * x
    at = math.atan(bx)
    phi = bx - E * (bx - at)
    ca = C * math.atan(phi)
    dphi = B * (1 - E + E / (1 + bx * bx))
    return D * math.sin(ca), D * math.cos(ca) * C / (1 + phi * phi) * dphi


def combined_fast(t: TyreConsts, kappa: float, alpha: float, fz: float, scale_x: float = 1.0,
                  scale_y: float = 1.0):
    """(Fx, Fy, dFx/dkappa) in one pass — same model as combined(), with separate grip scales
    for the two directions (camber helps cornering but costs braking/traction patch)."""
    if fz <= 0:
        return 0.0, 0.0, 0.0
    load = max(1 + t.ls * (fz - t.fz0) / t.fz0, 0.5) * t.gs * fz
    dx = t.mux * load * scale_x
    dy = t.muy * load * scale_y
    sx = kappa / t.kp
    sy = math.tan(alpha) / t.tap
    rho = math.hypot(sx, sy)
    if rho < 1e-9:
        return 0.0, 0.0, _mf_and_slope(0.0, t.Bx, t.Cx, dx, t.Ex)[1]
    F, dF = _mf_and_slope(rho * t.kp, t.Bx, t.Cx, dx, t.Ex)
    fx = F * sx / rho
    fy = magic_formula(math.atan(rho * t.tap), t.By, t.Cy, dy, t.Ey) * sy / rho
    cx, cy = sx * sx / (rho * rho), sy * sy / (rho * rho)
    slope = dF * cx + F / (t.kp * rho) * cy
    return fx, fy, slope
