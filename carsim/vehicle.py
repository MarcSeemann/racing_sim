"""Longitudinal (straight-line) vehicle model with spinning wheels — phases 1–3.

Rear-wheel drive. Each axle is one lumped wheel pair with its own rotational
speed; the tyre force comes from the slip ratio between wheel and road (tyres.py).
That gives wheelspin, lock-up, clutch slip at launch and the inertia of the
wheels and engine.

Wheel speeds are updated with a linearised implicit step: tyre slip stiffness
makes the wheel equations stiff, and this keeps a 1 ms step stable.
"""
from dataclasses import dataclass

import numpy as np

from .aero import aero_forces, ride_heights
from .params import G, CarParams, Environment
from .powertrain import RPM_TO_RADS, engine_torque, overall_ratio
from .tyres import dfx_dkappa, fx_pure

V_SLIP_MIN = 2.0  # m/s, floor on the slip-ratio denominator near standstill


@dataclass
class Inputs:
    throttle: float = 0.0  # 0..1
    brake: float = 0.0  # 0..1
    gear: int | None = None  # requested gear; None keeps the current one
    steer: float = 0.0  # road-wheel angle, rad, positive = left (used by Car2D)
    drs: bool = False  # rear-wing flap open


@dataclass
class State:
    t: float = 0.0
    x: float = 0.0  # distance travelled, m
    v: float = 0.0  # speed, m/s
    a: float = 0.0  # last longitudinal acceleration, m/s^2 (used for load transfer)
    gear: int = 1
    shift_timer: float = 0.0  # s of torque cut remaining
    rpm: float = 0.0
    w_front: float = 0.0  # wheel speeds, rad/s
    w_rear: float = 0.0
    kappa_front: float = 0.0  # slip ratios
    kappa_rear: float = 0.0
    hf: float = 0.035  # ride heights, m
    hr: float = 0.080


def slip_ratio(w: float, r: float, v: float) -> float:
    return (w * r - v) / max(abs(v), V_SLIP_MIN)


class LongitudinalCar:
    def __init__(self, params: CarParams | None = None, env: Environment | None = None,
                 launch_rpm: float = 9000.0):
        self.p = params or CarParams()
        self.env = env or Environment()
        self.launch_rpm = launch_rpm
        self.state = State(rpm=self.p.engine.idle_rpm, hf=self.p.chassis.ride_height_front,
                           hr=self.p.chassis.ride_height_rear)

    def set_speed(self, v: float):
        """Put the car at speed v with wheels rolling freely."""
        s, r = self.state, self.p.chassis.wheel_radius
        s.v, s.w_front, s.w_rear = v, v / r, v / r

    # ---- helpers ---------------------------------------------------------
    def wheel_rpm_to_engine(self, v: float, gear: int) -> float:
        """Engine rpm for road speed v in a gear, assuming no wheel slip."""
        return v / self.p.chassis.wheel_radius * overall_ratio(self.p.gearbox, gear) / RPM_TO_RADS

    def axle_loads(self, v: float, a: float, drs: bool = False):
        """Normal loads (front, rear) in N, including downforce and pitch load transfer, plus aero."""
        c = self.p.chassis
        w = c.mass * G * np.cos(self.env.grade)
        ar = aero_forces(self.p.aero, self.env.air_density, v + self.env.headwind, v, self.state.hf, drs)
        transfer = c.mass * a * c.cog_height / c.wheelbase  # + when accelerating -> rear
        nf = w * c.front_weight + ar.downforce * ar.front_share - transfer
        nr = w * (1 - c.front_weight) + ar.downforce * (1 - ar.front_share) + transfer
        return max(nf, 0.0), max(nr, 0.0), ar

    def _axle_force(self, kappa: float, n_axle: float):
        """Force and slip stiffness of an axle (two tyres sharing the load)."""
        tp = self.p.tyres
        k = float(np.clip(kappa, -1.0, 1.0))
        fz = n_axle / 2
        return 2 * fx_pure(tp, k, fz), 2 * dfx_dkappa(tp, k, fz)

    @staticmethod
    def _wheel_update(w, torque, brake_torque, fx, dfx_dw, r, inertia, dt):
        """Linearised implicit step of I dw/dt = T - Tb - Fx r. Brakes cannot reverse the wheel.

        A stopped, braked wheel stays locked while the brake can hold the net torque (static friction).
        """
        if w <= 0.0 and brake_torque > 0 and abs(torque - fx * r) <= brake_torque:
            return 0.0
        stiff = max(dfx_dw, 0.0) * r
        w_new = w + dt * (torque - brake_torque - fx * r) / inertia / (1 + dt * stiff / inertia)
        if w_new < 0 and brake_torque > 0:
            w_new = 0.0  # locked
        return w_new

    # ---- one fixed step --------------------------------------------------
    def step(self, inp: Inputs, dt: float) -> dict:
        s, p, c, env = self.state, self.p, self.p.chassis, self.env
        r = c.wheel_radius

        # Gear change: torque cut for shift_time.
        if inp.gear is not None and inp.gear != s.gear and 0 <= inp.gear <= len(p.gearbox.ratios):
            s.gear = inp.gear
            s.shift_timer = p.gearbox.shift_time

        nf, nr, ar = self.axle_loads(s.v, s.a, inp.drs)
        downforce = ar.downforce

        # Tyre forces from current slip.
        s.kappa_front = slip_ratio(s.w_front, r, s.v)
        s.kappa_rear = slip_ratio(s.w_rear, r, s.v)
        fxf, dff = self._axle_force(s.kappa_front, nf)
        fxr, dfr = self._axle_force(s.kappa_rear, nr)
        denom = max(abs(s.v), V_SLIP_MIN)

        # Engine and clutch.
        ratio = overall_ratio(p.gearbox, s.gear)
        eta = p.gearbox.efficiency
        rpm_wheel = s.w_rear * ratio / RPM_TO_RADS
        inertia_rear = c.wheel_inertia_rear
        clutch = "open"
        if s.gear == 0 or s.shift_timer > 0:
            rpm, t_wheel = max(rpm_wheel, p.engine.idle_rpm), 0.0
        elif rpm_wheel < self.launch_rpm and inp.throttle > 0:
            clutch = "slip"
            rpm = self.launch_rpm
            t_wheel = max(engine_torque(p.engine, rpm, inp.throttle, env.air_density), 0.0) * ratio * eta
        elif rpm_wheel < p.engine.idle_rpm:
            rpm, t_wheel = p.engine.idle_rpm, 0.0  # clutch in, engine idles
        else:
            clutch = "locked"
            rpm = rpm_wheel
            t_wheel = engine_torque(p.engine, rpm, inp.throttle, env.air_density) * ratio * eta
            inertia_rear += p.engine.inertia * ratio ** 2
        s.rpm = rpm

        # Traction control: cut drive torque in proportion to slip over target.
        # Traction control: cut drive torque in proportion to slip over target.
        tc_cut = 0.0
        if c.traction_control and t_wheel > 0 and s.kappa_rear > c.tc_target_slip:
            tc_cut = min((s.kappa_rear - c.tc_target_slip) / c.tc_target_slip * 2.0, 1.0)
            t_wheel *= 1 - tc_cut

        # Brakes (torque on each axle).
        brake = min(max(inp.brake, 0.0), 1.0)
        tb_front = brake * c.max_brake_force * c.brake_front_bias * r
        tb_rear = brake * c.max_brake_force * (1 - c.brake_front_bias) * r

        # Resistances on the chassis.
        f_drag = ar.drag
        f_roll = c.crr * (nf + nr) if s.v > 0 else 0.0
        f_grade = c.mass * G * np.sin(env.grade)

        # Integrate wheels first (implicit), then feed the chassis the same
        # end-of-step tyre force the wheel saw, so energy is conserved.
        dfw_f, dfw_r = dff * r / denom, dfr * r / denom
        w_f_old, w_r_old = s.w_front, s.w_rear
        s.w_front = self._wheel_update(w_f_old, 0.0, tb_front, fxf, dfw_f, r, c.wheel_inertia_front, dt)
        s.w_rear = self._wheel_update(w_r_old, t_wheel, tb_rear, fxr, dfw_r, r, inertia_rear, dt)
        if s.w_front > 0:
            fxf += max(dfw_f, 0.0) * (s.w_front - w_f_old)
        if s.w_rear > 0:
            fxr += max(dfw_r, 0.0) * (s.w_rear - w_r_old)

        a = (fxf + fxr - f_drag - f_roll - f_grade) / c.mass
        v_new = s.v + a * dt
        if v_new < 0:
            v_new, a = 0.0, -s.v / dt
        s.v = v_new
        s.x += s.v * dt
        s.a = a
        s.t += dt
        s.shift_timer = max(s.shift_timer - dt, 0.0)
        hf, hr = ride_heights(c, nf, nr)
        k = dt / (c.load_transfer_lag + dt)
        s.hf += k * (hf - s.hf)
        s.hr += k * (hr - s.hr)

        kp = p.tyres.longitudinal.peak_slip
        return {
            "t": s.t, "x": s.x, "v": s.v, "a": a, "gear": s.gear, "rpm": rpm,
            "throttle": inp.throttle, "brake": brake, "clutch": clutch, "tc_cut": tc_cut,
            "kappa_front": s.kappa_front, "kappa_rear": s.kappa_rear,
            "w_front": s.w_front, "w_rear": s.w_rear,
            "fx_front": fxf, "fx_rear": fxr, "f_drag": f_drag, "f_roll": f_roll,
            "downforce": downforce, "n_front": nf, "n_rear": nr,
            "wheelspin": s.kappa_rear > kp, "front_locked": s.w_front == 0.0 and s.v > 0,
            "rear_locked": s.w_rear == 0.0 and s.v > 0,
            "traction_limited": s.kappa_rear > kp,
            "p_wheel": fxr * s.v, "ride_front": s.hf, "ride_rear": s.hr,
            "ClA": ar.ClA, "CdA": ar.CdA, "aero_balance": ar.front_share, "drs": inp.drs,
        }
