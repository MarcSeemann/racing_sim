"""Planar four-wheel vehicle model — roadmap phases 4–7.

Phase 4 cornering, 5 brakes, 6 ride-height aero (aero.py), 7 tyre temperature
and wear (thermal.py), 8–9 track and weather (conditions.py, optional): slope and
crests, grip per wheel from rubber, water, kerbs and grass, wind, tyre compound.
Phase 10: fuel burns with engine power; mass, weight split, centre-of-gravity height
and yaw inertia follow the fuel load; the engine stops when the tank is empty.

Body states: position (x, y), heading psi, body-frame velocities (vx, vy) and yaw
rate r. Each wheel has its own spin speed and its own combined-slip tyre force.
Load transfer (longitudinal, lateral split by roll stiffness) is quasi-static
with a short lag that stands in for suspension dynamics.

Frames (ISO): x forward, y left, yaw positive anticlockwise. Wheel order:
0 front-left, 1 front-right, 2 rear-left, 3 rear-right.
"""
import math
from dataclasses import dataclass, field

import dataclasses

from . import thermal
from .conditions import COMPOUNDS, Conditions
from .aero import aero_forces, ride_heights
from .params import G, CarParams, Environment
from .powertrain import RPM_TO_RADS, engine_torque, overall_ratio
from .tyres import combined
from .vehicle import Inputs  # noqa: F401  (re-exported: same inputs plus steer)

V_MIN = 2.0  # m/s floor on slip denominators near standstill
WHEELS = ("fl", "fr", "rl", "rr")


@dataclass
class State2D:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    ax: float = 0.0  # body-frame accelerations (for load transfer)
    ay: float = 0.0
    gear: int = 1
    shift_timer: float = 0.0
    rpm: float = 0.0
    dist: float = 0.0
    w: list = field(default_factory=lambda: [0.0] * 4)
    kappa: list = field(default_factory=lambda: [0.0] * 4)
    alpha: list = field(default_factory=lambda: [0.0] * 4)
    steer: float = 0.0
    hf: float = 0.0  # ride heights, m
    hr: float = 0.0
    tyres: list = field(default_factory=list)  # TyreThermalState per wheel

    # Same interface the straight-line drivers use.
    @property
    def v(self) -> float:
        return self.vx

    @property
    def kappa_front(self) -> float:
        return min(self.kappa[0], self.kappa[1])

    @property
    def kappa_rear(self) -> float:
        return min(self.kappa[2], self.kappa[3])

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def sideslip(self) -> float:
        return math.atan2(self.vy, max(abs(self.vx), V_MIN))


class Car2D:
    def __init__(self, params: CarParams | None = None, env: Environment | None = None,
                 launch_rpm: float = 9000.0, conditions: Conditions | None = None,
                 compound: str = "slick"):
        self.p = params or CarParams()
        self.env = dataclasses.replace(env) if env else Environment()
        self.cond = conditions
        self.compound = COMPOUNDS[compound]
        # Each compound has its own temperature window and wear rate.
        self.thermal = dataclasses.replace(self.p.thermal, t_opt=self.compound.t_opt,
                                           wear_rate=self.p.thermal.wear_rate * self.compound.wear_factor)
        self.launch_rpm = launch_rpm
        self.state = State2D(rpm=self.p.engine.idle_rpm)
        c = self.p.chassis
        self.a = c.wheelbase * (1 - c.front_weight)  # CoG to front axle
        self.b = c.wheelbase * c.front_weight  # CoG to rear axle
        self.pos = ((self.a, c.track_front / 2), (self.a, -c.track_front / 2),
                    (-self.b, c.track_rear / 2), (-self.b, -c.track_rear / 2))
        self.inertia = (c.wheel_inertia_front / 2,) * 2 + (c.wheel_inertia_rear / 2,) * 2
        self.state.hf, self.state.hr = c.ride_height_front, c.ride_height_rear
        self.state.tyres = [thermal.TyreThermalState.from_blankets(self.thermal) for _ in range(4)]
        self._vert = (0.0, 0.0)  # (grade, vertical curvature) under the car
        self._wind = (0.0, 0.0)
        self.fuel = self.p.fuel.start_kg
        self.fuel_flow = 0.0
        self._update_mass()

    # ---- mass properties (phase 10) -----------------------------------------
    def _update_mass(self):
        c, f = self.p.chassis, self.p.fuel
        m_dry, m_f = c.mass, max(self.fuel, 0.0)
        self.mass = m_dry + m_f
        self.front_weight = (m_dry * c.front_weight + m_f * f.front_share) / self.mass
        self.cog_height = (m_dry * c.cog_height + m_f * f.cog_height) / self.mass
        self.yaw_inertia = c.yaw_inertia + m_f * f.yaw_radius ** 2
        self.a = c.wheelbase * (1 - self.front_weight)
        self.b = c.wheelbase * self.front_weight
        self.pos = ((self.a, c.track_front / 2), (self.a, -c.track_front / 2),
                    (-self.b, c.track_rear / 2), (-self.b, -c.track_rear / 2))

    def grip_factors(self):
        """Tyre-state grip (temperature, pressure, wear) per wheel; track surface comes on top."""
        if not self.p.tyre_thermals:
            return [1.0] * 4
        return [thermal.grip_factor(self.thermal, t) for t in self.state.tyres]

    def set_speed(self, v: float):
        s, r = self.state, self.p.chassis.wheel_radius
        s.vx = v
        s.w = [v / r] * 4

    def wheel_rpm_to_engine(self, v: float, gear: int) -> float:
        return v / self.p.chassis.wheel_radius * overall_ratio(self.p.gearbox, gear) / RPM_TO_RADS

    # ---- loads -----------------------------------------------------------
    def wheel_loads(self, drs: bool = False):
        s, c = self.state, self.p.chassis
        grade, vcurv = self._vert
        # Slope tilts weight off the wheels; a dip (vcurv > 0) presses the car down, a crest lifts it.
        m, fw, h = self.mass, self.front_weight, self.cog_height
        weight = max(m * (G * math.cos(grade) + s.vx * s.vx * vcurv), 0.0)
        if self.cond is not None:
            v_air = s.vx - self._wind[0]
            ar = aero_forces(self.p.aero, self.env.air_density, v_air, v_air, s.hf, drs)
        else:
            ar = aero_forces(self.p.aero, self.env.air_density, s.vx + self.env.headwind, s.vx, s.hf, drs)
        down = ar.downforce
        dlong = m * s.ax * h / c.wheelbase
        nf = weight * fw + down * ar.front_share - dlong
        nr = weight * (1 - fw) + down * (1 - ar.front_share) + dlong
        self._axles = (nf, nr)
        dlat = m * s.ay * h  # total lateral moment, N m
        dlf = dlat * c.roll_stiffness_front / c.track_front
        dlr = dlat * (1 - c.roll_stiffness_front) / c.track_rear
        # Positive ay (turning left) loads the right-hand wheels.
        n = [nf / 2 - dlf, nf / 2 + dlf, nr / 2 - dlr, nr / 2 + dlr]
        return [max(x, 0.0) for x in n], ar

    # ---- one fixed step --------------------------------------------------
    def step(self, inp, dt: float) -> dict:
        s, p, c, env, tp = self.state, self.p, self.p.chassis, self.env, self.p.tyres
        rad = c.wheel_radius

        if inp.gear is not None and inp.gear != s.gear and 0 <= inp.gear <= len(p.gearbox.ratios):
            s.gear = inp.gear
            s.shift_timer = p.gearbox.shift_time
        steer = max(-c.max_steer, min(c.max_steer, inp.steer))
        s.steer = steer

        # Track and weather under the car.
        cond = self.cond
        surf = ["track"] * 4
        depth = [0.0] * 4
        surf_grip = [self.compound.dry_grip] * 4
        cos_p, sin_p = math.cos(s.psi), math.sin(s.psi)
        if cond is not None:
            tr = cond.track
            ic = cond.locate(s.x, s.y)
            self._vert = cond.vertical(ic)
            self._wind = cond.wind_body(s.psi)
            env.air_temp_c, env.track_temp_c = cond.now.air_temp_c, cond.track_temp
            hc, sc = math.cos(tr.heading[ic]), math.sin(tr.heading[ic])
            for i in range(4):
                px, py = self.pos[i]
                xw = s.x + px * cos_p - py * sin_p
                yw = s.y + px * sin_p + py * cos_p
                ds_w = (xw - tr.x[ic]) * hc + (yw - tr.y[ic]) * sc
                off = -(xw - tr.x[ic]) * sc + (yw - tr.y[ic]) * hc
                surf_grip[i], depth[i], surf[i] = cond.wheel_surface(tr.s[ic] + ds_w, off, self.compound, s.vx)

        drs = bool(getattr(inp, "drs", False))
        loads, ar = self.wheel_loads(drs)
        downforce = ar.downforce
        grips = [g * sg for g, sg in zip(self.grip_factors(), surf_grip)]
        p_slide = [0.0] * 4

        # Engine, clutch, differential.
        ratio = overall_ratio(p.gearbox, s.gear)
        w_rear = (s.w[2] + s.w[3]) / 2
        rpm_wheel = w_rear * ratio / RPM_TO_RADS
        extra_inertia = 0.0
        clutch = "open"
        t_eng = 0.0
        if s.gear == 0 or s.shift_timer > 0:
            rpm, t_axle = max(rpm_wheel, p.engine.idle_rpm), 0.0
        elif rpm_wheel < self.launch_rpm and inp.throttle > 0:
            clutch, rpm = "slip", self.launch_rpm
            t_eng = max(engine_torque(p.engine, rpm, inp.throttle, env.air_density), 0.0)
            t_axle = t_eng * ratio * p.gearbox.efficiency
        elif rpm_wheel < p.engine.idle_rpm:
            rpm, t_axle = p.engine.idle_rpm, 0.0
        else:
            clutch, rpm = "locked", rpm_wheel
            t_eng = engine_torque(p.engine, rpm, inp.throttle, env.air_density)
            t_axle = t_eng * ratio * p.gearbox.efficiency
            extra_inertia = p.engine.inertia * ratio ** 2 / 2
        s.rpm = rpm
        # Fuel (phase 10): burn follows crank power; an empty tank stops the engine.
        fuel_on = p.fuel.start_kg > 0
        if fuel_on:
            if self.fuel <= 0.0:
                t_eng = 0.0
                t_axle = min(t_axle, 0.0)
                self.fuel_flow = 0.0
            else:
                fl = p.fuel
                p_crank = max(t_eng, 0.0) * rpm * RPM_TO_RADS
                self.fuel_flow = fl.idle_flow + p_crank / (fl.efficiency * fl.lhv)
        # Traction control: cut drive torque in proportion to slip over target.
        tc_cut = 0.0
        if c.traction_control and t_axle > 0:
            # Wheelspin, and total rear grip use (drive + cornering, normalised slip > 1 = past the peak).
            worst = max(s.kappa[2], s.kappa[3])
            kp, ap = tp.longitudinal.peak_slip, math.tan(tp.lateral.peak_slip)
            rho = max(math.hypot(max(s.kappa[i], 0.0) / kp, math.tan(s.alpha[i]) / ap) for i in (2, 3))
            cut_spin = (worst - c.tc_target_slip) / c.tc_target_slip * 2.0 if worst > c.tc_target_slip else 0.0
            cut_slide = (rho - 1.05) / 0.3 if (rho > 1.05 and worst > 0.02) else 0.0
            tc_cut = min(max(cut_spin, cut_slide), 1.0)
            t_axle *= 1 - tc_cut
        lock = c.diff_locking * (s.w[2] - s.w[3])
        drive = [0.0, 0.0, t_axle / 2 - lock, t_axle / 2 + lock]

        brake = min(max(inp.brake, 0.0), 1.0)
        bf = brake * c.max_brake_force * c.brake_front_bias * rad / 2
        br = brake * c.max_brake_force * (1 - c.brake_front_bias) * rad / 2
        tbrake = [bf, bf, br, br]

        cos_d, sin_d = math.cos(steer), math.sin(steer)
        fx_body = fy_body = mz = 0.0
        for i in range(4):
            px, py = self.pos[i]
            vxi = s.vx - s.r * py
            vyi = s.vy + s.r * px
            if i < 2:
                v_long = vxi * cos_d + vyi * sin_d
                v_lat = -vxi * sin_d + vyi * cos_d
            else:
                v_long, v_lat = vxi, vyi
            denom = max(abs(v_long), V_MIN)
            kappa = (s.w[i] * rad - v_long) / denom
            alpha = -math.atan(v_lat / denom)
            s.kappa[i], s.alpha[i] = kappa, alpha
            k_eval = max(-1.0, min(1.0, kappa))
            fz = loads[i]
            g = grips[i]
            fx, fy = combined(tp, k_eval, alpha, fz, g)

            # Wheel spin: linearised implicit step, then end-of-step force to the body.
            eps = 1e-4
            dfdk = (combined(tp, k_eval + eps, alpha, fz, g)[0] - combined(tp, k_eval - eps, alpha, fz, g)[0]) / (2 * eps)
            stiff = max(dfdk, 0.0) * rad / denom
            inertia = self.inertia[i] + (extra_inertia if i >= 2 else 0.0)
            tb = tbrake[i]
            w_old = s.w[i]
            if w_old <= 0.0 and tb > 0 and abs(drive[i] - fx * rad) <= tb:
                w_new = 0.0  # stays locked: the brake holds it (static friction)
            else:
                w_new = w_old + dt * (drive[i] - tb - fx * rad) / inertia / (1 + dt * stiff * rad / inertia)
                if w_new < 0 and tb > 0:
                    w_new = 0.0
                else:
                    fx += stiff * (w_new - w_old)
            s.w[i] = w_new
            p_slide[i] = abs(fx * (w_new * rad - v_long)) + abs(fy * v_lat)

            if i < 2:
                fxb, fyb = fx * cos_d - fy * sin_d, fx * sin_d + fy * cos_d
            else:
                fxb, fyb = fx, fy
            fx_body += fxb
            fy_body += fyb
            mz += px * fyb - py * fxb

        f_drag = ar.drag
        f_roll = c.crr * sum(loads) if s.vx > 0.1 else 0.0
        n_grass = surf.count("grass")
        f_grass = cond.sp.grass_drag * n_grass if (cond is not None and s.vx > 0.1) else 0.0
        fx_body -= f_drag + f_roll + f_grass + self.mass * G * math.sin(self._vert[0])
        if cond is not None:  # crosswind on the body
            v_side = self._wind[1] - s.vy
            fy_body += 0.5 * env.air_density * p.aero.side_force_area * v_side * abs(v_side)

        ax = fx_body / self.mass
        ay = fy_body / self.mass
        # Semi-implicit Euler on the body.
        s.vx += (ax + s.r * s.vy) * dt
        s.vy += (ay - s.r * s.vx) * dt
        s.r += mz / self.yaw_inertia * dt
        if s.vx < 0 and brake > 0 and abs(s.vy) < 0.5:
            s.vx = 0.0
        s.psi += s.r * dt
        cp, sp = math.cos(s.psi), math.sin(s.psi)
        s.x += (s.vx * cp - s.vy * sp) * dt
        s.y += (s.vx * sp + s.vy * cp) * dt
        s.dist += s.speed * dt
        k = dt / (c.load_transfer_lag + dt)
        s.ax += k * (ax - s.ax)
        s.ay += k * (ay - s.ay)
        hf, hr = ride_heights(c, *self._axles)
        s.hf += k * (hf - s.hf)
        s.hr += k * (hr - s.hr)
        if p.tyre_thermals:
            for i in range(4):
                wet = cond.sp.water_cooling * depth[i] * math.sqrt(s.speed) if cond is not None else 0.0
                thermal.update(self.thermal, s.tyres[i], p_slide[i], loads[i], s.speed,
                               env.air_temp_c, env.track_temp_c, dt, wet)
        if cond is not None:
            cond.update(dt, float(cond.track.s[cond.i]), sum(p_slide))
        s.t += dt
        s.shift_timer = max(s.shift_timer - dt, 0.0)
        if fuel_on and self.fuel > 0:
            self.fuel = max(self.fuel - self.fuel_flow * dt, 0.0)
            self._update_mass()

        row = {
            "t": s.t, "x": s.x, "y": s.y, "psi": s.psi, "vx": s.vx, "vy": s.vy, "v": s.speed,
            "yaw_rate": s.r, "ax": ax, "ay": ay, "sideslip": s.sideslip, "steer": steer,
            "throttle": inp.throttle, "brake": brake, "gear": s.gear, "rpm": rpm,
            "clutch": clutch, "tc_cut": tc_cut, "downforce": downforce, "dist": s.dist,
            "drag": f_drag, "ClA": ar.ClA, "CdA": ar.CdA, "aero_balance": ar.front_share,
            "ride_front": s.hf, "ride_rear": s.hr, "drs": drs,
            "grade": self._vert[0], "vert_curv": self._vert[1], "grass_wheels": n_grass,
            "fuel": self.fuel, "fuel_flow": self.fuel_flow, "mass": self.mass,
            "front_weight": self.front_weight,
        }
        if cond is not None:
            b = cond.bin_of(cond.track.s[cond.i])
            row.update({"track_temp": cond.track_temp, "air_temp": cond.now.air_temp_c,
                        "rain": cond.now.rain_mm_h, "water": cond.water[b], "rubber_grip": cond.rubber_grip(b),
                        "s_track": float(cond.track.s[cond.i]), "wind_fwd": self._wind[0], "wind_left": self._wind[1]})
        for i, n in enumerate(WHEELS):
            row[f"fz_{n}"] = loads[i]
            row[f"kappa_{n}"] = s.kappa[i]
            row[f"alpha_{n}"] = s.alpha[i]
            row[f"locked_{n}"] = s.w[i] == 0.0 and s.vx > 0.5
            ts = s.tyres[i]
            row[f"tsurf_{n}"] = ts.t_surface
            row[f"tcore_{n}"] = ts.t_core
            row[f"wear_{n}"] = ts.wear
            row[f"grip_{n}"] = grips[i]
            row[f"press_{n}"] = thermal.pressure(self.thermal, ts)
            row[f"surf_{n}"] = surf[i]
            row[f"sgrip_{n}"] = surf_grip[i]
            row[f"pslide_{n}"] = p_slide[i]
        return row
