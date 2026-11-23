"""Planar four-wheel vehicle model with a sprung body — roadmap phases 4–7, 10, 13, 14.

Phase 4 cornering, 5 brakes, 6 ride-height aero (aero.py), 7 tyre temperature
and wear (thermal.py), 8–9 track and weather (conditions.py, optional): slope and
crests, grip per wheel from rubber, water, kerbs and grass, wind, tyre compound.
Phase 10: fuel burns with engine power; mass, weight split, centre-of-gravity height
and yaw inertia follow the fuel load; the engine stops when the tank is empty.
Phase 13 (realism limits): the steering, throttle and brake move at finite rates (hands,
feet, rack and hydraulics), steering gets slower as the tyres load up with speed, the
tyres' lateral force lags the slip angle by a relaxation length, carbon brakes have a
temperature (little bite cold, fade when overheated), and the gearbox refuses a downshift
that would over-rev the engine.
Phase 14 (vehicle dynamics like a full driving sim):
  * suspension: the sprung body heaves, pitches and rolls on four corner springs, a heave
    (third) spring per axle, anti-roll bars and asymmetric dampers; four unsprung wheels sit
    on tyre springs. Lateral load transfer is split into the geometric part (roll centres,
    instant) and the elastic part (body roll, with its dynamics); anti-dive / anti-squat
    likewise. Bump stops, and a skid plank that hits the track and slides when the floor
    bottoms out. Ride heights (and so the aero) come from this motion.
  * road profile: asphalt waves, kerb ridges with serrations and bumpy grass are inputs at
    each tyre — kerbs kick the car and unload the wheels instead of just scaling grip.
  * camber: static camber plus the share of body roll the wheels follow; it shifts lateral
    grip (leaning into the corner helps) and costs some braking/traction grip.
  * clutch-pack limited-slip differential (preload + power/coast ramps), Ackermann steering.
  * aero rake and yaw sensitivity, drag acting above the ground (pitch moment).

Body states: position (x, y), heading psi, body-frame velocities (vx, vy) and yaw
rate r. Each wheel has its own spin speed and its own combined-slip tyre force.

Frames (ISO): x forward, y left, yaw positive anticlockwise. Wheel order:
0 front-left, 1 front-right, 2 rear-left, 3 rear-right. Body roll is positive with the
right-hand side going down (a left-hand corner), pitch positive nose down.
"""
import math
from dataclasses import dataclass, field

import dataclasses

from . import thermal
from .conditions import COMPOUNDS, Conditions
from .aero import aero_forces, ride_heights
from .params import G, CarParams, Environment, brake_friction, pressure_at_altitude
from .powertrain import RPM_TO_RADS, engine_torque, full_throttle_torque, overall_ratio, peak_power
from .tyres import TyreConsts, combined_fast
from .vehicle import Inputs  # noqa: F401  (re-exported: same inputs plus steer)

V_MIN = 2.0  # m/s floor on slip denominators near standstill
WHEELS = ("fl", "fr", "rl", "rr")
SIDE = (1.0, -1.0, 1.0, -1.0)  # +1 left wheels, -1 right wheels
DEG = math.pi / 180


@dataclass
class State2D:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    ax: float = 0.0  # body-frame accelerations
    ay: float = 0.0
    gear: int = 1
    shift_timer: float = 0.0
    rpm: float = 0.0
    dist: float = 0.0
    w: list = field(default_factory=lambda: [0.0] * 4)
    kappa: list = field(default_factory=lambda: [0.0] * 4)
    alpha: list = field(default_factory=lambda: [0.0] * 4)
    steer: float = 0.0  # actual road-wheel angle (rate limited), rad
    throttle: float = 0.0  # actual pedal positions (rate limited), 0..1
    brake: float = 0.0
    alpha_cmd: list = field(default_factory=lambda: [0.0] * 4)  # geometric slip angle before relaxation
    brake_temp: list = field(default_factory=lambda: [450.0] * 4)  # deg C per corner
    hf: float = 0.0  # ride heights, m
    hr: float = 0.0
    tyres: list = field(default_factory=list)  # TyreThermalState per wheel
    # Suspension (phase 14): sprung body heave z (m, up), pitch th (rad, nose down), roll ph (rad,
    # right side down) and their rates; unsprung wheel heights zu (m, up) and rates; road heights.
    z: float = 0.0
    zd: float = 0.0
    th: float = 0.0
    thd: float = 0.0
    ph: float = 0.0
    phd: float = 0.0
    zu: list = field(default_factory=lambda: [0.0] * 4)
    zud: list = field(default_factory=lambda: [0.0] * 4)
    road: list = field(default_factory=lambda: [0.0] * 4)
    fz: list = field(default_factory=lambda: [0.0] * 4)  # tyre contact loads, N
    camber: list = field(default_factory=lambda: [0.0] * 4)  # deg to the road

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
        altitude = getattr(getattr(conditions, "track", None), "altitude_m", None)
        if env is None and altitude is not None:  # thinner air up high (Mexico City: ~22 % less)
            self.env.pressure_pa = pressure_at_altitude(altitude)
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
        self.state.brake_temp = [self.p.brakes.start_temp] * 4
        self._vert = (0.0, 0.0)  # (grade, vertical curvature) under the car
        self._wind = (0.0, 0.0)
        self.fuel = self.p.fuel.start_kg
        self.fuel_flow = 0.0
        self.surface_grip = [1.0] * 4
        self._f_axle = (0.0, 0.0, 0.0)  # last tyre forces: front Fy, rear Fy, total Fx (body frame)
        self._update_mass()
        self._setup_suspension()
        e = self.p.ers
        self.ers_energy = e.store * e.start_soc if e.enabled else 0.0
        p_peak = peak_power(self.p.engine)[0]
        # Fuel burns on the engine's own (ICE) power; at peak with full deployment this matches the
        # combined-power calibration of Fuel.efficiency.
        self._eta_ice = self.p.fuel.efficiency * (p_peak - (e.power if e.enabled else 0.0)) / p_peak
        self.state.fz = self._static_loads()

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
        self.m_unsprung = (c.unsprung_front,) * 2 + (c.unsprung_rear,) * 2
        self.m_sprung = self.mass - sum(self.m_unsprung)

    # ---- suspension set-up (phase 14) ------------------------------------------
    def _setup_suspension(self):
        """Spring, bar and damper rates from the set-up targets (heave stiffness per axle, roll
        gradient, roll-stiffness split), and preloads so the car starts parked at equilibrium."""
        c = self.p.chassis
        L, tf, tr = c.wheelbase, c.track_front, c.track_rear
        ms = self.m_sprung
        h_ra = c.roll_centre_front + (c.roll_centre_rear - c.roll_centre_front) * self.a / L
        k_roll = ms * G * max(self.cog_height - h_ra, 0.05) / (c.roll_gradient_deg_per_g * DEG)
        kf, kr = k_roll * c.roll_stiffness_front, k_roll * (1 - c.roll_stiffness_front)
        # The tyres are springs in series with the suspension: stiffen the suspension so the body
        # roll (to the ground) matches the target gradient.
        kt = c.tyre_vertical_rate
        ktf, ktr = kt * tf ** 2 / 2, kt * tr ** 2 / 2
        kf = 1 / max(1 / kf - 1 / ktf, 1 / (20 * kf))
        kr = 1 / max(1 / kr - 1 / ktr, 1 / (20 * kr))
        # 40% of each axle's roll stiffness from the corner springs, the rest from the bar; the heave
        # (third) spring makes up the axle's heave stiffness.
        ksf, ksr = 0.4 * kf * 2 / tf ** 2, 0.4 * kr * 2 / tr ** 2
        self.k_corner = (ksf, ksf, ksr, ksr)
        self.k_arb = (0.6 * kf / tf ** 2, 0.6 * kr / tr ** 2)
        # heave_stiffness_* is the axle's stiffness to the ground (springs and tyres in series, the
        # platform the aero was calibrated on): the suspension itself must be stiffer than that.
        k2t = 2 * kt
        hf_s = 1 / max(1 / c.heave_stiffness_front - 1 / k2t, 1 / (20 * c.heave_stiffness_front))
        hr_s = 1 / max(1 / c.heave_stiffness_rear - 1 / k2t, 1 / (20 * c.heave_stiffness_rear))
        self.k_heave = (max(hf_s - 2 * ksf, 0.0), max(hr_s - 2 * ksr, 0.0))
        # Preloads: sprung weight split by the CoG position (pitch equilibrium), plus wheel weight.
        fs_f, fs_r = ms * G * self.b / L / 2, ms * G * self.a / L / 2
        self.fs0 = (fs_f, fs_f, fs_r, fs_r)
        self.ft0 = tuple(f + mu * G for f, mu in zip(self.fs0, self.m_unsprung))
        heave_corner = (ksf + self.k_heave[0] / 2,) * 2 + (ksr + self.k_heave[1] / 2,) * 2
        m_corner = (ms * self.b / L / 2,) * 2 + (ms * self.a / L / 2,) * 2
        crit = [2 * math.sqrt(k * m) for k, m in zip(heave_corner, m_corner)]
        self.c_bump = tuple(c.damping_bump * x for x in crit)
        self.c_reb = tuple(c.damping_rebound * x for x in crit)

    def _static_loads(self):
        c = self.p.chassis
        nf = self.mass * G * self.front_weight
        nr = self.mass * G - nf
        return [nf / 2, nf / 2, nr / 2, nr / 2] if c.suspension != "dynamic" else list(self.ft0)

    def grip_factors(self):
        """Tyre-state grip (temperature, pressure, wear) per wheel; track surface comes on top."""
        if not self.p.tyre_thermals:
            return [1.0] * 4
        return [thermal.grip_factor(self.thermal, t) for t in self.state.tyres]

    def set_speed(self, v: float):
        s, r = self.state, self.p.chassis.wheel_radius
        s.vx = v
        s.w = [v / r] * 4
        self.settle()

    def settle(self, t: float = 0.4, dt: float = 0.001):
        """Let the body settle on its springs under the current aero load (no horizontal motion)."""
        if self.p.chassis.suspension != "dynamic":
            return
        s = self.state
        for _ in range(int(t / dt)):
            ar = self._aero(False)
            self._suspension(dt, ar, 0.0, 0.0, 0.0, [0.0] * 4, G, s.road)
            # Damp the transient hard so the car starts at rest on its springs.
            s.zd *= 0.9
            s.thd *= 0.9
            s.phd *= 0.9
            s.zud = [x * 0.9 for x in s.zud]
        s.fz = self._tyre_spring(s.road, [0.0] * 4)

    def place(self, track, s: float = 0.0, offset: float = 0.0):
        """Put the car on a track at distance s along it, `offset` m left of the centreline (or of
        the line given as `track`), pointing along it. Made-up circuits start at the origin
        heading east (where a new car already is); real ones do not."""
        i = track.index_at(s)
        h = float(track.heading[i])
        st = self.state
        st.x = float(track.x[i]) - offset * math.sin(h)
        st.y = float(track.y[i]) + offset * math.cos(h)
        st.psi = h
        if getattr(self, "cond", None) is not None:
            self.cond.i = self.cond.track.index_at(s)

    def wheel_rpm_to_engine(self, v: float, gear: int) -> float:
        return v / self.p.chassis.wheel_radius * overall_ratio(self.p.gearbox, gear) / RPM_TO_RADS

    # ---- aero and loads ------------------------------------------------------
    def _aero(self, drs: bool):
        s = self.state
        dynamic = self.p.chassis.suspension == "dynamic"
        yaw = s.sideslip if (dynamic and s.vx > 5.0) else 0.0
        h_rear = s.hr if dynamic else None
        if self.cond is not None:
            v_air = s.vx - self._wind[0]
            return aero_forces(self.p.aero, self.env.air_density, v_air, v_air, s.hf, drs, h_rear, yaw)
        return aero_forces(self.p.aero, self.env.air_density, s.vx + self.env.headwind, s.vx, s.hf, drs,
                           h_rear, yaw)

    def wheel_loads(self, drs: bool = False):
        """Quasi-static loads (the phase-4 model, used when chassis.suspension == 'quasi_static')."""
        s, c = self.state, self.p.chassis
        grade, vcurv = self._vert
        # Slope tilts weight off the wheels; a dip (vcurv > 0) presses the car down, a crest lifts it.
        m, fw, h = self.mass, self.front_weight, self.cog_height
        weight = max(m * (G * math.cos(grade) + s.vx * s.vx * vcurv), 0.0)
        ar = self._aero(drs)
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

    def _corner_kinematics(self):
        """Body heights over each wheel, suspension deflections (compression +) and their rates."""
        s = self.state
        zb, d, dd = [0.0] * 4, [0.0] * 4, [0.0] * 4
        for i in range(4):
            px, py = self.pos[i]
            zb[i] = s.z - px * s.th + py * s.ph
            d[i] = s.zu[i] - zb[i]
            dd[i] = s.zud[i] - (s.zd - px * s.thd + py * s.phd)
        return zb, d, dd

    def _spring_forces(self, d, dd):
        """Force of each corner's suspension on the body (up +): spring, heave spring, bar, damper, bump stop."""
        c = self.p.chassis
        f = [0.0] * 4
        for axle, (l, r) in enumerate(((0, 1), (2, 3))):
            arb = self.k_arb[axle] * (d[l] - d[r])
            heave = self.k_heave[axle] * 0.25 * (d[l] + d[r])
            for i, sgn in ((l, 1.0), (r, -1.0)):
                fi = self.fs0[i] + self.k_corner[i] * d[i] + heave + sgn * arb
                fi += (self.c_bump[i] if dd[i] > 0 else self.c_reb[i]) * dd[i]
                if d[i] > c.bump_travel:
                    fi += c.bump_stop_rate * (d[i] - c.bump_travel)
                f[i] = max(fi, 0.0)  # a spring cannot pull the wheel up (droop limit)
        return f

    def _tyre_spring(self, road, road_dot):
        c, s = self.p.chassis, self.state
        return [max(self.ft0[i] + c.tyre_vertical_rate * (road[i] - s.zu[i])
                    + c.tyre_vertical_damping * (road_dot[i] - s.zud[i]), 0.0) for i in range(4)]

    def _geometric_transfer(self):
        """Load moved through the links (roll centres, anti-dive/squat, unsprung inertia): instant."""
        c, s = self.p.chassis, self.state
        fyf, fyr, fxt = self._f_axle
        dlf = (fyf * c.roll_centre_front + 2 * c.unsprung_front * s.ay * c.wheel_radius) / c.track_front
        dlr = (fyr * c.roll_centre_rear + 2 * c.unsprung_rear * s.ay * c.wheel_radius) / c.track_rear
        anti = c.anti_dive if fxt < 0 else c.anti_squat
        dlong = anti * fxt * self.cog_height / c.wheelbase / 2  # per wheel; braking (fx < 0) loads the front
        return [-dlf - dlong, dlf - dlong, -dlr + dlong, dlr + dlong]

    def _suspension(self, dt, ar, fx_t, fy_t, drag, road_dot, g_eff, road, road_floor=(0.0, 0.0)):
        """Advance the body and wheel heights by dt. Returns (tyre spring forces, plank force).
        road_floor: road height under the floor/plank (car centre line) at the front and rear axle."""
        s, c = self.state, self.p.chassis
        zb, d, dd = self._corner_kinematics()
        fs = self._spring_forces(d, dd)
        ft = self._tyre_spring(road, road_dot)
        # Ride heights: the floor over the asphalt under the car's centre line (not the kerbs).
        hf = c.ride_height_front + 0.5 * (zb[0] + zb[1]) - road_floor[0]
        hr = c.ride_height_rear + 0.5 * (zb[2] + zb[3]) - road_floor[1]
        pf = c.plank_rate * (c.min_ride_height - hf) if hf < c.min_ride_height else 0.0
        pr = c.plank_rate * (c.min_ride_height - hr) if hr < c.min_ride_height else 0.0
        ms, m = self.m_sprung, self.mass
        down = ar.downforce
        df, dr = down * ar.front_share, down * (1 - ar.front_share)
        # Heave, pitch (nose down +), roll (right side down +).
        f_z = sum(fs) + pf + pr - ms * g_eff - df - dr
        m_th = (-sum(self.pos[i][0] * fs[i] for i in range(4)) - self.a * pf + self.b * pr
                + self.a * df - self.b * dr)
        anti = c.anti_dive if fx_t < 0 else c.anti_squat
        h = self.cog_height
        m_th += ms / m * (-(1 - anti) * h * fx_t + drag * (h - self.p.aero.drag_height))
        h_ra = c.roll_centre_front + (c.roll_centre_rear - c.roll_centre_front) * self.a / c.wheelbase
        m_ph = sum(self.pos[i][1] * fs[i] for i in range(4)) + ms / m * fy_t * (h - h_ra)
        s.zd += f_z / ms * dt
        s.thd += m_th / c.pitch_inertia * dt
        s.phd += m_ph / c.roll_inertia * dt
        s.z += s.zd * dt
        s.th += s.thd * dt
        s.ph += s.phd * dt
        for i in range(4):
            mu = self.m_unsprung[i]
            s.zud[i] += (ft[i] - fs[i] - mu * g_eff) / mu * dt
            s.zu[i] += s.zud[i] * dt
        s.hf, s.hr = max(hf, 0.0), max(hr, 0.0)
        return ft, pf + pr

    # ---- one fixed step --------------------------------------------------
    def step(self, inp, dt: float, row: bool = True) -> dict | None:
        """Advance the car by dt. row=False skips building the telemetry row (training, speed)."""
        s, p, c, env, tp = self.state, self.p, self.p.chassis, self.env, self.p.tyres
        rad = c.wheel_radius
        dynamic = c.suspension == "dynamic"

        if inp.gear is not None and inp.gear != s.gear and 0 <= inp.gear <= len(p.gearbox.ratios):
            ok = True
            if c.overrev_protection and 0 < inp.gear < s.gear:
                w_r = (s.w[2] + s.w[3]) / 2
                ok = w_r * overall_ratio(p.gearbox, inp.gear) / RPM_TO_RADS < p.engine.rev_limit_rpm
            if ok:
                s.gear = inp.gear
                s.shift_timer = p.gearbox.shift_time
        target = max(-c.max_steer, min(c.max_steer, inp.steer))
        thr_t = min(max(inp.throttle, 0.0), 1.0)
        brk_t = min(max(inp.brake, 0.0), 1.0)
        if c.actuator_limits:
            f = min(s.speed / c.steer_rate_speed, 1.0)
            ds_max = (c.max_steer_rate + (c.max_steer_rate_fast - c.max_steer_rate) * f) * dt
            s.steer += max(-ds_max, min(ds_max, target - s.steer))
            s.throttle += max(-c.throttle_rate_down * dt, min(c.throttle_rate_up * dt, thr_t - s.throttle))
            s.brake += max(-c.brake_rate_down * dt, min(c.brake_rate_up * dt, brk_t - s.brake))
        else:
            s.steer, s.throttle, s.brake = target, thr_t, brk_t
        steer, throttle = s.steer, s.throttle

        # Track and weather under the car.
        cond = self.cond
        surf = ["track"] * 4
        depth = [0.0] * 4
        surf_grip = [self.compound.dry_grip] * 4
        road = [0.0] * 4
        floor = (0.0, 0.0)
        cos_p, sin_p = math.cos(s.psi), math.sin(s.psi)
        if cond is not None:
            tr = cond.track
            ic = cond.locate(s.x, s.y)
            self._vert = cond.vertical(ic)
            self._wind = cond.wind_body(s.psi)
            env.air_temp_c, env.track_temp_c = cond.now.air_temp_c, cond.track_temp
            hc, sc = math.cos(tr.heading[ic]), math.sin(tr.heading[ic])
            s_c, x_c, y_c = tr.s[ic], tr.x[ic], tr.y[ic]
            n_pts = len(tr.s)
            ds_mean = tr.length / n_pts
            for i in range(4):
                px, py = self.pos[i]
                xw = s.x + px * cos_p - py * sin_p
                yw = s.y + px * sin_p + py * cos_p
                ds_w = (xw - x_c) * hc + (yw - y_c) * sc
                off = -(xw - x_c) * sc + (yw - y_c) * hc
                hint = (ic + int(round(ds_w / ds_mean))) % n_pts
                surf_grip[i], depth[i], surf[i] = cond.wheel_surface(s_c + ds_w, off, self.compound, s.vx, hint)
                if dynamic:
                    road[i] = cond.road_height(s_c + ds_w, off, False, hint)
            if dynamic:  # the floor runs down the middle of the car: asphalt only, never the kerb ridge
                off_c = -(s.x - x_c) * sc + (s.y - y_c) * hc
                ds_c = (s.x - x_c) * hc + (s.y - y_c) * sc
                floor = (cond.road_height(s_c + ds_c + self.a, off_c, True),
                         cond.road_height(s_c + ds_c - self.b, off_c, True))
        self.surface_grip = surf_grip  # what the driver feels through the seat (kerbs, grass, green track)
        road_dot = [(road[i] - s.road[i]) / dt for i in range(4)]
        s.road = road

        drs = bool(getattr(inp, "drs", False))
        grade, vcurv = self._vert
        g_eff = G * math.cos(grade) + s.vx * s.vx * vcurv  # a dip presses the car down, a crest lifts it
        if dynamic:
            ar = self._aero(drs)
            ft = self._tyre_spring(road, road_dot)
            geo = self._geometric_transfer()
            loads = [max(ft[i] + geo[i], 0.0) if ft[i] > 0 else 0.0 for i in range(4)]
        else:
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
        elif rpm_wheel < self.launch_rpm and throttle > 0:
            clutch, rpm = "slip", self.launch_rpm
            t_eng = max(engine_torque(p.engine, rpm, throttle, env.air_density), 0.0)
            t_axle = t_eng * ratio * p.gearbox.efficiency
        elif rpm_wheel < p.engine.idle_rpm:
            rpm, t_axle = p.engine.idle_rpm, 0.0
        else:
            clutch, rpm = "locked", rpm_wheel
            t_eng = engine_torque(p.engine, rpm, throttle, env.air_density)
            t_axle = t_eng * ratio * p.gearbox.efficiency
            extra_inertia = p.engine.inertia * ratio ** 2 / 2
        s.rpm = rpm
        # ERS (phase 14): the torque curve includes full deployment; take it out when the battery is
        # flat, and count the energy when it is not.
        e = p.ers
        omega = rpm * RPM_TO_RADS
        t_ers = 0.0
        if e.enabled and t_eng > 0 and omega > 0:
            t_full = full_throttle_torque(p.engine, rpm)
            t_ers_max = min(throttle * e.power / omega, 0.5 * throttle * t_full)
            soc = self.ers_energy / e.store
            strategy = min(max((soc - e.soc_floor) / (e.soc_taper - e.soc_floor), 0.0), 1.0)
            t_ers = t_ers_max * strategy
            lost = t_ers_max - t_ers  # the curve includes full deployment: take out what is not deployed
            t_eng -= lost
            t_axle -= lost * ratio * p.gearbox.efficiency
            if rpm > 9000.0:  # MGU-H: exhaust energy on throttle
                self.ers_energy = min(self.ers_energy + e.exhaust_harvest * throttle * dt, e.store)
            need = t_ers * omega * dt / e.deploy_efficiency
            if self.ers_energy < need:  # flat (or nearly): deploy what is left
                share = self.ers_energy / need if need > 0 else 0.0
                lost = t_ers * (1 - share)
                t_eng -= lost
                t_axle -= lost * ratio * p.gearbox.efficiency
                t_ers *= share
                need = self.ers_energy
            self.ers_energy -= need
        # Fuel (phase 10): burn follows the engine's own power; an empty tank stops the engine.
        fuel_on = p.fuel.start_kg > 0
        if fuel_on:
            if self.fuel <= 0.0:
                t_eng = 0.0
                t_axle = min(t_axle, 0.0)
                self.fuel_flow = 0.0
            else:
                fl = p.fuel
                p_ice = max(t_eng - t_ers, 0.0) * omega
                self.fuel_flow = fl.idle_flow + p_ice / (self._eta_ice * fl.lhv)
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
        # Limited-slip differential: clutch pack with preload and torque-sensing ramps; the torque
        # needed to hold the wheels together is applied up to the clutch capacity (stable at any dt).
        dw = s.w[2] - s.w[3]
        cap = c.diff_preload + (c.diff_power_lock if t_axle > 0 else c.diff_coast_lock) * abs(t_axle)
        hold = dw * 0.5 * (self.inertia[2] + extra_inertia) / (2 * dt)
        lock = max(-cap, min(cap, hold)) + c.diff_locking * dw
        drive = [0.0, 0.0, t_axle / 2 - lock, t_axle / 2 + lock]

        brake = s.brake
        bf = brake * c.max_brake_force * c.brake_front_bias * rad / 2
        br = brake * c.max_brake_force * (1 - c.brake_front_bias) * rad / 2
        tbrake = [bf, bf, br, br]
        bk = p.brakes
        if bk.thermal:
            tbrake = [tb * brake_friction(bk, tt) for tb, tt in zip(tbrake, s.brake_temp)]
        # MGU-K harvest under braking: brake-by-wire swaps rear friction braking for regeneration.
        p_harvest = 0.0
        if e.enabled and brake > 0 and s.vx > e.min_harvest_speed and s.gear > 0 and self.ers_energy < e.store:
            w_r = max(0.5 * (s.w[2] + s.w[3]), 1.0)
            t_regen = min(e.harvest_power / w_r, tbrake[2] + tbrake[3])
            tbrake[2] -= t_regen / 2
            tbrake[3] -= t_regen / 2
            drive[2] -= t_regen / 2
            drive[3] -= t_regen / 2
            p_harvest = t_regen * w_r
            self.ers_energy = min(self.ers_energy + p_harvest * e.harvest_efficiency * dt, e.store)
        p_brake = [0.0] * 4
        relax = c.relaxation_length
        tc = TyreConsts(tp)

        # Steering geometry (Ackermann) and camber to the road.
        ack = c.ackermann * steer * steer / (2 * c.wheelbase)
        roll_deg = s.ph / DEG if dynamic else 0.0
        fx_body = fy_body = mz = 0.0
        fyf = fyr = fxt = 0.0
        for i in range(4):
            px, py = self.pos[i]
            side = SIDE[i]
            vxi = s.vx - s.r * py
            vyi = s.vy + s.r * px
            if i < 2:
                d_i = steer + side * ack * c.track_front
                cos_d, sin_d = math.cos(d_i), math.sin(d_i)
                v_long = vxi * cos_d + vyi * sin_d
                v_lat = -vxi * sin_d + vyi * cos_d
            else:
                cos_d, sin_d = 1.0, 0.0
                v_long, v_lat = vxi, vyi
            denom = max(abs(v_long), V_MIN)
            kappa = (s.w[i] * rad - v_long) / denom
            alpha_geo = -math.atan(v_lat / denom)
            s.alpha_cmd[i] = alpha_geo
            if relax > 0:  # first-order lag over the relaxation length (exact for constant speed)
                alpha = s.alpha[i] + (alpha_geo - s.alpha[i]) * (1 - math.exp(-denom * dt / relax))
            else:
                alpha = alpha_geo
            s.kappa[i], s.alpha[i] = kappa, alpha
            k_eval = max(-1.0, min(1.0, kappa))
            fz = loads[i]
            g = grips[i]
            # Camber: the wheel follows part of the body roll. Leaning into the force (top of the tyre
            # towards the inside of the corner) helps the lateral grip; camber shrinks the patch in
            # braking and traction. Normalised so the static camber of the outside wheel = 1.
            cam0 = c.camber_front_deg if i < 2 else c.camber_rear_deg
            cam = cam0 - side * c.camber_roll_follow * roll_deg
            s.camber[i] = cam
            inward = (alpha > 0) == (side < 0)  # lateral force points to the car's centre line
            lean = cam if inward else -cam
            base = cam0 - c.camber_opt_deg
            gy = 1 - c.camber_lat_loss * ((lean - c.camber_opt_deg) ** 2 - base * base)
            gx = 1 - c.camber_long_loss * cam * cam
            fx, fy, dfdk = combined_fast(tc, k_eval, alpha, fz, g * gx, g * min(gy, 1.05))
            if abs(kappa) > 1.0:
                dfdk = 0.0

            # Wheel spin: linearised implicit step, then end-of-step force to the body.
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
            if tb > 0 and w_new > 0:  # the brake only heats while the wheel turns against it
                p_brake[i] = tb * 0.5 * (w_old + w_new)
            p_slide[i] = abs(fx * (w_new * rad - v_long)) + abs(fy * v_lat)

            if i < 2:
                fxb, fyb = fx * cos_d - fy * sin_d, fx * sin_d + fy * cos_d
                fyf += fyb
            else:
                fxb, fyb = fx, fy
                fyr += fyb
            fxt += fxb
            fx_body += fxb
            fy_body += fyb
            mz += px * fyb - py * fxb
        self._f_axle = (fyf, fyr, fxt)

        f_drag = ar.drag
        f_roll = c.crr * sum(loads) if s.vx > 0.1 else 0.0
        n_grass = surf.count("grass")
        f_grass = cond.sp.grass_drag * n_grass if (cond is not None and s.vx > 0.1) else 0.0
        plank = 0.0
        if dynamic:
            _, plank = self._suspension(dt, ar, fxt, fyf + fyr, f_drag, road_dot, g_eff, road, floor)
        f_plank = c.plank_friction * plank if s.vx > 0.1 else 0.0
        fx_body -= f_drag + f_roll + f_grass + f_plank + self.mass * G * math.sin(self._vert[0])
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
        if dynamic:
            s.ax, s.ay = ax, ay
        else:
            k = dt / (c.load_transfer_lag + dt)
            s.ax += k * (ax - s.ax)
            s.ay += k * (ay - s.ay)
            hf, hr = ride_heights(c, *self._axles)
            s.hf += k * (hf - s.hf)
            s.hr += k * (hr - s.hr)
        s.fz = loads
        if p.tyre_thermals:
            for i in range(4):
                wet = cond.sp.water_cooling * depth[i] * math.sqrt(s.speed) if cond is not None else 0.0
                thermal.update(self.thermal, s.tyres[i], p_slide[i], loads[i], s.speed,
                               env.air_temp_c, env.track_temp_c, dt, wet)
        if bk.thermal:
            h = bk.h_base + bk.h_per_ms * s.speed
            for i in range(4):
                tt = s.brake_temp[i]
                s.brake_temp[i] = tt + dt * (bk.heat_share * p_brake[i] - h * (tt - env.air_temp_c)) / bk.heat_capacity
        if cond is not None:
            cond.update(dt, float(cond.track.s[cond.i]), sum(p_slide))
        s.t += dt
        s.shift_timer = max(s.shift_timer - dt, 0.0)
        if fuel_on and self.fuel > 0:
            self.fuel = max(self.fuel - self.fuel_flow * dt, 0.0)
            self._update_mass()
        if not row:
            return None

        row = {
            "t": s.t, "x": s.x, "y": s.y, "psi": s.psi, "vx": s.vx, "vy": s.vy, "v": s.speed,
            "yaw_rate": s.r, "ax": ax, "ay": ay, "sideslip": s.sideslip, "steer": steer,
            "throttle": throttle, "brake": brake, "steer_cmd": target, "throttle_cmd": thr_t, "brake_cmd": brk_t,
            "gear": s.gear, "rpm": rpm,
            "clutch": clutch, "tc_cut": tc_cut, "downforce": downforce, "dist": s.dist,
            "drag": f_drag, "ClA": ar.ClA, "CdA": ar.CdA, "aero_balance": ar.front_share,
            "ride_front": s.hf, "ride_rear": s.hr, "drs": drs,
            "grade": self._vert[0], "vert_curv": self._vert[1], "grass_wheels": n_grass,
            "fuel": self.fuel, "fuel_flow": self.fuel_flow, "mass": self.mass,
            "front_weight": self.front_weight, "diff_lock": lock,
            "ers_soc": self.ers_energy / e.store if e.enabled else 0.0, "ers_deploy_kw": t_ers * omega / 1e3,
            "ers_harvest_kw": p_harvest / 1e3,
            "roll_deg": s.ph / DEG, "pitch_deg": s.th / DEG, "heave": s.z, "plank": plank,
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
            row[f"tbrake_{n}"] = s.brake_temp[i]
            row[f"camber_{n}"] = s.camber[i]
            row[f"road_{n}"] = road[i]
        return row
