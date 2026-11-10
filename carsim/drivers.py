"""Scripted drivers: functions (state, car) -> Inputs."""
import math
from .powertrain import full_throttle_torque, overall_ratio
from .vehicle import Inputs, LongitudinalCar, State


class FullThrottleDriver:
    """Flat out with automatic upshifts.

    shift_rpm=None shifts at the optimum: as soon as the next gear gives more
    wheel force at the current speed.
    """

    def __init__(self, shift_rpm: float | None = None):
        self.shift_rpm = shift_rpm

    def __call__(self, s: State, car: LongitudinalCar) -> Inputs:
        gb, eng = car.p.gearbox, car.p.engine
        gear = s.gear
        if s.shift_timer == 0 and gear < len(gb.ratios):
            rpm = car.wheel_rpm_to_engine(s.v, gear)
            rpm_next = car.wheel_rpm_to_engine(s.v, gear + 1)
            if self.shift_rpm is not None:
                up = rpm >= self.shift_rpm
            else:
                f_now = full_throttle_torque(eng, rpm) * overall_ratio(gb, gear)
                f_next = full_throttle_torque(eng, rpm_next) * overall_ratio(gb, gear + 1)
                up = rpm_next > car.launch_rpm and (f_next > f_now or rpm >= eng.rev_limit_rpm - 50)
            if up:
                gear += 1
        return Inputs(throttle=1.0, brake=0.0, gear=gear)


class BrakeFromSpeedDriver:
    """Accelerate to target speed, then brake fully (throttle off)."""

    def __init__(self, v_target: float, brake: float = 1.0):
        self.v_target = v_target
        self.brake = brake
        self.braking = False
        self._accel = FullThrottleDriver()

    def __call__(self, s: State, car: LongitudinalCar) -> Inputs:
        if s.v >= self.v_target:
            self.braking = True
        if self.braking:
            return Inputs(throttle=0.0, brake=self.brake)
        return self._accel(s, car)


class ThresholdBrakeDriver(BrakeFromSpeedDriver):
    """Like BrakeFromSpeedDriver, but eases the pedal when a wheel slips past the tyre's peak.

    This is what a skilled driver does: brake at the edge of lock-up.
    """

    def __init__(self, v_target: float, slip_limit: float | None = None):
        super().__init__(v_target)
        self.slip_limit = slip_limit
        self.pedal = 1.0

    def __call__(self, s: State, car: LongitudinalCar) -> Inputs:
        inp = super().__call__(s, car)
        if not self.braking:
            return inp
        limit = self.slip_limit or car.p.tyres.longitudinal.peak_slip
        worst = -min(s.kappa_front, s.kappa_rear)
        if worst > limit:
            self.pedal = max(self.pedal - 0.02, 0.2)
        else:
            self.pedal = min(self.pedal + 0.005, 1.0)
        return Inputs(throttle=0.0, brake=self.pedal)


def auto_gear(s, car) -> int:
    """Optimal upshift (same rule as FullThrottleDriver) plus a downshift when the revs drop low."""
    gear = FullThrottleDriver()(s, car).gear
    if s.shift_timer == 0 and gear > 1 and car.wheel_rpm_to_engine(s.v, gear) < 10000:
        gear -= 1
    return gear


class SkidpadDriver:
    """Drive a left-hand circle of radius R (centre at (0, R)), slowly raising speed.

    Steering: pure pursuit on the circle plus a correction on radius error.
    Speed: proportional throttle towards a target that ramps up at `ramp` m/s^2.
    """

    def __init__(self, radius: float, v_start: float = 10.0, ramp: float = 0.4, v_max: float = 120.0):
        self.R, self.v_start, self.ramp, self.v_max = radius, v_start, ramp, v_max

    def target_speed(self, t: float) -> float:
        return min(self.v_start + self.ramp * t, self.v_max)

    def radius_error(self, s) -> float:
        return math.hypot(s.x, s.y - self.R) - self.R

    def __call__(self, s, car):
        R, L = self.R, car.p.chassis.wheelbase
        ld = max(6.0, 0.6 * s.speed)
        phi = math.atan2(s.x, R - s.y) + ld / R
        tx, ty = R * math.sin(phi), R - R * math.cos(phi)
        dx, dy = tx - s.x, ty - s.y
        lat = -dx * math.sin(s.psi) + dy * math.cos(s.psi)  # target offset to the left, body frame
        steer = math.atan(2 * L * lat / (ld * ld)) + 0.05 * self.radius_error(s)
        throttle = min(max(0.5 * (self.target_speed(s.t) - s.speed), 0.0), 1.0)
        return Inputs(throttle=throttle, brake=0.0, gear=auto_gear(s, car), steer=steer)


class LapDriver:
    """Drives laps of a Track: pure-pursuit steering plus a grip-limited speed profile.

    The speed profile is the classic lap-time-simulation shape: corner speed from
    lateral grip (with downforce), then a backward pass for braking and a forward
    pass for acceleration, both sharing grip with cornering (friction circle).
    The driver "feels" the tyres: the profile is rebuilt every few seconds using the
    car's current tyre grip, scaled by `margin`. With track conditions (phases 8–9) the
    driver also knows the grip along the lap (rubber, water), keeps below the
    aquaplaning speed and allows for slopes.
    """

    def __init__(self, track, margin: float = 0.93, lateral_efficiency: float = 0.86, line="centre"):
        # lateral_efficiency: how much of the "one big tyre" grip the whole car can use in a
        # corner (load transfer, balance). 0.86 is calibrated against the skidpad results.
        # line: "centre" (follow the centreline), "racing" (minimum-curvature line, phase 11),
        # or (line Track, offsets) from racing_line().
        self.track, self.margin, self.lat_eff = track, margin, lateral_efficiency
        self.path, self.offsets = track, None
        if line == "racing":
            from .racing_line import cached_racing_line
            self.path, self.offsets = cached_racing_line(track)
        elif isinstance(line, tuple):
            self.path, self.offsets = line
        self.yaw_damping = 0.03  # rad of steer per rad/s of unwanted yaw rate
        self.traction_share = 0.75  # at low grip, drive force capped at this share of rear grip
        self.brake_eff = 0.85  # share of tyre grip usable in braking (fixed bias, lock-up modulation)
        # Confidence: each scare (big slide, front washing out) costs 2% pace; clean driving
        # earns it back slowly. Low grip also means a bigger margin from the start.
        self.confidence = 1.0
        self._scare_t = -1e9
        self.i = 0
        self.lap = 0
        self.lap_start = 0.0
        self.lap_times: list[float] = []
        self.off_track = False
        self._profile = None
        self._profile_t = -1e9
        self._last_s = 0.0
        self._steer = 0.0
        self._throttle = 0.0
        self._t_last = 0.0
        self._slide = 0.0
        self.slide_filter = 0.05  # s, smoothing of the felt slide

    # ---- speed profile ----------------------------------------------------
    def build_profile(self, car, grip: float):
        """Speed target along the lap. Uses the tyre model's load-sensitive grip."""
        import numpy as np
        from .params import G
        p, rho, tr, tp = car.p, car.env.air_density, self.path, car.p.tyres
        m = getattr(car, "mass", p.chassis.mass)  # includes fuel (phase 10)
        down = 0.5 * rho * p.aero.ClA / m  # downforce per mass per v^2
        cond = getattr(car, "cond", None)
        tyre_grip = grip
        if cond is not None:
            grip = tyre_grip * cond.grip_profile(car.compound)  # per point along the lap
            v_aqua = 0.92 * cond.aquaplane_profile(car.compound)
            slope = np.sin(self.track.grade)
        else:
            grip = np.full(len(tr.s), tyre_grip * getattr(getattr(car, "compound", None), "dry_grip", 1.0))
            v_aqua = np.full(len(tr.s), np.inf)
            slope = np.zeros(len(tr.s))

        g_mean = float(np.mean(grip)) * tp.grip_scale
        margin = self.margin * (0.7 + 0.3 * min(g_mean, 1.0)) * self.confidence

        def mu(curve, v2):
            fz = m * (G + down * v2) / 4  # average tyre load
            dfz = (fz - tp.nominal_load) / tp.nominal_load
            eff = self.lat_eff if curve is tp.lateral else 1.0
            return curve.mu0 * np.maximum(1 + tp.load_sensitivity * dfz, 0.5) * tp.grip_scale * grip * margin * eff

        def mu_s(curve, v2, gj):  # scalar version for the passes below
            fz = m * (G + down * v2) / 4
            f = max(1 + tp.load_sensitivity * (fz - tp.nominal_load) / tp.nominal_load, 0.5)
            eff = self.lat_eff if curve is tp.lateral else 1.0
            return curve.mu0 * f * tp.grip_scale * gj * margin * eff

        k = np.abs(tr.curvature)
        v2 = np.full(len(k), 90.0 ** 2)
        for _ in range(30):  # fixed point: v^2 k = mu(v) (g + down v^2)
            mu_y = mu(tp.lateral, v2)
            with np.errstate(divide="ignore", invalid="ignore"):
                v2_new = np.where(k > 1e-6, mu_y * G / np.maximum(k - mu_y * down, 1e-9), 90.0 ** 2)
            v2_new = np.where(k - mu_y * down > 1e-9, v2_new, 90.0 ** 2)
            v2 = np.minimum(0.5 * v2 + 0.5 * v2_new, 90.0 ** 2)
        v = np.minimum(np.sqrt(v2), v_aqua)
        n, ds = len(v), np.diff(np.append(tr.s, tr.length))
        p_max = 7.0e5 * 0.92  # W at the wheels, rough
        lat_cap = lambda vv, gj: mu_s(tp.lateral, vv * vv, gj) * (G + down * vv * vv)
        long_cap = lambda vv, gj: mu_s(tp.longitudinal, vv * vv, gj) * (G + down * vv * vv)
        v = [float(x) for x in v]
        k = [float(x) for x in k]
        ds = [float(x) for x in ds]
        g_l = [float(x) for x in grip]
        sl = [float(x) for x in slope]
        for _ in range(2):  # two loops so the profile closes on itself
            for j in range(n - 1, -n, -1):  # backward: braking (uphill helps, downhill hurts)
                a, b = j % n, (j + 1) % n
                vb = v[b]
                lat = vb * vb * k[a] / lat_cap(vb, g_l[a])
                a_av = (self.brake_eff * long_cap(vb, g_l[a]) * math.sqrt(max(1 - min(lat, 1.0) ** 2, 0.0))
                        + 0.5 * rho * p.aero.CdA * vb * vb / m + G * sl[a])
                v[a] = min(v[a], math.sqrt(vb * vb + 2 * max(a_av, 0.5) * ds[a]))
            for j in range(0, 2 * n):  # forward: traction and power (uphill costs)
                a, b = j % n, (j + 1) % n
                va = max(v[a], 5.0)
                a_trac = 0.55 * long_cap(va, g_l[a])
                a_pow = p_max / (m * va) - 0.5 * rho * p.aero.CdA * va * va / m - G * sl[a]
                lat = va * va * k[a] / lat_cap(va, g_l[a])
                a_av = max(min(a_trac, a_pow), 0.0) * math.sqrt(max(1 - min(lat, 1.0) ** 2, 0.0))
                v[b] = min(v[b], math.sqrt(va * va + 2 * a_av * ds[a]))
        self._profile = np.array(v)
        self._profile_grip = tyre_grip

    def _traction_cap(self, s, car) -> float:
        from .params import G
        from .powertrain import RPM_TO_RADS, full_throttle_torque, overall_ratio
        p, tp = car.p, car.p.tyres
        g = car.grip_factors()
        grip = (g[2] + g[3]) / 2 * tp.grip_scale
        cond = getattr(car, "cond", None)
        if cond is not None:
            grip *= cond.grip_profile(car.compound)[cond.i]
        else:
            grip *= getattr(getattr(car, "compound", None), "dry_grip", 1.0)
        share = self.traction_share + (1 - self.traction_share) * min(max((grip - 0.6) / 0.3, 0.0), 1.0)  # no cap at grip >= 0.9
        if share >= 1.0:
            return 1.0
        c = p.chassis
        v = max(s.speed, 1.0)
        m = getattr(car, "mass", c.mass)
        fw = getattr(car, "front_weight", c.front_weight)
        n_rear = m * G * (1 - fw) + 0.5 * car.env.air_density * p.aero.ClA * v * v * (1 - p.aero.front_balance)
        ratio = overall_ratio(p.gearbox, s.gear)
        rpm = max((s.w[2] + s.w[3]) / 2 * ratio / RPM_TO_RADS, car.launch_rpm)
        f_full = full_throttle_torque(p.engine, rpm) * ratio * p.gearbox.efficiency / c.wheel_radius
        if f_full <= 0:
            return 1.0
        allow = share * tp.longitudinal.mu0 * grip * n_rear
        return min(max(allow / f_full, 0.05), 1.0)

    # ---- control ------------------------------------------------------------
    def __call__(self, s, car):
        tr = self.path
        cond = getattr(car, "cond", None)
        if self.offsets is not None and cond is not None and cond.line_offset is not self.offsets:
            cond.set_racing_line(self.offsets)  # rubber and the drying line follow the cars' line
        if hasattr(car, "grip_factors"):
            g = car.grip_factors()
            # The weaker axle sets the pace, and its worse tyre counts for more.
            axle = lambda a, b: 0.6 * min(a, b) + 0.4 * max(a, b)
            grip = min(axle(g[0], g[1]), axle(g[2], g[3]))
        else:
            grip = 1.0
        # Confidence: scares cost pace, clean running earns it back.
        if hasattr(s, "alpha"):
            front_over = abs(s.alpha[0] + s.alpha[1]) / 2 / car.p.tyres.lateral.peak_slip
            scared = abs(s.sideslip) > math.radians(4.0) or front_over > 1.4
            if scared and s.t - self._scare_t > 0.5:
                self.confidence = max(self.confidence * 0.98, 0.75)
                self._scare_t = s.t
                self._profile_t = -1e9  # replan now
            elif s.t - self._scare_t > 1.0:
                self.confidence = min(self.confidence + 0.002 * 0.001, 1.0)  # ~+0.2% per clean second
        changing = getattr(car, "cond", None) is not None  # track and weather evolve
        if self._profile is None or (s.t - self._profile_t > 2.0 and
                                     (changing or abs(grip - self._profile_grip) > 0.003)):
            self.build_profile(car, grip)
            self._profile_t = s.t

        self.i = tr.nearest(s.x, s.y, self.i)
        s_now = tr.s[self.i]
        if s_now < self._last_s - tr.length / 2:  # crossed the line
            self.lap += 1
            self.lap_times.append(s.t - self.lap_start)
            self.lap_start = s.t
        self._last_s = s_now
        err = tr.cross_track_error(self.i, s.x, s.y)  # to the line being driven
        if abs(self.track.cross_track_error(self.i, s.x, s.y)) > self.track.width / 2 + 2:
            self.off_track = True

        # Steering: pure pursuit to a point ahead on the centreline.
        ld = max(8.0, 0.45 * s.speed)
        j = tr.index_at(s_now + ld)
        dx, dy = tr.x[j] - s.x, tr.y[j] - s.y
        lat = -dx * math.sin(s.psi) + dy * math.cos(s.psi)
        dist2 = dx * dx + dy * dy
        steer = math.atan(2 * car.p.chassis.wheelbase * lat / max(dist2, 1.0)) - 0.02 * err
        # Damp the car's rotation: steer against yaw rate that the path does not ask for.
        if hasattr(s, "r"):
            r_path = s.speed * tr.curvature[self.i]
            steer -= self.yaw_damping * (s.r - r_path)
        # Feel the front: no extra lock once the fronts are past their peak slip angle.
        if hasattr(s, "alpha"):
            a_front = abs(s.alpha[0] + s.alpha[1]) / 2
            if a_front > 1.1 * car.p.tyres.lateral.peak_slip and abs(steer) > abs(self._steer):
                steer = self._steer
        self._steer = steer

        # Speed: track the profile a little ahead (reaction + brake build-up).
        v_t = self._profile[tr.index_at(s_now + 0.25 * s.speed + 3)]
        dv = v_t - s.speed
        throttle = min(max(0.4 * dv + 0.1, 0.0), 1.0) if dv > -0.5 else 0.0
        brake = min(max(-0.25 * dv, 0.0), 1.0) if dv < -0.5 else 0.0
        lock = -min(s.kappa_front, s.kappa_rear)
        if brake > 0 and lock > 0.25:  # a wheel is locking: release so it spins up again
            brake *= 0.3
        elif brake > 0 and lock > 0.12:  # past the peak: ease off
            brake *= 0.7
        # Feel the rear: ease the throttle as the car starts to slide (sideslip over ~1.5 deg).
        # Smoothed feel (0.3 s): reacting to every twitch would pump a weave.
        dt_f = max(s.t - self._t_last, 0.0)
        self._slide += (abs(math.degrees(s.sideslip)) - self._slide) * min(dt_f / self.slide_filter, 1.0)
        throttle *= min(max(1 - (self._slide - 1.5) / 2.5, 0.0), 1.0)
        # Feel the front: if the fronts are well past their peak (understeer), lift and
        # scrub a little speed until they grip again.
        if hasattr(s, "alpha"):
            over = abs(s.alpha[0] + s.alpha[1]) / 2 / car.p.tyres.lateral.peak_slip
            if over > 1.25:
                throttle = 0.0
                brake = max(brake, min(0.25 * (over - 1.25) / 0.5, 0.25))
        # Low grip: keep the drive force well inside the rear tyres' grip (short-shift, gentle
        # throttle in the wet) — near the traction limit a low-grip car loses yaw stability.
        if hasattr(s, "w") and throttle > 0:
            throttle = min(throttle, self._traction_cap(s, car))
        # Feed the throttle in: at most full travel in 0.4 s (no rate limit lifting off).
        dt = s.t - self._t_last if s.t > self._t_last else 0.001
        throttle = min(throttle, self._throttle + 2.5 * dt)
        self._throttle, self._t_last = throttle, s.t
        return Inputs(throttle=throttle, brake=brake, gear=auto_gear(s, car), steer=steer)
