"""Track evolution and weather — roadmap phases 8 and 9.

`Conditions` ties a Track to a Weather timeline and keeps the state of the
surface along the lap, in bins of `bin_m` metres:

  * rubber  — laid by sliding tyres on the line the cars drive; raises grip from
              "green" towards fully rubbered. Rain washes it away.
  * water   — standing water depth (mm): rain fills it, it drains, evaporates
              (warm track, wind) and is cleared by cars on their line (drying line).
  * track temperature — air temperature plus sun (less under cloud), cooled by rain.

The car asks `Conditions` for what each wheel sits on (track, kerb or grass, plus
grip from rubber and water), for the slope and vertical curvature under it, and
for the wind. Tyre compounds (slick / intermediate / wet) decide how grip falls
with water depth, when the tyre aquaplanes, and their temperature window.

`cars_on_track` scales how much rubber is laid and water cleared per lap: one
simulated car stands in for a full field.
"""
import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81


# ---- tyre compounds -----------------------------------------------------------
@dataclass(frozen=True)
class Compound:
    name: str
    dry_grip: float  # grip on a dry track relative to the slick
    wet_loss: float  # share of grip lost on deep water (before aquaplaning)
    water_scale: float  # mm: depth at which ~63% of wet_loss has set in
    aquaplane_coeff: float  # m/s * sqrt(mm): aquaplaning starts at coeff / sqrt(depth)
    t_opt: float  # deg C, peak-grip tread temperature
    wear_factor: float  # wear multiplier on a dry track (soft wet rubber wears fast)


COMPOUNDS = {
    "slick": Compound("slick", 1.00, 0.62, 0.25, 50.0, 100.0, 1.0),
    "intermediate": Compound("intermediate", 0.90, 0.30, 0.6, 75.0, 80.0, 2.0),
    "wet": Compound("wet", 0.82, 0.18, 1.2, 100.0, 65.0, 3.0),
}


def wet_grip(c: Compound, depth_mm: float, speed: float) -> float:
    """Grip factor of compound c on water depth (mm) at speed (m/s), relative to a dry slick."""
    g = c.dry_grip * (1 - c.wet_loss * (1 - math.exp(-depth_mm / c.water_scale)))
    if depth_mm > 0.05:
        v_aqua = c.aquaplane_coeff / math.sqrt(depth_mm)
        if speed > v_aqua:
            g *= max(0.15, 1 - 2.0 * (speed / v_aqua - 1))
    return g


def aquaplane_speed(c: Compound, depth_mm: float) -> float:
    return math.inf if depth_mm <= 0.05 else c.aquaplane_coeff / math.sqrt(depth_mm)


# ---- weather --------------------------------------------------------------------
@dataclass
class WeatherPoint:
    t: float  # s from session start
    air_temp_c: float = 22.0
    rain_mm_h: float = 0.0
    cloud: float = 0.2  # 0 clear .. 1 overcast
    wind_ms: float = 0.0
    wind_from_deg: float = 0.0  # direction the wind blows FROM, world frame (0 = from +x)


@dataclass
class Weather:
    """Piecewise-linear weather timeline."""
    points: list = field(default_factory=lambda: [WeatherPoint(0.0)])

    def at(self, t: float) -> WeatherPoint:
        pts = self.points
        if t <= pts[0].t:
            return pts[0]
        for a, b in zip(pts, pts[1:]):
            if t <= b.t:
                f = (t - a.t) / (b.t - a.t)
                lerp = lambda u, v: u + f * (v - u)
                return WeatherPoint(t, lerp(a.air_temp_c, b.air_temp_c), lerp(a.rain_mm_h, b.rain_mm_h),
                                    lerp(a.cloud, b.cloud), lerp(a.wind_ms, b.wind_ms),
                                    lerp(a.wind_from_deg, b.wind_from_deg))
        return pts[-1]

    @classmethod
    def dry(cls, air_temp_c=22.0, cloud=0.2):
        return cls([WeatherPoint(0.0, air_temp_c, 0.0, cloud)])

    @classmethod
    def constant(cls, **kw):
        return cls([WeatherPoint(0.0, **kw)])

    @classmethod
    def shower(cls, start=300.0, peak=600.0, end=1200.0, rain_mm_h=8.0, air_temp_c=20.0):
        """Dry, then a shower building to `rain_mm_h`, then drying."""
        return cls([WeatherPoint(0.0, air_temp_c, 0.0, 0.3),
                    WeatherPoint(start, air_temp_c, 0.0, 0.8),
                    WeatherPoint(start + 60.0, air_temp_c - 1.0, rain_mm_h * 0.5, 1.0),
                    WeatherPoint(peak, air_temp_c - 3.0, rain_mm_h, 1.0),
                    WeatherPoint(end, air_temp_c - 3.0, 0.0, 0.9),
                    WeatherPoint(end + 600.0, air_temp_c - 1.0, 0.0, 0.4)])


# ---- surface parameters -----------------------------------------------------------
@dataclass
class SurfaceParams:
    green_grip: float = 0.93  # grip of an unrubbered track
    rubber_scale: float = 2.0e6  # J of sliding energy per bin to reach ~63% of full rubber (~30 laps of a field)
    rubber_wash: float = 0.15  # share of rubber washed away per mm of rain
    kerb_grip: float = 0.85
    grass_grip: float = 0.50
    grass_drag: float = 1500.0  # N per wheel on grass (rolling in soft ground)
    drain_tau: float = 400.0  # s, standing water drains with this time constant
    evap_mm_h: float = 0.6  # evaporation on a 30 C track with light wind
    line_clear_per_lap: float = 0.25  # share of water a full field clears off its line each lap
    line_width: float = 3.0  # m, half-width of the band the cars run on
    solar_gain: float = 18.0  # K the sun adds to track over air on a clear day
    track_tau: float = 1200.0  # s, track temperature time constant (dry)
    rain_track_tau: float = 300.0  # s, faster when rain is cooling it
    water_cooling: float = 25.0  # W/K per mm of water per sqrt(m/s), extra tyre cooling


class Conditions:
    def __init__(self, track, weather: Weather | None = None, surface: SurfaceParams | None = None,
                 bin_m: float = 5.0, cars_on_track: int = 20, rubber: float = 0.0, water_mm: float = 0.0,
                 track_temp_c: float | None = None):
        self.track = track
        self.weather = weather or Weather.dry()
        self.sp = surface or SurfaceParams()
        self.bin_m = bin_m
        self.n_bins = int(math.ceil(track.length / bin_m))
        self.cars = cars_on_track
        self.rubber = np.full(self.n_bins, float(rubber))  # J of sliding energy laid per bin
        self.water = np.full(self.n_bins, float(water_mm))  # mm, on the racing line
        w0 = self.weather.at(0.0)
        self.track_temp = (w0.air_temp_c + self.sp.solar_gain * (1 - w0.cloud)) if track_temp_c is None else track_temp_c
        self.t = 0.0
        self.i = 0  # car's centreline index
        self.now = w0
        self._last_bin = None
        self._off_line = float(water_mm)  # water off the racing line (nobody clears it)
        self.line_offset = np.zeros(len(track.s))  # where the cars drive, m from the centreline

    def set_racing_line(self, offsets):
        """Tell the track where the cars drive (rubber and the drying line follow it)."""
        self.line_offset = offsets

    # ---- lookups ------------------------------------------------------------------
    def bin_of(self, s: float) -> int:
        return int((s % self.track.length) // self.bin_m) % self.n_bins

    def rubber_grip(self, b: int) -> float:
        sp = self.sp
        return sp.green_grip + (1 - sp.green_grip) * (1 - math.exp(-self.rubber[b] / sp.rubber_scale))

    def locate(self, x: float, y: float):
        self.i = self.track.nearest(x, y, self.i)
        return self.i

    def wheel_surface(self, s_wheel: float, offset: float, compound: Compound, speed: float):
        """(grip factor, water depth mm, surface name) for a wheel at s along the lap, lateral offset."""
        b = self.bin_of(s_wheel)
        surf = self.track.surface_at(offset)
        on_line = abs(offset - self.line_offset[self.track.index_at(s_wheel)]) <= self.sp.line_width
        depth = self.water[b] if on_line else max(self.water[b], self._off_line)
        grip = wet_grip(compound, depth, speed)
        if surf == "track":
            # Rubber lives on the line; off it the track is green.
            grip *= self.rubber_grip(b) if on_line else self.sp.green_grip
        elif surf == "kerb":
            grip *= self.sp.kerb_grip
        else:
            grip *= self.sp.grass_grip
        return grip, depth, surf

    def vertical(self, i: int):
        return float(self.track.grade[i]), float(self.track.vert_curv[i])

    def wind_body(self, psi: float):
        """Wind velocity (the air's motion) in the car's body frame: (forward, left) m/s."""
        w = self.now
        if w.wind_ms == 0.0:
            return 0.0, 0.0
        ang = math.radians(w.wind_from_deg) + math.pi  # direction the air moves to
        wx, wy = w.wind_ms * math.cos(ang), w.wind_ms * math.sin(ang)
        c, s = math.cos(psi), math.sin(psi)
        return wx * c + wy * s, -wx * s + wy * c

    def grip_profile(self, compound: Compound, speeds=None) -> np.ndarray:
        """Grip factor on the line at each centreline point (what the driver plans with)."""
        tr = self.track
        bins = ((tr.s % tr.length) // self.bin_m).astype(int) % self.n_bins
        rub = self.sp.green_grip + (1 - self.sp.green_grip) * (1 - np.exp(-self.rubber / self.sp.rubber_scale))
        depth = self.water[bins]
        c = compound
        g = c.dry_grip * (1 - c.wet_loss * (1 - np.exp(-depth / c.water_scale)))
        return g * rub[bins]

    def aquaplane_profile(self, compound: Compound) -> np.ndarray:
        tr = self.track
        bins = ((tr.s % tr.length) // self.bin_m).astype(int) % self.n_bins
        d = self.water[bins]
        with np.errstate(divide="ignore"):
            return np.where(d > 0.05, compound.aquaplane_coeff / np.sqrt(np.maximum(d, 1e-9)), np.inf)

    # ---- evolution --------------------------------------------------------------------
    def update(self, dt: float, s_car: float, sliding_power: float):
        """Advance weather, track temperature, water and rubber by dt. The car is at s_car."""
        self.t += dt
        w = self.weather.at(self.t)
        self.now = w
        sp = self.sp
        # Track temperature.
        target = w.air_temp_c + sp.solar_gain * (1 - w.cloud)
        tau = sp.track_tau
        if w.rain_mm_h > 0.1:
            target, tau = w.air_temp_c - 1.0, sp.rain_track_tau
        self.track_temp += (target - self.track_temp) * dt / tau
        # Water everywhere: rain in, drainage and evaporation out.
        rain = w.rain_mm_h / 3600.0
        evap = sp.evap_mm_h / 3600.0 * max(self.track_temp - 5.0, 0.0) / 25.0 * (1 + 0.1 * w.wind_ms)
        decay = dt / sp.drain_tau
        self.water += rain * dt - self.water * decay
        self.water = np.maximum(self.water - evap * dt, 0.0)
        off = self._off_line
        self._off_line = max(off + rain * dt - off * decay - evap * dt, 0.0)
        # Rain washes rubber away.
        if rain > 0:
            self.rubber *= max(1 - sp.rubber_wash * rain * dt, 0.0)
        # The car's own bin: lay rubber, clear water (once per pass, scaled to the field).
        b = self.bin_of(s_car)
        self.rubber[b] += sliding_power * dt * self.cars
        if b != self._last_bin:
            self.water[b] *= 1 - sp.line_clear_per_lap * self.cars / 20.0
            self._last_bin = b

    def summary(self) -> dict:
        return {"t": self.t, "air_temp": self.now.air_temp_c, "track_temp": self.track_temp,
                "rain": self.now.rain_mm_h, "water_line_mm": float(self.water.mean()),
                "water_off_line_mm": self._off_line,
                "rubber_grip": float(np.mean([self.rubber_grip(b) for b in range(self.n_bins)]))}
