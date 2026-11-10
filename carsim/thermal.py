"""Tyre temperature, pressure and wear — roadmap phase 7.

Each tyre has two thermal masses: the surface (tread skin) and the core
(carcass + inflation gas). Heat sources and sinks:

  * sliding friction: |Fx * slip speed_x| + |Fy * slip speed_y|; a share goes into the tread
  * carcass flexing: proportional to load * speed, heats the core
  * surface -> core conduction
  * surface -> air convection (grows with speed) and surface -> track conduction
  * core -> rim/air

Grip factor = temperature window x pressure window x wear. Wear grows with
sliding energy and accelerates when the tread runs hot. Past the wear "cliff"
grip falls away quickly.
"""
import math
from dataclasses import dataclass


@dataclass
class ThermalParams:
    c_surface: float = 2500.0  # J/K, tread skin heat capacity
    c_core: float = 14000.0  # J/K, carcass + gas
    k_surface_core: float = 120.0  # W/K
    h_air_base: float = 5.0  # W/K, surface to air at standstill
    h_air_per_ms: float = 0.6  # W/K per m/s of road speed
    k_track: float = 15.0  # W/K, surface to track
    k_core_out: float = 50.0  # W/K, core to rim, brake ducts and air
    slide_heat_share: float = 0.35  # share of sliding power that enters the tyre
    flex_coeff: float = 0.016  # core heating = coeff * Fz * v (W)

    t_opt: float = 100.0  # deg C, peak-grip tread temperature
    t_window: float = 30.0  # deg C, hot side of the window
    t_window_cold: float = 45.0  # deg C, cold side: a cold tyre loses grip more gently than an overheated one  # deg C, half-width of the window
    temp_penalty: float = 0.08  # grip lost at one window-width off the optimum
    surface_weight: float = 0.7  # effective grip temperature = mix of surface and core

    pressure_set: float = 1.45  # bar gauge, cold set at blanket temperature
    pressure_opt: float = 1.60  # bar gauge, hot optimum
    pressure_penalty: float = 2.0

    wear_rate: float = 0.5e-7  # tread fraction per joule of sliding energy (at t_opt)
    wear_temp_scale: float = 30.0  # wear doubles roughly every 21 deg C above t_opt
    wear_grip_loss: float = 0.10  # linear grip loss over the useful life
    cliff: float = 0.75  # wear fraction where the cliff starts
    cliff_loss: float = 1.2  # extra grip loss per unit wear past the cliff

    blanket_temp: float = 70.0  # deg C at the start of a stint


@dataclass
class TyreThermalState:
    t_surface: float
    t_core: float
    wear: float = 0.0  # 0 new .. 1 worn through
    p_slide: float = 0.0  # last sliding power, W (for telemetry)

    @classmethod
    def from_blankets(cls, tp: ThermalParams):
        return cls(tp.blanket_temp, tp.blanket_temp)


def pressure(tp: ThermalParams, st: TyreThermalState) -> float:
    """Hot gauge pressure (bar) from the core temperature: ideal gas, constant volume."""
    abs_set = tp.pressure_set + 1.013
    return abs_set * (st.t_core + 273.15) / (tp.blanket_temp + 273.15) - 1.013


def grip_factor(tp: ThermalParams, st: TyreThermalState) -> float:
    t_eff = tp.surface_weight * st.t_surface + (1 - tp.surface_weight) * st.t_core
    window = tp.t_window if t_eff > tp.t_opt else tp.t_window_cold
    g_temp = max(0.6, 1 - tp.temp_penalty * ((t_eff - tp.t_opt) / window) ** 2)
    dp = (pressure(tp, st) - tp.pressure_opt) / (tp.pressure_opt + 1.013)
    g_press = max(0.85, 1 - tp.pressure_penalty * dp * dp)
    g_wear = 1 - tp.wear_grip_loss * min(st.wear, 1.0)
    if st.wear > tp.cliff:
        g_wear -= tp.cliff_loss * (st.wear - tp.cliff)
    return max(g_temp * g_press * max(g_wear, 0.3), 0.3)


def update(tp: ThermalParams, st: TyreThermalState, p_slide: float, fz: float, v: float,
           t_air: float, t_track: float, dt: float, extra_cooling: float = 0.0):
    """Advance one tyre's temperatures and wear by dt. extra_cooling (W/K) = e.g. water on track."""
    st.p_slide = p_slide
    q_slide = tp.slide_heat_share * p_slide
    q_flex = tp.flex_coeff * fz * abs(v)
    q_sc = tp.k_surface_core * (st.t_surface - st.t_core)
    q_air = (tp.h_air_base + tp.h_air_per_ms * abs(v)) * (st.t_surface - t_air)
    q_track = tp.k_track * (st.t_surface - t_track) + extra_cooling * (st.t_surface - min(t_air, t_track))
    q_core_out = tp.k_core_out * (st.t_core - t_air)
    st.t_surface += dt * (q_slide - q_sc - q_air - q_track) / tp.c_surface
    st.t_core += dt * (q_flex + q_sc - q_core_out) / tp.c_core
    hot = math.exp((st.t_surface - tp.t_opt) / tp.wear_temp_scale)
    st.wear = min(st.wear + dt * tp.wear_rate * p_slide * hot, 1.0)
