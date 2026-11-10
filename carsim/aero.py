"""Aerodynamics with ride height — roadmap phase 6.

Downforce depends on the front ride height (the floor and front wing work in
ground effect): lower = more downforce, until the floor stalls below a critical
height and downforce falls away. Aero balance moves forward as the nose drops.
Drag grows with downforce (induced drag). DRS opens the rear wing: less rear
downforce, less drag.

Ride heights come from the suspension: each axle sinks under its load beyond the
static (parked) load, against its heave spring, down to the skid blocks.
"""
from dataclasses import dataclass


@dataclass
class AeroResult:
    downforce: float  # N, total
    front_share: float  # share of downforce on the front axle
    drag: float  # N
    ClA: float
    CdA: float


def aero_forces(aero, rho: float, v_air: float, v: float, h_front: float, drs: bool = False) -> AeroResult:
    """Aero forces for air speed v_air (drag) and road speed v (downforce), front ride height h_front (m)."""
    # Ground effect on the front ride height.
    h_ref, h_stall = aero.ride_height_ref, aero.stall_height
    if h_front >= h_stall:
        factor = 1 + aero.ride_height_sensitivity * (h_ref - h_front) / h_ref
    else:
        at_stall = 1 + aero.ride_height_sensitivity * (h_ref - h_stall) / h_ref
        factor = at_stall * (1 - aero.stall_loss * (h_stall - h_front) / h_stall)
    cla = aero.ClA * max(factor, 0.3)
    front = aero.front_balance + aero.balance_per_m * (h_ref - h_front)

    cda = aero.CdA + aero.induced_drag * (cla - aero.ClA)
    if drs:
        front_cla = cla * front
        rear_cla = cla * (1 - front) * (1 - aero.drs_rear_loss)
        cla = front_cla + rear_cla
        front = front_cla / cla
        cda -= aero.drs_drag_reduction

    q = 0.5 * rho * v * v
    return AeroResult(downforce=q * cla, front_share=front,
                      drag=0.5 * rho * cda * v_air * abs(v_air), ClA=cla, CdA=cda)


def ride_heights(chassis, n_front_axle: float, n_rear_axle: float):
    """(front, rear) ride heights in m from axle loads, clamped at the skid blocks."""
    from .params import G  # local import avoids a cycle
    static_f = chassis.mass * G * chassis.front_weight
    static_r = chassis.mass * G * (1 - chassis.front_weight)
    hf = chassis.ride_height_front - (n_front_axle - static_f) / chassis.heave_stiffness_front
    hr = chassis.ride_height_rear - (n_rear_axle - static_r) / chassis.heave_stiffness_rear
    return max(hf, chassis.min_ride_height), max(hr, chassis.min_ride_height)
