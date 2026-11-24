"""Car and environment parameters.

Generic, plausible values for a modern single-seater (F1-style) race car; `f1_2025()` sizes
it to the 2025 F1 rules (mass, wheelbase, width, wheels).
All units SI: kg, m, s, N, W, rad. Engine speeds are given in rpm for readability.
"""
from dataclasses import dataclass, field

import numpy as np

from .thermal import ThermalParams
from .tyres import TyreParams

G = 9.81  # m/s^2
R_AIR = 287.05  # J/(kg K), specific gas constant for dry air


@dataclass
class Engine:
    # Full-throttle torque curve (rpm -> N m), combined ICE + hybrid deployment.
    rpm_points: tuple = (3000, 5000, 7000, 9000, 10500, 11500, 12500, 13500, 15000)
    torque_points: tuple = (300, 420, 520, 600, 620, 610, 560, 500, 380)
    idle_rpm: float = 4000.0
    rev_limit_rpm: float = 15000.0
    # Engine braking torque at closed throttle, scales with rpm fraction.
    engine_brake_torque: float = 60.0
    inertia: float = 0.04  # kg m^2, crankshaft + flywheel, seen at the crank
    # Air density effect on power: P ~ (rho / rho_ref)^k. A turbo compensates
    # much of it, so k is well below the 1.0 of a naturally aspirated engine.
    ref_density: float = 1.204
    density_exponent: float = 0.3


@dataclass
class Gearbox:
    # Gearbox ratios (gear 1..8). Overall ratio = gear ratio * final_drive.
    ratios: tuple = (4.25, 3.547, 2.96, 2.47, 2.062, 1.72, 1.437, 1.20)
    final_drive: float = 4.0
    efficiency: float = 0.92  # crank-to-wheel
    shift_time: float = 0.03  # s of torque interruption per shift


@dataclass
class Aero:
    CdA: float = 1.10  # drag coefficient * frontal area, m^2, at the reference ride height
    ClA: float = 4.00  # downforce coefficient * area, m^2, at the reference ride height
    # Slightly rearward of the 45% weight split: keeps the car stable in fast corners.
    front_balance: float = 0.42  # share of downforce on the front axle, at the reference ride height
    # Ride-height sensitivity (phase 6). All heights are the front ride height, m.
    ride_height_ref: float = 0.020  # where ClA and balance take the values above
    ride_height_sensitivity: float = 0.25  # +25% ClA for the full drop from h_ref to 0 (before stall)
    stall_height: float = 0.008  # below this the floor stalls
    stall_loss: float = 0.30  # share of downforce lost from stall height down to 0
    balance_per_m: float = 2.0  # front-balance shift per m of front drop (2% per 10 mm)
    induced_drag: float = 0.08  # extra CdA per unit of extra ClA
    drs_rear_loss: float = 0.32  # share of rear downforce lost with DRS open
    drs_drag_reduction: float = 0.10  # CdA removed with DRS open
    # 2026 active aero: with `active_aero`, the drs flag means "straight mode": BOTH wings flatten,
    # ClA and CdA scale by these factors (the balance stays). Estimates: ~55 % less drag was the
    # pre-season prediction; calibrate against 2026 lap times (tools/calibrate.py --car 2026).
    active_aero: bool = False
    straight_cla_factor: float = 0.55
    straight_cda_factor: float = 0.55
    side_force_area: float = 1.5  # Cy * side area, m^2 (crosswind force)
    # Phase 14: rake (rear ride height) and yaw sensitivity, drag pitching moment.
    rear_height_ref: float = 0.040  # m, rear ride height (at speed) at which ClA/balance take their nominal values
    rake_sensitivity: float = 0.6  # +ClA share per m of extra rear height (+3% per +50 mm), until it stalls the diffuser
    rake_balance_per_m: float = 0.5  # front-balance gain per m of extra rear height
    yaw_loss: float = 16.0  # ClA share lost per rad^2 of sideslip (~12% at 5 deg)
    yaw_loss_max: float = 0.40
    drag_height: float = 0.40  # m, height at which drag acts (pitches the car nose-up on the straight)


@dataclass
class Chassis:
    mass: float = 850.0  # kg, car + driver + some fuel
    wheelbase: float = 3.60
    cog_height: float = 0.30
    front_weight: float = 0.45  # static share of weight on the front axle
    wheel_radius: float = 0.36
    crr: float = 0.012  # rolling resistance coefficient
    wheel_inertia_front: float = 2.0  # kg m^2 per axle (two wheels, tyres, brake discs)
    wheel_inertia_rear: float = 2.8  # kg m^2 per axle, incl. driveshafts
    # Torque cut above the target rear slip. On by default: it stands in for a driver
    # modulating the throttle until the driver model (phase 11) does that itself.
    traction_control: bool = True
    tc_target_slip: float = 0.10
    track_front: float = 1.60  # m
    track_rear: float = 1.55
    yaw_inertia: float = 1000.0  # kg m^2
    # Share of lateral load transfer taken by the front axle (springs + anti-roll
    # bars). More front share = more understeer.
    roll_stiffness_front: float = 0.55
    load_transfer_lag: float = 0.02  # s, stands in for suspension dynamics
    diff_locking: float = 60.0  # N m per rad/s of left-right rear wheel speed difference
    max_steer: float = 0.35  # rad at the road wheel
    # Suspension heave (phase 6): parked ride heights and axle heave stiffness.
    ride_height_front: float = 0.035  # m
    ride_height_rear: float = 0.080
    heave_stiffness_front: float = 500e3  # N/m per axle, body to the ground (springs and tyres in series)
    heave_stiffness_rear: float = 300e3
    min_ride_height: float = 0.002  # skid blocks
    max_brake_force: float = 40000.0  # N at full pedal, before grip limit
    brake_front_bias: float = 0.58
    # Physical limits on how fast the controls can move (applied inside the car, so they bind
    # every driver: human, scripted or learned). A driver's hands and the steering rack cannot
    # flick from lock to lock in an instant, and it gets harder as the tyres load up with speed.
    # Rates are at the road wheel: 1.6 rad/s ~ 92 deg/s ~ 500 deg/s at a 5.5:1 single-seater wheel.
    actuator_limits: bool = True
    max_steer_rate: float = 1.6  # rad/s at the road wheel, low speed
    max_steer_rate_fast: float = 0.9  # rad/s at the road wheel at steer_rate_speed and above (heavy steering)
    steer_rate_speed: float = 80.0  # m/s
    throttle_rate_up: float = 8.0  # pedal travel per s (0 -> full in 0.125 s)
    throttle_rate_down: float = 20.0  # lifting is quicker (0.05 s)
    brake_rate_up: float = 10.0  # 0 -> full brake pressure in 0.1 s (foot + hydraulics)
    brake_rate_down: float = 15.0
    # Tyre relaxation length (phase 13): a tyre needs to roll this far for its lateral force
    # to build up after the slip angle changes. 0 = instant (the old behaviour).
    relaxation_length: float = 0.35  # m
    overrev_protection: bool = True  # the gearbox refuses a downshift that would over-rev the engine
    # Differential (phase 14): clutch-pack limited slip. Locking torque = preload + ramp x input torque
    # (power ramp on throttle, coast ramp under engine braking); diff_locking adds a viscous part.
    diff_preload: float = 80.0  # N m
    diff_power_lock: float = 0.35  # share of the drive torque that can be locked across on power
    diff_coast_lock: float = 0.20  # same, off throttle (engine braking)
    # Suspension (phase 14): "dynamic" = sprung body with heave, pitch and roll on springs, dampers
    # and anti-roll bars, four unsprung wheels on tyre springs, roll centres, anti-dive/squat and a
    # plank that touches the ground. "quasi_static" = the old lagged load-transfer model.
    suspension: str = "dynamic"
    unsprung_front: float = 18.0  # kg per corner (wheel, tyre, upright, brakes, half the links)
    unsprung_rear: float = 20.0
    roll_inertia: float = 120.0  # kg m^2, sprung body about its roll axis
    pitch_inertia: float = 850.0  # kg m^2, sprung body about its pitch axis
    roll_gradient_deg_per_g: float = 0.35  # body roll per g of lateral acceleration (springs + bars)
    roll_centre_front: float = 0.04  # m above the ground
    roll_centre_rear: float = 0.07
    anti_dive: float = 0.30  # share of the braking pitch moment taken by the links (not the springs)
    anti_squat: float = 0.25
    damping_bump: float = 0.45  # damping ratio of the body on each corner's spring, compression
    damping_rebound: float = 0.65  # extension
    bump_travel: float = 0.030  # m of wheel travel before the bump stop
    bump_stop_rate: float = 2.0e6  # N/m of the (packer) bump stop
    tyre_vertical_rate: float = 350e3  # N/m per tyre (low-profile 18-inch race tyre)
    tyre_vertical_damping: float = 400.0  # N s/m per tyre
    plank_rate: float = 3.0e6  # N/m of the skid plank on the track when the floor bottoms out
    plank_friction: float = 0.35  # friction of the plank sliding on the track
    # Camber (phase 14): static camber and how much of the body roll the wheels follow.
    camber_front_deg: float = -3.0
    camber_rear_deg: float = -1.5
    camber_roll_follow: float = 0.45  # share of body roll that tilts the wheel (1 = rigid to the body)
    camber_opt_deg: float = -3.5  # camber (leaning into the corner) of peak lateral grip
    camber_lat_loss: float = 0.0010  # lateral grip lost per deg^2 away from the optimum
    camber_long_loss: float = 0.0020  # braking/traction grip lost per deg^2 of camber
    ackermann: float = 0.0  # 0 = parallel steer, 1 = full Ackermann (inner wheel steers more)


@dataclass
class Brakes:
    """Carbon-carbon brake temperature (phase 13). Carbon has little bite when cold, a wide
    window when hot, and fades (and wears) once it overheats."""
    thermal: bool = True
    heat_capacity: float = 1500.0  # J/K per corner, disc + pads
    heat_share: float = 0.9  # share of the braking power that ends up in the disc and pads
    h_base: float = 5.0  # W/K to the air at standstill
    h_per_ms: float = 1.0  # W/K per m/s through the brake ducts (time constant ~20 s at 250 km/h)
    start_temp: float = 450.0  # deg C at the start (already warm; set lower for an out-lap)
    t_cold: float = 350.0  # below this the carbon loses bite
    cold_floor_temp: float = 100.0  # temperature of the cold_mu value
    cold_mu: float = 0.65  # friction factor of a stone-cold brake
    t_hot: float = 1000.0  # above this the brake fades
    fade_per_100k: float = 0.10  # friction lost per 100 K over t_hot
    min_mu: float = 0.5


def brake_friction(b: Brakes, temp_c: float) -> float:
    """Friction factor (1 = full bite) of a carbon brake at temp_c."""
    if temp_c < b.t_cold:
        x = min(max((temp_c - b.cold_floor_temp) / (b.t_cold - b.cold_floor_temp), 0.0), 1.0)
        return b.cold_mu + (1 - b.cold_mu) * x
    if temp_c > b.t_hot:
        return max(1 - b.fade_per_100k * (temp_c - b.t_hot) / 100.0, b.min_mu)
    return 1.0


def pressure_at_altitude(altitude_m: float) -> float:
    """Standard-atmosphere air pressure (Pa) at a height above sea level."""
    return 101325.0 * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588


@dataclass
class Environment:
    air_temp_c: float = 20.0
    pressure_pa: float = 101325.0
    track_temp_c: float = 30.0
    headwind: float = 0.0  # m/s, positive opposes the car's motion
    grade: float = 0.0  # road slope, rad (positive = uphill)

    @property
    def air_density(self) -> float:
        return self.pressure_pa / (R_AIR * (self.air_temp_c + 273.15))


@dataclass
class Fuel:
    """Fuel load (roadmap phase 10). start_kg = 0 keeps the car at its chassis mass (earlier phases)."""
    start_kg: float = 0.0
    capacity_kg: float = 110.0
    lhv: float = 43.0e6  # J/kg, lower heating value of race fuel
    efficiency: float = 0.60  # crank power per unit fuel power; >0.5 because ~120 kW is hybrid (about 100 kg/h at full power)
    idle_flow: float = 0.0003  # kg/s with the throttle closed
    front_share: float = 0.47  # where the tank sits, as a front-axle weight share
    cog_height: float = 0.25  # m, tank centre of mass
    yaw_radius: float = 0.35  # m, radius of gyration of the fuel about the car's centre


@dataclass
class ERS:
    """Hybrid energy recovery (phase 14): the MGU-K adds up to `power` at the crank while the battery
    has charge, and harvests up to `harvest_power` from the rear axle under braking (brake-by-wire
    takes that share off the rear friction brakes). The engine torque curve includes the full
    deployment, so with charge in the battery the car is exactly as fast as before; flat, it loses
    `power` at full throttle."""
    enabled: bool = True
    power: float = 120e3  # W at full throttle
    harvest_power: float = 120e3  # W
    store: float = 4.0e6  # J usable in the battery
    start_soc: float = 1.0  # share of `store` at the start
    deploy_efficiency: float = 0.95  # battery to crank
    harvest_efficiency: float = 0.85  # wheel to battery
    min_harvest_speed: float = 8.0  # m/s
    exhaust_harvest: float = 40e3  # W recovered from the exhaust (MGU-H) at full throttle, high revs
    max_torque_share: float = 0.5  # the MGU-K never gives more than this share of the full torque curve
    # 2026: deployment tapers with speed: full up to the first value (km/h), nothing at the second.
    deploy_taper_kmh: tuple | None = None
    # Deployment strategy of the ECU: full power above soc_taper, tapering to nothing at soc_floor
    # ("clipping" at the end of the straights instead of running the battery flat).
    soc_floor: float = 0.08
    soc_taper: float = 0.35


@dataclass
class CarParams:
    engine: Engine = field(default_factory=Engine)
    gearbox: Gearbox = field(default_factory=Gearbox)
    aero: Aero = field(default_factory=Aero)
    chassis: Chassis = field(default_factory=Chassis)
    tyres: TyreParams = field(default_factory=TyreParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    fuel: Fuel = field(default_factory=Fuel)
    brakes: Brakes = field(default_factory=Brakes)
    ers: ERS = field(default_factory=ERS)
    tyre_thermals: bool = True  # False = tyres always at optimum, never wear


# 2025 F1 technical regulations (the last year of the 2022-2025 rules). Hard limits from the
# rules; the rest are typical values for a car built to them.
F1_2025 = {
    "min_mass_kg": 800.0,  # car + driver (driver and seat at least 80 kg), no fuel; 798 kg in 2023-24
    "max_wheelbase_m": 3.60,
    "max_width_m": 2.00,
    "wheel_rim_in": 18,
    "tyre_diameter_m": (0.720, 0.725),  # front, rear (Pirelli 18-inch)
    "tyre_width_m": (0.305, 0.405),  # front, rear tread width, roughly
    "max_fuel_kg": 110.0,
}

# Downforce trims: (ClA, CdA, front balance) of a 2022-2025 car: "low" is a Monza wing, "high" a
# Monaco/Hungary wing. Calibrated (docs/calibration/2025.md) against the 2025 pole laps with the
# lap-time model: x1.25 downforce and x1.45 drag on the first estimates (which were 21 % too slow);
# Monza top speed ~356 km/h with DRS, ~5-6 g at 300 km/h, downforce = weight at ~185 km/h.
TRIMS = {"low": (4.00, 1.23, 0.41), "medium": (5.00, 1.60, 0.42), "high": (5.88, 1.89, 0.43)}
# Tyres of the calibrated car: more grip at low load, falling off faster with load (real slicks are
# strongly load-sensitive); braking grip rises less than cornering grip (peak braking ~5-6 g).
TYRE_2025 = {"lateral_mu0": 2.38, "longitudinal_mu0": 2.10, "load_sensitivity": -0.25}


def f1_2025(fuel_kg: float = 0.0, downforce: str = "medium", **kw) -> CarParams:
    """A car built to the 2025 F1 rules: 800 kg with driver, 3.60 m wheelbase, 2.0 m wide on
    18-inch wheels (track = overall width less a tyre width, at each axle), plus fuel.
    `downforce` picks the wing trim: "low" (Monza), "medium", "high"."""
    p = CarParams(**kw)
    c, r = p.chassis, F1_2025
    c.mass = r["min_mass_kg"]
    c.wheelbase = r["max_wheelbase_m"]
    c.wheel_radius = sum(r["tyre_diameter_m"]) / 4
    c.track_front = r["max_width_m"] - r["tyre_width_m"][0]  # 1.695 m
    c.track_rear = r["max_width_m"] - r["tyre_width_m"][1]  # 1.595 m
    p.fuel.capacity_kg = r["max_fuel_kg"]
    p.fuel.start_kg = fuel_kg
    p.aero.ClA, p.aero.CdA, p.aero.front_balance = TRIMS[downforce]
    p.tyres.lateral.mu0 = TYRE_2025["lateral_mu0"]
    p.tyres.longitudinal.mu0 = TYRE_2025["longitudinal_mu0"]
    p.tyres.load_sensitivity = TYRE_2025["load_sensitivity"]
    return p


def race_car(fuel_kg: float = 100.0, spec: str = "2025", **kw) -> CarParams:
    """Race-weekend setup: the 2025-spec car (800 kg with driver, minimum weight) plus fuel, or the
    2026-spec car (spec="2026")."""
    return f1_2026(fuel_kg, **kw) if str(spec) == "2026" else f1_2025(fuel_kg, **kw)


# The 2026 rules (FIA 2026 Technical Regulations, as reported by F1 and the teams; see
# docs/car_2026.md for sources). Hard numbers from the rules; the power-unit split and the aero are
# estimates to calibrate against 2026 lap times.
F1_2026 = {
    "min_mass_kg": 768.0,  # car + driver, no fuel (2025: 800)
    "max_wheelbase_m": 3.40,  # 2025: 3.60
    "max_width_m": 1.90,  # 2025: 2.00
    "tyre_diameter_m": (0.720, 0.725),  # 18-inch, as before
    "tyre_width_m": (0.280, 0.375),  # front 25 mm, rear 30 mm narrower than 2025
    "max_fuel_kg": 110.0,
    "ice_kw": 400.0,  # estimate: about half the power is electric
    "mguk_kw": 350.0,  # 2025: 120 kW (and an MGU-H, now gone)
    "deploy_taper_kmh": (290.0, 355.0),  # full MGU-K power up to 290 km/h, none at 355
}
# Corner-mode trims (ClA, CdA, front balance), calibrated (docs/calibration/2025.md) against the 2026
# pole laps: ~26 % less downforce than the calibrated 2025 car (the FIA aimed for ~30 %), drag a
# little lower; straight mode scales them in the aero zones.
TRIMS_2026 = {"low": (3.09, 1.13, 0.41), "medium": (3.72, 1.38, 0.42), "high": (4.36, 1.60, 0.43)}
TYRE_GRIP_2026 = 0.94  # narrower tyres (25 mm front, 30 mm rear): grip relative to the 2025 car


def f1_2026(fuel_kg: float = 0.0, downforce: str = "medium", **kw) -> CarParams:
    """A car built to the 2026 F1 rules: 768 kg with driver, 3.40 m wheelbase, 1.90 m wide on
    narrower 18-inch tyres; ~400 kW engine + 350 kW MGU-K (tapering from 290 to 355 km/h), no MGU-H;
    active aero (straight mode in the aero zones). Plus fuel."""
    from .powertrain import RPM_TO_RADS
    p = f1_2025(fuel_kg, downforce=downforce, **kw)
    c, r = p.chassis, F1_2026
    c.mass = r["min_mass_kg"]
    c.wheelbase = r["max_wheelbase_m"]
    c.wheel_radius = sum(r["tyre_diameter_m"]) / 4
    c.track_front = r["max_width_m"] - r["tyre_width_m"][0]
    c.track_rear = r["max_width_m"] - r["tyre_width_m"][1]
    p.fuel.capacity_kg = r["max_fuel_kg"]
    p.aero.ClA, p.aero.CdA, p.aero.front_balance = TRIMS_2026[downforce]
    p.tyres.lateral.mu0 *= TYRE_GRIP_2026
    p.tyres.longitudinal.mu0 *= TYRE_GRIP_2026
    p.aero.active_aero = True
    # Engine: the 2025 curve's shape scaled to a ~400 kW ICE, plus up to 350 kW of MGU-K.
    e, rpm = p.engine, np.asarray(p.engine.rpm_points, float)
    t25 = np.asarray(e.torque_points, float)
    omega = rpm * RPM_TO_RADS
    ers25 = np.minimum(120e3 / omega, 0.5 * t25)
    ice = t25 - ers25
    ice *= r["ice_kw"] * 1e3 / float(np.max(ice * omega))
    mguk = np.minimum(r["mguk_kw"] * 1e3 / omega, 1.5 * ice)  # never more than 60 % of the total
    e.torque_points = tuple(float(x) for x in ice + mguk)
    p.ers.power = p.ers.harvest_power = r["mguk_kw"] * 1e3
    p.ers.exhaust_harvest = 0.0  # no MGU-H
    p.ers.max_torque_share = 0.6
    p.ers.deploy_taper_kmh = r["deploy_taper_kmh"]
    return p
