"""Car and environment parameters.

Generic, plausible values for a modern single-seater (F1-style) race car.
All units SI: kg, m, s, N, W, rad. Engine speeds are given in rpm for readability.
"""
from dataclasses import dataclass, field

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


def race_car(fuel_kg: float = 100.0, **kw) -> CarParams:
    """Race-weekend setup: 798 kg car with driver (minimum weight) plus fuel."""
    p = CarParams(**kw)
    p.chassis.mass = 798.0
    p.fuel.start_kg = fuel_kg
    return p
