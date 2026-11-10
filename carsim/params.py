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
    heave_stiffness_front: float = 500e3  # N/m per axle
    heave_stiffness_rear: float = 300e3
    min_ride_height: float = 0.002  # skid blocks
    max_brake_force: float = 40000.0  # N at full pedal, before grip limit
    brake_front_bias: float = 0.58


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
class CarParams:
    engine: Engine = field(default_factory=Engine)
    gearbox: Gearbox = field(default_factory=Gearbox)
    aero: Aero = field(default_factory=Aero)
    chassis: Chassis = field(default_factory=Chassis)
    tyres: TyreParams = field(default_factory=TyreParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    fuel: Fuel = field(default_factory=Fuel)
    tyre_thermals: bool = True  # False = tyres always at optimum, never wear


def race_car(fuel_kg: float = 100.0, **kw) -> CarParams:
    """Race-weekend setup: 798 kg car with driver (minimum weight) plus fuel."""
    p = CarParams(**kw)
    p.chassis.mass = 798.0
    p.fuel.start_kg = fuel_kg
    return p
