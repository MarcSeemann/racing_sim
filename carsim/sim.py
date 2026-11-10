"""Fixed-timestep simulation loop."""
from .telemetry import Telemetry
from .vehicle import LongitudinalCar

DT = 0.001  # 1 kHz physics


def run(car: LongitudinalCar, driver, t_end: float, dt: float = DT, stop=None, log_every: int = 1) -> Telemetry:
    """Step the car until t_end or until stop(state) returns True. Logs every `log_every` steps."""
    log = Telemetry()
    n = int(round(t_end / dt))
    for k in range(n):
        inp = driver(car.state, car)
        row = car.step(inp, dt)
        if k % log_every == 0:
            log.log(row)
        if stop is not None and stop(car.state):
            break
    return log


def run_laps(car, driver, laps: int, dt: float = DT, log_every: int = 20, t_max: float = 3600.0):
    """Drive `laps` laps with a LapDriver from a rolling start. Returns (telemetry, lap times)."""
    stop = lambda s: driver.lap >= laps or driver.off_track
    log = run(car, driver, t_end=t_max, dt=dt, stop=stop, log_every=log_every)
    return log, list(driver.lap_times)
