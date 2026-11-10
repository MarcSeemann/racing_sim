"""Driver inputs: live devices, recording and replay — roadmap phase 11.

* `HumanDriver` turns raw device input (a keyboard or gamepad, read by any
  front end) into car Inputs the way a real cockpit does: steering-wheel angle
  through a steering ratio, a deadzone, a steering rate limit (keyboards are
  on/off, so the wheel is eased towards its target), pedal smoothing, and
  paddle shifts with a rev check (no downshift that would over-rev the engine).
* `Recorder` wraps any driver and records its inputs every step.
* `Replay` plays recorded inputs back open loop, step by step — the same
  driver inputs on a different car or in different conditions show exactly how
  the change alters the car's behaviour.
"""
import csv
from dataclasses import dataclass

from .vehicle import Inputs


@dataclass
class DeviceState:
    """Raw controls as a front end reads them."""
    steer: float = 0.0  # -1 (full right) .. +1 (full left) of the wheel's travel
    throttle: float = 0.0  # 0..1
    brake: float = 0.0  # 0..1
    shift_up: bool = False  # paddle held this frame
    shift_down: bool = False
    drs: bool = False


class HumanDriver:
    """Live input adapter. `source(t)` returns a DeviceState for time t."""

    def __init__(self, source, wheel_lock_deg: float = 180.0, steering_ratio: float = 12.0,
                 deadzone: float = 0.03, steer_rate: float = 1.2, pedal_tau: float = 0.03):
        self.source = source
        self.max_road = (wheel_lock_deg / steering_ratio) * 3.141592653589793 / 180  # rad at the road wheel
        self.deadzone, self.steer_rate, self.pedal_tau = deadzone, steer_rate, pedal_tau
        self._steer = self._thr = self._brk = 0.0
        self._t = 0.0
        self._up_held = self._down_held = False
        self.gear = 1

    def __call__(self, s, car) -> Inputs:
        dev = self.source(s.t)
        dt = max(s.t - self._t, 0.0)
        self._t = s.t
        # Steering: deadzone, then rate-limited towards the target road-wheel angle.
        raw = 0.0 if abs(dev.steer) < self.deadzone else dev.steer
        target = max(-1.0, min(1.0, raw)) * self.max_road
        step = self.steer_rate * dt
        self._steer += max(-step, min(step, target - self._steer))
        # Pedals: first-order smoothing (a pedal cannot jump in zero time).
        k = min(dt / self.pedal_tau, 1.0) if self.pedal_tau > 0 else 1.0
        self._thr += (max(0.0, min(1.0, dev.throttle)) - self._thr) * k
        self._brk += (max(0.0, min(1.0, dev.brake)) - self._brk) * k
        # Paddles: one gear per press; refuse a downshift that would over-rev.
        gear = s.gear
        if dev.shift_up and not self._up_held and gear < len(car.p.gearbox.ratios):
            gear += 1
        if dev.shift_down and not self._down_held and gear > 1:
            if car.wheel_rpm_to_engine(s.v, gear - 1) < car.p.engine.rev_limit_rpm - 300:
                gear -= 1
        self._up_held, self._down_held = dev.shift_up, dev.shift_down
        self.gear = gear
        return Inputs(throttle=self._thr, brake=self._brk, gear=gear, steer=self._steer, drs=dev.drs)


class Recorder:
    """Wraps a driver and records what it does, step by step."""

    FIELDS = ("t", "throttle", "brake", "steer", "gear", "drs")

    def __init__(self, driver):
        self.driver = driver
        self.rows: list[tuple] = []
        # The line the recorded driver used: the track's rubber and drying line follow it.
        self.line_offsets = getattr(driver, "offsets", None)

    def __call__(self, s, car) -> Inputs:
        inp = self.driver(s, car)
        self.rows.append((s.t, inp.throttle, inp.brake, inp.steer,
                          inp.gear if inp.gear is not None else s.gear, int(bool(inp.drs))))
        return inp

    def __getattr__(self, name):  # lap counts etc. pass through to the wrapped driver
        return getattr(self.driver, name)

    def save(self, path: str):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.FIELDS)
            w.writerows(self.rows)


class Replay:
    """Plays recorded inputs back open loop, one recorded step per simulation step."""

    def __init__(self, rows, line_offsets=None):
        self.rows = list(rows)
        self.k = 0
        self.line_offsets = line_offsets

    @classmethod
    def of(cls, recorder: "Recorder"):
        """Replay a Recorder, including the racing line it was driven on."""
        return cls(recorder.rows, recorder.line_offsets)

    @classmethod
    def load(cls, path: str):
        with open(path) as fh:
            r = csv.reader(fh)
            next(r)
            return cls([(float(t), float(a), float(b), float(c), int(float(g)), int(float(d))) for t, a, b, c, g, d in r])

    @property
    def done(self) -> bool:
        return self.k >= len(self.rows)

    def __call__(self, s, car) -> Inputs:
        cond = getattr(car, "cond", None)
        if self.line_offsets is not None and cond is not None and cond.line_offset is not self.line_offsets:
            cond.set_racing_line(self.line_offsets)
        if self.done:  # out of recording: coast to a stop
            return Inputs(throttle=0.0, brake=0.5, gear=s.gear, steer=0.0)
        _, thr, brk, steer, gear, drs = self.rows[self.k]
        self.k += 1
        return Inputs(throttle=thr, brake=brk, gear=gear, steer=steer, drs=bool(drs))
