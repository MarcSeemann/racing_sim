"""Real-time simulation server for the 3D front end (Godot).

The physics runs here, at 1 kHz paced to the wall clock; a front end connects
over TCP on localhost and exchanges newline-delimited JSON:

    server -> client, once:   {"type": "track", ...}     centreline, width, elevation, racing line
    server -> client, ~60 Hz: {"type": "state", ...}     car pose, wheels, HUD values
    client -> server, any rate: {"type": "input", "steer": -1..1, "throttle": 0..1, "brake": 0..1,
                                 "shift_up": bool, "shift_down": bool, "drs": bool}
                                {"type": "reset"}         put the car back on track, stopped
                                {"type": "quit"}

    python -m carsim.server --track test --weather shower --compound intermediate
    python -m carsim.server --ai            # the AI drives; the client just watches

Coordinates are the simulation's: x east, y north, z up (metres); yaw in radians,
anticlockwise from +x. The client converts to its own axes.
"""
import argparse
import json
import math
import selectors
import socket
import time

from .compare import LiveDelta
from .conditions import Conditions, Weather
from .drivers import LapDriver
from .dynamics import Car2D
from .inputs import DeviceState, HumanDriver
from .params import race_car
from .track import hill_circuit, short_circuit
from .track import test_circuit as long_circuit

DT = 0.001
SEND_HZ = 60.0
TRACKS = {"short": short_circuit, "test": long_circuit, "hill": hill_circuit}
WEATHER = {"dry": lambda: Weather.dry(),
           "shower": lambda: Weather.shower(start=60, peak=180, end=360, rain_mm_h=8.0),
           "wet": lambda: Weather.constant(rain_mm_h=6.0, cloud=1.0, air_temp_c=16.0)}


class Session:
    """One car on one track, driven by a player (inputs from the client) or the AI."""

    def __init__(self, track="test", weather="dry", compound="slick", fuel=40.0, ai=False, learned=False):
        self.track_name = track
        self.learned = learned  # drive with the self-taught network (carsim.learn) instead of LapDriver
        self.track = TRACKS[track]()
        self.weather_name, self.compound, self.fuel, self.ai = weather, compound, fuel, ai
        self.ai_driver = LapDriver(self.track, line="racing")  # also gives the racing line to draw
        self.device = DeviceState()
        self.reset(full=True)

    def reset(self, full=False):
        if full:
            self.cond = Conditions(self.track, WEATHER[self.weather_name](), cars_on_track=20, rubber=2e6)
            self.car = Car2D(race_car(self.fuel), conditions=self.cond, compound=self.compound)
            self.car.set_speed(20.0 if self.ai else 0.0)  # you start from a standstill
            self.ai_driver = LapDriver(self.track, line=(self.ai_driver.path, self.ai_driver.offsets))
            if self.learned:
                from .learn import LearnedDriver
                self.driver = LearnedDriver(self.track, self.track_name)
            else:
                self.driver = self.ai_driver if self.ai else HumanDriver(lambda t: self.device)
            self.live = LiveDelta(self.track)
            self.lap, self.lap_start, self.last_lap, self.best_lap = 0, 0.0, None, None
            self._last_i = 0
            self.wheel_angle = [0.0] * 4
            self.row = None
            return
        # Back on track at the nearest centreline point, pointing along it, at rest.
        s, tr = self.car.state, self.track
        i = tr.nearest(s.x, s.y, self.cond.i, window=len(tr.s))
        s.x, s.y, s.psi = float(tr.x[i]), float(tr.y[i]), float(tr.heading[i])
        s.vx = s.vy = s.r = s.ax = s.ay = 0.0
        s.w = [0.0] * 4
        s.gear = 1
        self.cond.i = i

    def step(self, n: int):
        s = self.car.state
        for _ in range(n):
            inp = self.driver(s, self.car)
            self.row = self.car.step(inp, DT)
            for k in range(4):
                self.wheel_angle[k] = (self.wheel_angle[k] + s.w[k] * DT) % (2 * math.pi)
            i = self.cond.i
            if i < self._last_i - len(self.track.s) // 2:  # crossed the line
                if self.lap > 0 or s.t > 5.0:
                    lap_time = s.t - self.lap_start
                    self.last_lap = lap_time
                    self.best_lap = lap_time if self.best_lap is None else min(self.best_lap, lap_time)
                self.lap += 1
                self.lap_start = s.t
                if self.learned and self.driver.maybe_reload():  # training saved a better driver
                    print("loaded the newest trained driver", flush=True)
            self._last_i = i
        self.gap = self.live.update(self.cond.i, s.t)

    # ---- messages ----------------------------------------------------------------
    def track_message(self) -> dict:
        tr, line = self.track, self.ai_driver.path
        r = lambda a: [round(float(v), 2) for v in a]
        return {"type": "track", "name": tr.name, "width": tr.width, "kerb": tr.kerb_width,
                "x": r(tr.x), "y": r(tr.y), "z": r(tr.z), "heading": [round(float(v), 4) for v in tr.heading],
                "line_x": r(line.x), "line_y": r(line.y),
                "wheelbase": self.car.p.chassis.wheelbase, "track_front": self.car.p.chassis.track_front,
                "track_rear": self.car.p.chassis.track_rear, "wheel_radius": self.car.p.chassis.wheel_radius,
                "a": self.car.a, "b": self.car.b, "compound": self.compound, "weather": self.weather_name,
                "ai": self.ai or self.learned, "learned": self.learned}

    def state_message(self) -> dict:
        s, row, c, tr = self.car.state, self.row or {}, self.car.p.chassis, self.track
        i = self.cond.i
        hf, hr = s.hf, s.hr
        # The car (wheels) follows the track's slope; only the body pitches, rolls and squats on
        # its suspension. Roll: ~0.35 deg per g, about what a stiff single-seater shows.
        grade = float(tr.grade[i])
        body_pitch = math.atan2((hf - hr) - (c.ride_height_front - c.ride_height_rear), c.wheelbase)
        roll = -0.0006 * s.ay
        body_dz = (hf + hr) / 2 - (c.ride_height_front + c.ride_height_rear) / 2
        z = float(tr.z[i])
        g = lambda k, d=0.0: float(row.get(k, d))
        return {"type": "state", "t": round(s.t, 3), "x": round(s.x, 3), "y": round(s.y, 3), "z": round(z, 3),
                "yaw": round(s.psi, 4), "pitch": round(grade, 4), "body_pitch": round(body_pitch, 4),
                "roll": round(roll, 4), "body_dz": round(body_dz, 4),
                "v": round(s.speed * 3.6, 1), "gear": s.gear, "rpm": round(s.rpm), "steer": round(s.steer, 4),
                "throttle": round(g("throttle"), 3), "brake": round(g("brake"), 3),
                "wheels": [round(a, 3) for a in self.wheel_angle],
                "lap": self.lap + 1, "lap_time": round(s.t - self.lap_start, 2),
                "last": self.last_lap and round(self.last_lap, 3), "best": self.best_lap and round(self.best_lap, 3),
                "gap": None if self.gap is None else round(self.gap, 2),
                "tyres": [round(g(f"tsurf_{n}"), 0) for n in ("fl", "fr", "rl", "rr")],
                "wear": [round(g(f"wear_{n}"), 3) for n in ("fl", "fr", "rl", "rr")],
                "fuel": round(self.car.fuel, 2), "rain": round(g("rain"), 2), "water": round(g("water"), 3),
                "track_temp": round(g("track_temp"), 1), "grass": int(g("grass_wheels")),
                "drs": bool(row.get("drs", False)), "s": round(float(tr.s[i]), 1)}


def serve(session: Session, host="127.0.0.1", port=4545, once=False, max_seconds=None):
    """Accept one client at a time and run the session in real time while it is connected."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"carsim server on {host}:{port} — track {session.track.name}, {session.weather_name}, "
          f"{session.compound}{', AI driving' if session.ai else ''}. Waiting for the 3D client...", flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            print(f"client connected from {addr[0]}", flush=True)
            _run_client(session, conn, max_seconds)
            print("client disconnected", flush=True)
            if once:
                return
    finally:
        srv.close()


def _send(conn, msg):
    conn.sendall((json.dumps(msg, separators=(",", ":")) + "\n").encode())


def _run_client(session, conn, max_seconds=None):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setblocking(False)
    sel = selectors.DefaultSelector()
    sel.register(conn, selectors.EVENT_READ)
    conn.setblocking(True)
    _send(conn, session.track_message())
    conn.setblocking(False)
    buf = b""
    start = last = last_send = time.perf_counter()
    sim_behind = 0.0
    try:
        while True:
            # Inputs.
            for _key, _ in sel.select(timeout=0):
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "input":
                    d = session.device
                    d.steer = float(msg.get("steer", 0.0))
                    d.throttle = float(msg.get("throttle", 0.0))
                    d.brake = float(msg.get("brake", 0.0))
                    d.shift_up = bool(msg.get("shift_up", False))
                    d.shift_down = bool(msg.get("shift_down", False))
                    d.drs = bool(msg.get("drs", False))
                elif kind == "reset":
                    session.reset()
                elif kind == "restart":
                    session.reset(full=True)
                elif kind == "quit":
                    return
            # Physics, paced to the wall clock (catch up at most 50 ms per loop).
            now = time.perf_counter()
            sim_behind = min(sim_behind + (now - last), 0.05)
            last = now
            n = int(sim_behind / DT)
            if n:
                session.step(n)
                sim_behind -= n * DT
            if now - last_send >= 1.0 / SEND_HZ and session.row is not None:
                _send(conn, session.state_message())
                last_send = now
            if max_seconds is not None and now - start > max_seconds:
                return
            time.sleep(0.0005)
    except (BrokenPipeError, ConnectionResetError):
        return
    finally:
        sel.close()
        conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=TRACKS, default="test")
    ap.add_argument("--weather", choices=WEATHER, default="dry")
    ap.add_argument("--compound", choices=("slick", "intermediate", "wet"), default="slick")
    ap.add_argument("--fuel", type=float, default=40.0)
    ap.add_argument("--ai", action="store_true", help="the AI drives (watch mode)")
    ap.add_argument("--learned", action="store_true",
                    help="the self-taught driver drives (python -m carsim.learn train ...); reloads as it improves")
    ap.add_argument("--port", type=int, default=4545)
    a = ap.parse_args(argv)
    serve(Session(a.track, a.weather, a.compound, a.fuel, a.ai, a.learned), port=a.port)


if __name__ == "__main__":
    main()
