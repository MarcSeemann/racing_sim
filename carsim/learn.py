"""An AI driver that teaches itself to drive a track — evolution strategies (numpy only).

The driver is a small neural network (an MLP). It sees what a driver feels and sees:
speed, sideslip, yaw rate, where it is across the track, its angle to the track, how
the track curves ahead (several look-ahead distances), wheel slip; and outputs steering
and one pedal (positive = throttle, negative = brake). Gears are automatic.

It starts knowing nothing. Training (OpenAI-style evolution strategies): each
generation, try many randomly perturbed copies of the network, drive each one for an
episode, and move the network towards the perturbations that went further round the
track without leaving it. Rewards:

    distance along the track (m)  - time-outs/crashes end the episode early
    + a bonus for a completed lap, larger the faster the lap

Practice episodes are capped by DISTANCE, not time (a time cap rewards crashing fast over
surviving slowly): the run from the line covers the whole lap, runs from random starts cover
`start_distance` metres. Leaving the track costs a penalty, and reaching the cap earlier earns a
small bonus, so the order of priorities is: stay on track > cover the distance > be quick.

Train in the background (resumes from the last checkpoint):

    python -m carsim.learn train --track short --workers 8          # Ctrl+C any time
    nohup python -m carsim.learn train --track test --hours 3 > train.log 2>&1 &
    python -m carsim.learn status --track short                     # progress so far
    python -m carsim.learn drive --track short                      # one lap with the best driver
    python drive3d.py --track short --learned                       # watch it in 3D (reloads as it learns)
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .drivers import auto_gear
from .dynamics import Car2D
from .params import CarParams
from .track import hill_circuit, short_circuit
from .track import test_circuit as long_circuit
from .vehicle import Inputs

TRACKS = {"short": short_circuit, "test": long_circuit, "hill": hill_circuit}
LOOKAHEAD_M = (5, 15, 30, 50, 80, 120, 170, 230)
CONTROL_DT = 0.02  # the network decides 50 times a second; physics runs at 1 kHz
PHYS_DT = 0.001
RUNS_DIR = Path(os.environ.get("CARSIM_RUNS", "ai_runs"))


# ---- the policy network --------------------------------------------------------------
class Policy:
    """MLP: observation -> (steer, pedal), tanh outputs in -1..1."""

    def __init__(self, n_obs: int, hidden=(32, 32), n_act: int = 2, theta=None):
        self.shapes = []
        sizes = (n_obs, *hidden, n_act)
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.shapes += [(a, b), (b,)]
        self.size = sum(int(np.prod(s)) for s in self.shapes)
        self.theta = np.zeros(self.size) if theta is None else np.asarray(theta, float)

    def init_random(self, rng):
        parts = []
        for s in self.shapes:
            parts.append(rng.standard_normal(s).ravel() / math.sqrt(s[0]) if len(s) == 2 else np.zeros(s))
        self.theta = np.concatenate(parts)
        return self

    def act(self, obs, theta=None):
        th = self.theta if theta is None else theta
        x, k = obs, 0
        for i in range(0, len(self.shapes), 2):
            (a, b), _ = self.shapes[i], self.shapes[i + 1]
            W = th[k:k + a * b].reshape(a, b)
            k += a * b
            bias = th[k:k + b]
            k += b
            x = np.tanh(x @ W + bias)
        return x


# ---- observations and one episode --------------------------------------------------------
def observe(car, track, i: int) -> np.ndarray:
    s = car.state
    h = track.heading[i]
    err = track.cross_track_error(i, s.x, s.y) / (track.width / 2)
    ang = math.remainder(s.psi - h, 2 * math.pi)
    curv = [track.curvature[track.index_at(track.s[i] + d)] * 30.0 for d in LOOKAHEAD_M]
    a_front = (s.alpha[0] + s.alpha[1]) / 2
    a_rear = (s.alpha[2] + s.alpha[3]) / 2
    k_rear = (s.kappa[2] + s.kappa[3]) / 2
    obs = [s.vx / 60.0, s.vy / 5.0, s.r / 1.5, err, math.sin(ang) * 3, math.cos(ang) - 1,
           a_front * 5, a_rear * 5, max(min(k_rear * 5, 3), -3), s.steer / 0.35]
    return np.array(obs + curv)


N_OBS = 10 + len(LOOKAHEAD_M)


@dataclass
class EpisodeConfig:
    track: str = "short"
    max_time: float = 45.0  # s of simulated driving per episode
    laps: int = 1
    start_speed: float = 15.0
    start_s: float = 0.0  # m along the lap where the episode starts
    off_margin: float = 1.0  # m beyond the track edge that ends the episode
    alive_bonus: float = 0.0  # fitness per second on track (rewards slowing down rather than crashing)
    lap_bonus: bool = True  # practice runs from random starts do not count laps
    max_distance: float = 0.0  # m; the episode ends with reason "distance" when reached (0 = no cap)
    crash_penalty: float = 0.0  # fitness lost for leaving the track or getting stuck
    crash_speed_penalty: float = 0.0  # extra fitness lost per m/s of speed at impact (crashing slower is less bad)
    time_bonus: float = 0.0  # fitness per second saved reaching max_distance faster than ref_speed
    ref_speed: float = 15.0  # m/s; reaching the cap slower than this earns no time bonus


def _start_gear(car, v):
    """Highest gear that still keeps the engine above ~10 000 rpm at speed v."""
    best = 1
    for g in range(1, len(car.p.gearbox.ratios) + 1):
        if car.wheel_rpm_to_engine(v, g) > 10000:
            best = g
    return best


def run_episode(theta, cfg: EpisodeConfig, track=None, record=False):
    """Drive one episode with network weights theta. Returns (fitness, info[, telemetry])."""
    track = track or TRACKS[cfg.track]()
    policy = Policy(N_OBS, theta=theta)
    car = Car2D(CarParams(tyre_thermals=False))
    s = car.state
    i = track.index_at(cfg.start_s)
    s.x, s.y, s.psi = float(track.x[i]), float(track.y[i]), float(track.heading[i])
    car.set_speed(cfg.start_speed)
    s.gear = _start_gear(car, cfg.start_speed)
    progress, last_s = 0.0, float(track.s[i])
    n_sub = int(round(CONTROL_DT / PHYS_DT))
    lap_times, lap_start = [], 0.0
    reason = "time"
    rows = []
    slow = 0.0
    while s.t < cfg.max_time:
        i = track.nearest(s.x, s.y, i)
        act = policy.act(observe(car, track, i))
        steer = float(act[0]) * car.p.chassis.max_steer
        pedal = float(act[1])
        inp = Inputs(throttle=max(pedal, 0.0), brake=max(-pedal, 0.0), steer=steer, gear=auto_gear(s, car))
        for _ in range(n_sub):
            car.step(inp, PHYS_DT)
        i = track.nearest(s.x, s.y, i)
        s_now = track.s[i]
        ds = s_now - last_s
        if ds < -track.length / 2:  # crossed the line
            ds += track.length
            lap_times.append(s.t - lap_start)
            lap_start = s.t
        elif ds > track.length / 2:
            ds -= track.length
        progress += ds
        last_s = s_now
        if record:
            rows.append((s.t, s.x, s.y, s.speed * 3.6, steer, pedal))
        if abs(track.cross_track_error(i, s.x, s.y)) > track.width / 2 + cfg.off_margin:
            reason = "off track"
            break
        slow = slow + CONTROL_DT if s.speed < 3.0 else 0.0
        if slow > 3.0 or progress < -20:
            reason = "stuck"
            break
        if cfg.max_distance and progress >= cfg.max_distance:
            reason = "distance"
            break
        if len(lap_times) >= cfg.laps:
            reason = "lap"
            break
    fitness = progress + cfg.alive_bonus * s.t
    if reason == "off track":
        fitness -= cfg.crash_penalty + cfg.crash_speed_penalty * s.speed
    elif reason == "stuck":  # never cheaper than a fast crash, or standing still becomes attractive
        fitness -= cfg.crash_penalty + cfg.crash_speed_penalty * 60.0
    elif reason == "distance" and cfg.time_bonus:
        fitness += cfg.time_bonus * max(0.0, cfg.max_distance / cfg.ref_speed - s.t)
    if lap_times and cfg.lap_bonus:  # finishing matters most; faster laps score more
        fitness += 2000.0 + 50.0 * max(0.0, 3.0 * track.length / 60.0 - lap_times[0])
    info = {"progress": progress, "laps": lap_times, "reason": reason, "time": s.t, "speed": s.speed}
    return (fitness, info, rows) if record else (fitness, info)


# ---- evolution strategies --------------------------------------------------------------------
@dataclass
class TrainConfig:
    track: str = "short"
    population: int = 32  # perturbations per generation (half antithetic pairs)
    starts: int = 4  # practice runs per generation: one from the line, the rest at random places
    start_time: float = 14.0  # s per random-start episode
    start_speed: tuple = (10.0, 35.0)  # m/s range of start speeds
    start_distance: float = 500.0  # m each random-start practice run must cover (the line run covers the full lap)
    crash_penalty_frac: float = 0.5  # crash penalty as a fraction of the distance the run had to cover
    time_bonus: float = 10.0  # fitness per second saved on a practice run (small next to the crash penalty)
    crash_speed_penalty: float = 6.0  # per m/s at impact: gives a gradient towards braking before a corner
    corner_focus: float = 0.5  # share of the random starts placed on the approach to a corner
    corner_speed: tuple = (25.0, 55.0)  # m/s start speeds for those corner-approach runs
    corner_approach: tuple = (40.0, 250.0)  # m before the corner entry that those runs start
    sigma: float = 0.05  # perturbation size
    lr: float = 0.03
    weight_decay: float = 0.002
    hidden: tuple = (32, 32)
    seed: int = 1
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999):
        self.m, self.v, self.t, self.lr, self.b1, self.b2 = np.zeros(n), np.zeros(n), 0, lr, b1, b2

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + 1e-8)


def _eval_one(args):
    theta, cfgs = args
    if isinstance(cfgs, EpisodeConfig):
        return run_episode(theta, cfgs)
    res = [run_episode(theta, c) for c in cfgs]  # several random starts: average the fitness
    return float(np.mean([r[0] for r in res])), {"progress": float(np.mean([r[1]["progress"] for r in res])),
                                                 "laps": [], "reason": "starts"}


def _ranks(x):
    """Centred ranks in [-0.5, 0.5] (robust to the reward scale)."""
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def corner_entries(track, threshold: float = 0.015):
    """Distances (m along the lap) where the track turns tighter than 1/threshold metres radius."""
    k = np.abs(np.asarray(track.curvature))
    idx = np.where((k[1:] > threshold) & (k[:-1] <= threshold))[0] + 1
    return [float(track.s[i]) for i in idx]


def run_dir(track: str) -> Path:
    return RUNS_DIR / track


def save_checkpoint(path: Path, theta, adam, gen, best, cfg, history):
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "checkpoint.tmp.npz"
    np.savez(tmp, theta=theta, m=adam.m, v=adam.v, t=adam.t, gen=gen,
             best_theta=best["theta"], best_fitness=best["fitness"])
    os.replace(tmp, path / "checkpoint.npz")
    np.save(path / "policy.tmp.npy", best["theta"])
    os.replace(path / "policy.tmp.npy", path / "policy.npy")  # the best driver so far (for drive / 3D)
    (path / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    with open(path / "progress.csv", "a") as fh:
        if fh.tell() == 0:
            fh.write("generation,practice_mean,practice_best,best_ever,test_progress_m,test_lap_done,test_lap_s,minutes\n")
        fh.write(history + "\n")


def train(cfg: TrainConfig, workers: int = 2, generations: int | None = None, hours: float | None = None,
          log=print):
    """Train (or continue training) the driver for cfg.track. Ctrl+C stops after saving."""
    out = run_dir(cfg.track)
    rng = np.random.default_rng(cfg.seed)
    policy = Policy(N_OBS, cfg.hidden).init_random(rng)
    adam = Adam(policy.size, cfg.lr)
    gen, best = 0, {"theta": policy.theta.copy(), "fitness": -np.inf}
    ck = out / "checkpoint.npz"
    if ck.exists():
        d = np.load(ck)
        policy.theta, adam.m, adam.v, adam.t, gen = d["theta"], d["m"], d["v"], int(d["t"]), int(d["gen"])
        best = {"theta": d["best_theta"], "fitness": float(d["best_fitness"])}
        rng = np.random.default_rng(cfg.seed + gen)
        log(f"resuming {cfg.track} from generation {gen} (best fitness {best['fitness']:.0f})")
    t_start = time.time()
    t_before = 0.0  # minutes trained in earlier runs (training time adds up across resumes)
    prog_file = out / "progress.csv"
    if ck.exists() and prog_file.exists():
        last = prog_file.read_text().strip().splitlines()[-1].split(",")
        try:
            t_before = float(last[-1])
        except ValueError:
            pass
    half = cfg.population // 2
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            while True:
                if generations is not None and gen >= generations:
                    break
                if hours is not None and time.time() - t_start > hours * 3600:
                    break
                eps = rng.standard_normal((half, policy.size))
                thetas = [policy.theta + cfg.sigma * e for e in eps] + [policy.theta - cfg.sigma * e for e in eps]
                # Training episodes: the same few random starts for every perturbation (fair comparison).
                tr = TRACKS[cfg.track]()
                track_len = tr.length
                # One practice run always starts from the line and must cover the WHOLE lap like the real
                # test (its braking zones matter most); the others start at random places and speeds and
                # cover start_distance metres. Distance caps (not time caps) so that surviving beats
                # crashing fast; the time limit is only a safety net for crawling cars.
                start_dist = min(cfg.start_distance, track_len)
                corners = corner_entries(tr)
                starts = [EpisodeConfig(track=cfg.track, max_time=cfg.episode.max_time, laps=99,
                                        start_s=0.0, start_speed=cfg.episode.start_speed,
                                        lap_bonus=False, max_distance=track_len,
                                        crash_penalty=cfg.crash_penalty_frac * track_len,
                                        crash_speed_penalty=cfg.crash_speed_penalty,
                                        time_bonus=cfg.time_bonus)]
                for _ in range(cfg.starts - 1):
                    if corners and rng.random() < cfg.corner_focus:  # practise braking for a corner
                        c = corners[int(rng.integers(len(corners)))]
                        start_s = float((c - rng.uniform(*cfg.corner_approach)) % track_len)
                        speed = float(rng.uniform(*cfg.corner_speed))
                    else:
                        start_s = float(rng.uniform(0, track_len))
                        speed = float(rng.uniform(*cfg.start_speed))
                    starts.append(EpisodeConfig(track=cfg.track, max_time=start_dist / 8.0, laps=99,
                                                start_s=start_s, start_speed=speed,
                                                lap_bonus=False, max_distance=start_dist,
                                                crash_penalty=cfg.crash_penalty_frac * start_dist,
                                                crash_speed_penalty=cfg.crash_speed_penalty,
                                                time_bonus=cfg.time_bonus))
                # Plus the real test: a full lap from the line with the current network.
                jobs = [(th, starts) for th in thetas] + [(policy.theta, cfg.episode)]
                results = list(ex.map(_eval_one, jobs))
                fit = np.array([r[0] for r in results[:-1]])
                centre_fit, centre_info = results[-1]
                w = _ranks(fit)
                grad = (w[:half] - w[half:]) @ eps / (half * cfg.sigma)
                policy.theta = policy.theta + adam.step(grad - cfg.weight_decay * policy.theta)
                gen += 1
                # The best driver is judged on the real test (a lap from the line), not the practice starts.
                if centre_fit > best["fitness"]:
                    best = {"theta": jobs[-1][0].copy(), "fitness": float(centre_fit)}
                lap = centre_info["laps"][0] if centre_info["laps"] else None
                mins = t_before + (time.time() - t_start) / 60
                line = (f"{gen},{fit.mean():.1f},{fit.max():.1f},{best['fitness']:.1f},"
                        f"{centre_info['progress']:.1f},{int(lap is not None)},"
                        f"{lap if lap else float('nan'):.3f},{mins:.2f}")
                save_checkpoint(out, policy.theta, adam, gen, best, cfg, line)
                test = f"lap {lap:.2f} s" if lap else f"{centre_info['progress']:.0f} m then {centre_info['reason']}"
                log(f"gen {gen:4d}  practice mean {fit.mean():6.0f} best {fit.max():6.0f}  |  test drive: {test}"
                    f"  |  best ever {best['fitness']:.0f}  {mins:5.1f} min")
    except KeyboardInterrupt:
        log("stopped; progress is saved — run the same command to continue")
    return best


def load_policy(track: str):
    p = run_dir(track) / "policy.npy"
    if not p.exists():
        raise FileNotFoundError(f"no trained driver for {track!r} yet: run python -m carsim.learn train --track {track}")
    return np.load(p)


class LearnedDriver:
    """Drives with a trained network (for the 3D server and comparisons). Reloads newer files."""

    def __init__(self, track, track_name: str, reload=True):
        self.track, self.name, self.reload = track, track_name, reload
        self.path = run_dir(track_name) / "policy.npy"
        self._mtime = 0.0
        self.policy = Policy(N_OBS)
        self._load()
        self.i = 0
        self._next_t = 0.0
        self._inp = Inputs()

    def _load(self):
        if self.path.exists():
            m = self.path.stat().st_mtime
            if m != self._mtime:
                self.policy.theta = np.load(self.path)
                self._mtime = m
                return True
        return False

    def maybe_reload(self):
        return self.reload and self._load()

    def __call__(self, s, car):
        if s.t + 1e-9 >= self._next_t:  # decide at 50 Hz, like in training
            self._next_t = s.t + CONTROL_DT
            self.i = self.track.nearest(s.x, s.y, self.i)
            act = self.policy.act(observe(car, self.track, self.i))
            pedal = float(act[1])
            self._inp = Inputs(throttle=max(pedal, 0.0), brake=max(-pedal, 0.0),
                               steer=float(act[0]) * car.p.chassis.max_steer, gear=auto_gear(s, car))
        else:
            self._inp = Inputs(throttle=self._inp.throttle, brake=self._inp.brake, steer=self._inp.steer,
                               gear=auto_gear(s, car))
        return self._inp


# ---- command line ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m carsim.learn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--track", choices=TRACKS, default="short")
    t.add_argument("--workers", type=int, default=max(os.cpu_count() or 2, 1))
    t.add_argument("--generations", type=int)
    t.add_argument("--hours", type=float)
    t.add_argument("--population", type=int, default=32)
    t.add_argument("--max-time", type=float, default=None, help="s per episode (default: enough for ~1 lap)")
    for name in ("status", "drive"):
        p = sub.add_parser(name)
        p.add_argument("--track", choices=TRACKS, default="short")
    a = ap.parse_args(argv)

    if a.cmd == "train":
        default_time = {"short": 70.0, "test": 120.0, "hill": 120.0}[a.track]  # room for a slow first lap
        cfg = TrainConfig(track=a.track, population=a.population,
                          episode=EpisodeConfig(track=a.track, max_time=a.max_time or default_time))
        print(f"training a driver for the {a.track} circuit with {a.workers} workers; checkpoints in {run_dir(a.track)}/")
        train(cfg, workers=a.workers, generations=a.generations, hours=a.hours)
    elif a.cmd == "status":
        f = run_dir(a.track) / "progress.csv"
        if not f.exists():
            print("no training yet")
            return
        lines = f.read_text().strip().splitlines()
        print(lines[0])
        for ln in lines[1:][-15:]:
            print(ln)
    elif a.cmd == "drive":
        cfg = EpisodeConfig(track=a.track, max_time=200.0)
        fit, info = run_episode(load_policy(a.track), cfg)
        lap = f"lap {info['laps'][0]:.2f} s" if info["laps"] else f"no lap ({info['reason']} after {info['progress']:.0f} m)"
        print(f"learned driver on {a.track}: {lap}")


if __name__ == "__main__":
    main()
