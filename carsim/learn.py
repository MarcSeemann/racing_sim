"""An AI driver that teaches itself to drive a track — evolution strategies (numpy only).

The driver is a small neural network (an MLP). It sees what a driver feels and sees:
speed, sideslip, yaw rate, where it is across the track, its angle to the track, how
the track curves (and climbs) ahead, wheel slip, how the tyres and brakes feel (grip,
temperature), the pedals and steering as they actually are, and the fuel load. It outputs
a steering target and one pedal (positive = throttle, negative = brake). Gears are automatic.

It drives the SAME car the 3D view shows: the race car with fuel, tyre temperatures and
wear, carbon brakes with a temperature window, kerbs and grass, a rubbered racing line —
and the physical control limits of phase 13: the steering, throttle and brake move at
finite rates, so the network sets a target and the car's actuators chase it. (It cannot
flick the wheel from lock to lock to scrub heat into the tyres.)

Training (OpenAI-style evolution strategies): each generation, try many randomly perturbed
copies of the network, drive each one through the same practice runs, and move the network
towards the perturbations that scored best. Practice runs:

    one run from the line covering `laps` laps (default 3) — the second and third laps
        start with different momentum and warmer tyres than the first, so the driver has
        to manage a stint, not one perfect lap
    a few short runs (start_distance m) from random places, speeds, tyre and brake
        temperatures and fuel loads — half of them on the approach to a corner

Runs are capped by DISTANCE, not time. Leaving the track costs a penalty (more if fast);
reaching the cap earlier earns a small bonus: stay on track > cover the distance > be quick.
The best driver is judged on a test run of `laps` laps from the line.

Several tracks: --tracks short,test,hill trains one driver for all of them (the practice
runs are spread over the tracks; the test drives every track). Until the driver is good
on one track it is better to keep training on that one.

Train (resumes from the last checkpoint). A run saved with older physics/observations is moved to
ai_runs/<track>/legacy_<n>/ and training starts fresh; --carry-over keeps its weights instead
(matching inputs keep their weights, new inputs start at zero):

    python -m carsim.learn train --track short --workers 64          # Ctrl+C any time
    nohup python -m carsim.learn train --track short --hours 12 > train.log 2>&1 &
    python -m carsim.learn train --tracks short,test,hill --init-from short   # later: several tracks
    python -m carsim.learn status --track short                       # progress so far
    python -m carsim.learn drive --track short                        # 3 laps with the best driver
    python drive3d.py --track short --learned                         # watch it in 3D
"""
import argparse
import dataclasses
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .conditions import Conditions, Weather
from .drivers import auto_gear
from .dynamics import Car2D
from .params import CarParams, race_car
from .racing_line import cached_racing_line
from .track import hill_circuit, short_circuit
from .track import test_circuit as long_circuit
from .vehicle import Inputs

TRACKS = {"short": short_circuit, "test": long_circuit, "hill": hill_circuit}
LOOKAHEAD_M = (5, 15, 30, 50, 80, 120, 170, 230)
GRADE_AHEAD_M = (0, 60, 150)
CONTROL_DT = 0.02  # the network decides 50 times a second; physics runs at 1 kHz
PHYS_DT = 0.001
RUNS_DIR = Path(os.environ.get("CARSIM_RUNS", "ai_runs"))
# Bumped whenever the car's physics changes enough that old drivers/scores are not comparable:
# a saved run with another version is archived (see _load_start). 2 = phase 13 limits, 3 = phase 14
# suspension, kerbs, camber, LSD, ERS.
PHYSICS_VERSION = 3

# Observation layout, by name: a checkpoint saved with an older layout is mapped onto this one
# (matching inputs keep their weights, new inputs start at zero, so the old driver drives the same).
OBS_V1 = (["vx", "vy", "yaw_rate", "cross_track", "sin_heading_err", "cos_heading_err",
           "alpha_front", "alpha_rear", "kappa_rear", "steer"] + [f"curv_{d}" for d in LOOKAHEAD_M])
OBS_NAMES = (OBS_V1 + ["throttle", "brake", "kappa_front", "grip_front", "grip_rear",
                      "tyre_temp_front", "tyre_temp_rear", "brake_temp_front", "brake_temp_rear",
                      "fuel", "surface_grip"] + [f"grade_{d}" for d in GRADE_AHEAD_M] + ["crest_40"]
             + ["body_roll", "body_pitch", "ride_front", "ers_soc"])


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


def migrate_theta(theta, old_names, new_names=None, hidden=(32, 32)):
    """Map network weights trained on observation layout old_names onto new_names.

    The first layer gets one row per input: rows of inputs that exist in both layouts are
    copied, new inputs get zero rows (so they do not change what the driver does until training
    finds a use for them). Deeper layers are unchanged."""
    new_names = list(OBS_NAMES if new_names is None else new_names)
    old_names = list(old_names)
    old = Policy(len(old_names), hidden, theta=theta)
    new = Policy(len(new_names), hidden)
    h0 = hidden[0]
    W_old = np.asarray(theta[:len(old_names) * h0]).reshape(len(old_names), h0)
    W_new = np.zeros((len(new_names), h0))
    for j, n in enumerate(new_names):
        if n in old_names:
            W_new[j] = W_old[old_names.index(n)]
    rest = np.asarray(theta[len(old_names) * h0:])
    out = np.concatenate([W_new.ravel(), rest])
    assert out.size == new.size and rest.size == old.size - len(old_names) * h0
    return out


def adapt_theta(theta, hidden=(32, 32), obs_names=None):
    """Return weights for the current observation layout (migrating older layouts)."""
    theta = np.asarray(theta, float)
    if theta.size == Policy(N_OBS, hidden).size and (obs_names is None or list(obs_names) == OBS_NAMES):
        return theta, False
    if obs_names is None:
        if theta.size != Policy(len(OBS_V1), hidden).size:
            raise ValueError(f"network of size {theta.size} does not fit observation layout v1 or current")
        obs_names = OBS_V1
    return migrate_theta(theta, obs_names, OBS_NAMES, hidden), True


# ---- observations and one episode --------------------------------------------------------
def _mean(a, b):
    return 0.5 * (a + b)


def observe(car, track, i: int) -> np.ndarray:
    s = car.state
    h = track.heading[i]
    err = track.cross_track_error(i, s.x, s.y) / (track.width / 2)
    ang = math.remainder(s.psi - h, 2 * math.pi)
    s_i = track.s[i]
    curv = [track.curvature[track.index_at(s_i + d)] * 30.0 for d in LOOKAHEAD_M]
    a_front = _mean(s.alpha[0], s.alpha[1])
    a_rear = _mean(s.alpha[2], s.alpha[3])
    k_rear = _mean(s.kappa[2], s.kappa[3])
    k_front = min(s.kappa[0], s.kappa[1])
    obs = [s.vx / 60.0, s.vy / 5.0, s.r / 1.5, err, math.sin(ang) * 3, math.cos(ang) - 1,
           a_front * 5, a_rear * 5, max(min(k_rear * 5, 3), -3), s.steer / 0.35]
    obs += curv
    # How the car feels: pedals as they are, front lock-up, tyre grip and temperature, brakes, fuel.
    g = car.grip_factors()
    t_opt = car.thermal.t_opt
    ty = s.tyres
    bt = s.brake_temp
    sg = getattr(car, "surface_grip", None) or [1.0] * 4
    obs += [s.throttle, s.brake, max(min(k_front * 5, 3), -3),
            (_mean(g[0], g[1]) - 1) * 10, (_mean(g[2], g[3]) - 1) * 10,
            (_mean(ty[0].t_surface, ty[1].t_surface) - t_opt) / 30, (_mean(ty[2].t_surface, ty[3].t_surface) - t_opt) / 30,
            (_mean(bt[0], bt[1]) - 650) / 300, (_mean(bt[2], bt[3]) - 650) / 300,
            car.fuel / 100.0, (min(sg) - 1) * 3]
    obs += [float(track.grade[track.index_at(s_i + d)]) * 10 for d in GRADE_AHEAD_M]
    obs += [float(track.vert_curv[track.index_at(s_i + 40)]) * 1000]
    # The body's motion through the seat, the floor's height, the battery.
    e = car.p.ers
    obs += [s.ph * 100.0, s.th * 200.0, (s.hf - 0.02) * 100.0,
            (car.ers_energy / e.store - 0.5) * 2 if e.enabled else 0.0]
    return np.array(obs)


N_OBS = len(OBS_NAMES)


@dataclass
class EpisodeConfig:
    track: str = "short"
    max_time: float = 210.0  # s of simulated driving per episode (safety net for crawling cars)
    laps: int = 3
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
    # The car and the track (realistic = the same car and conditions as the 3D view).
    realistic: bool = True  # False: the old training car (no fuel, tyres always optimal, no kerbs/grass)
    fuel_kg: float = 40.0
    tyre_temp: float | None = None  # deg C for all four tyres; None = straight off the blankets
    brake_temp: float | None = None  # deg C; None = the car's default start temperature
    rubber: float = 2e6  # rubber on the racing line (J per bin), as in the 3D view
    phys_dt: float = PHYS_DT


def make_car(track, cfg: EpisodeConfig, i: int = 0):
    """The car for an episode (and the 3D view): race car, fuel, tyre and brake temperatures,
    kerbs, grass and a rubbered racing line."""
    if not cfg.realistic:
        return Car2D(CarParams(tyre_thermals=False))
    cond = Conditions(track, Weather.dry(), cars_on_track=20, rubber=cfg.rubber)
    cond.set_racing_line(cached_racing_line(track)[1])  # the field has rubbered in the racing line
    cond.i = i  # where the car starts (the surface lookup searches near the last position)
    car = Car2D(race_car(cfg.fuel_kg), conditions=cond)
    if cfg.tyre_temp is not None:
        for ts in car.state.tyres:
            ts.t_surface, ts.t_core = cfg.tyre_temp, cfg.tyre_temp - 5.0
    if cfg.brake_temp is not None:
        car.state.brake_temp = [cfg.brake_temp] * 4
    return car


def lap_score(lap_times, laps: int, track_length: float) -> float:
    """Bonus for completed laps: each lap counts, and finishing all of them faster scores more."""
    n = min(len(lap_times), laps)
    score = 1000.0 * n
    if n >= laps:
        ref_lap = 3.0 * track_length / 60.0  # a 20 m/s lap earns no time bonus
        score += 50.0 * max(0.0, laps * ref_lap - sum(lap_times[:laps]))
    return score


def _start_gear(car, v):
    """Highest gear that still keeps the engine above ~10 000 rpm at speed v."""
    best = 1
    for g in range(1, len(car.p.gearbox.ratios) + 1):
        if car.wheel_rpm_to_engine(v, g) > 10000:
            best = g
    return best


def run_episode(theta, cfg: EpisodeConfig, track=None, record=False, hidden=(32, 32)):
    """Drive one episode with network weights theta. Returns (fitness, info[, telemetry])."""
    track = track or TRACKS[cfg.track]()
    policy = Policy(N_OBS, hidden, theta=theta)
    i = track.index_at(cfg.start_s)
    car = make_car(track, cfg, i)
    s = car.state
    s.x, s.y, s.psi = float(track.x[i]), float(track.y[i]), float(track.heading[i])
    car.set_speed(cfg.start_speed)
    s.gear = _start_gear(car, cfg.start_speed)
    progress, last_s = 0.0, float(track.s[i])
    n_sub = max(int(round(CONTROL_DT / cfg.phys_dt)), 1)
    lap_times, lap_start = [], 0.0
    reason = "time"
    rows = []
    slow = 0.0
    steer_travel = 0.0
    while s.t < cfg.max_time:
        i = track.nearest(s.x, s.y, i)
        act = policy.act(observe(car, track, i))
        steer = float(act[0]) * car.p.chassis.max_steer
        pedal = float(act[1])
        inp = Inputs(throttle=max(pedal, 0.0), brake=max(-pedal, 0.0), steer=steer, gear=auto_gear(s, car))
        before = s.steer
        for _ in range(n_sub):
            car.step(inp, cfg.phys_dt, row=False)
        steer_travel += abs(s.steer - before)
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
            rows.append((s.t, s.x, s.y, s.speed * 3.6, s.steer, pedal, steer, s.throttle, s.brake,
                         s.tyres[0].t_surface, s.brake_temp[0]))
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
        fitness += lap_score(lap_times, cfg.laps, track.length)
    info = {"progress": progress, "laps": lap_times, "reason": reason, "time": s.t, "speed": s.speed,
            "steer_travel": steer_travel, "track": cfg.track}
    return (fitness, info, rows) if record else (fitness, info)


# ---- evolution strategies --------------------------------------------------------------------
@dataclass
class TrainConfig:
    track: str = "short"
    tracks: tuple = ()  # several tracks: one driver for all of them (empty = just `track`)
    population: int = 32  # perturbations per generation (half antithetic pairs)
    laps: int = 3  # laps of the run from the line and of the test drive
    starts: int = 4  # practice runs per generation: one from the line (laps laps), the rest at random places
    start_time: float = 14.0  # (kept for old configs; random-start runs are capped by distance)
    start_speed: tuple = (10.0, 35.0)  # m/s range of start speeds
    start_distance: float = 500.0  # m each random-start practice run must cover
    crash_penalty_frac: float = 0.5  # crash penalty as a fraction of the distance the run had to cover
    time_bonus: float = 10.0  # fitness per second saved on a practice run (small next to the crash penalty)
    crash_speed_penalty: float = 6.0  # per m/s at impact: gives a gradient towards braking before a corner
    corner_focus: float = 0.5  # share of the random starts placed on the approach to a corner
    corner_speed: tuple = (25.0, 55.0)  # m/s start speeds for those corner-approach runs
    corner_approach: tuple = (40.0, 250.0)  # m before the corner entry that those runs start
    # Random starts also vary how the car feels, so the driver learns to read it:
    tyre_temp: tuple = (60.0, 115.0)  # deg C
    brake_temp: tuple = (250.0, 900.0)  # deg C
    fuel_kg: tuple = (5.0, 60.0)
    sigma: float = 0.05  # perturbation size
    lr: float = 0.03
    weight_decay: float = 0.002
    hidden: tuple = (32, 32)
    seed: int = 1
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)  # the test drive (laps from the line)

    def track_list(self) -> list:
        return list(self.tracks) if self.tracks else [self.track]

    def run_name(self) -> str:
        t = self.track_list()
        return t[0] if len(t) == 1 else "multi_" + "-".join(t)


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


_TRACK_CACHE = {}


def _track(name):
    """One Track object per name and process (keeps the racing-line cache warm)."""
    if name not in _TRACK_CACHE:
        _TRACK_CACHE[name] = TRACKS[name]()
    return _TRACK_CACHE[name]


def _eval_one(args):
    theta, cfgs, hidden = args
    if isinstance(cfgs, EpisodeConfig):
        return run_episode(theta, cfgs, _track(cfgs.track), hidden=hidden)
    res = [run_episode(theta, c, _track(c.track), hidden=hidden) for c in cfgs]  # average the practice runs
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


PROGRESS_HEADER = ("generation,practice_mean,practice_best,best_ever,test_progress_m,test_lap_done,"
                   "test_lap_s,minutes,test_laps,test_total_s,test_steer_rad_per_s")


def save_checkpoint(path: Path, theta, adam, gen, best, cfg, history):
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "checkpoint.tmp.npz"
    np.savez(tmp, theta=theta, m=adam.m, v=adam.v, t=adam.t, gen=gen,
             best_theta=best["theta"], best_fitness=best["fitness"])
    os.replace(tmp, path / "checkpoint.npz")
    np.save(path / "policy.tmp.npy", best["theta"])
    os.replace(path / "policy.tmp.npy", path / "policy.npy")  # the best driver so far (for drive / 3D)
    meta = asdict(cfg)
    meta["obs_names"] = OBS_NAMES
    meta["physics_version"] = PHYSICS_VERSION
    (path / "config.json").write_text(json.dumps(meta, indent=1))
    with open(path / "progress.csv", "a") as fh:
        if fh.tell() == 0:
            fh.write(PROGRESS_HEADER + "\n")
        fh.write(history + "\n")


def _saved_obs_names(path: Path):
    f = path / "config.json"
    if f.exists():
        try:
            return json.loads(f.read_text()).get("obs_names")
        except (ValueError, OSError):
            return None
    return None


def _saved_physics_version(path: Path):
    f = path / "config.json"
    try:
        return json.loads(f.read_text()).get("physics_version", 1) if f.exists() else None
    except (ValueError, OSError):
        return None


def _archive_old_run(out: Path, log):
    """Move a run trained with an older observation layout / physics into out/legacy_<n>/."""
    k = 1
    while (out / f"legacy_{k}").exists():
        k += 1
    dst = out / f"legacy_{k}"
    dst.mkdir(parents=True)
    for name in ("checkpoint.npz", "policy.npy", "progress.csv", "config.json"):
        if (out / name).exists():
            os.replace(out / name, dst / name)
    log(f"the saved driver used older physics/observations: archived to {dst}/")
    return dst


def _load_start(out: Path, cfg: TrainConfig, policy, adam, log, init_from: str | None = None,
                carry_over: bool = False):
    """Resume from out/, migrate an older run in out/, or start from another run's driver."""
    gen, best = 0, {"theta": policy.theta.copy(), "fitness": -np.inf}
    ck = out / "checkpoint.npz"
    if ck.exists():
        d = np.load(ck)
        names = _saved_obs_names(out)
        theta, migrated = adapt_theta(d["theta"], cfg.hidden, names)
        migrated = migrated or _saved_physics_version(out) != PHYSICS_VERSION
        if not migrated:
            policy.theta, adam.m, adam.v, adam.t, gen = theta, d["m"], d["v"], int(d["t"]), int(d["gen"])
            best = {"theta": d["best_theta"], "fitness": float(d["best_fitness"])}
            log(f"resuming {cfg.run_name()} from generation {gen} (best fitness {best['fitness']:.0f})")
            return gen, best
        # Older layout = older physics: archive it and restart the optimiser and the scores (old
        # fitness values are not comparable). By default the network starts fresh, because the
        # old drivers relied on instant (unphysical) steering; carry_over keeps their weights.
        best_theta, _ = adapt_theta(d["best_theta"], cfg.hidden, names)
        _archive_old_run(out, log)
        if carry_over:
            policy.theta = best_theta
            log("carrying over the old driver's weights (new inputs start at zero)")
            return 0, {"theta": best_theta.copy(), "fitness": -np.inf}
        log("starting a fresh driver for the new physics (use --carry-over to keep the old weights)")
        return 0, best
    if init_from:
        src = run_dir(init_from)
        theta, _ = adapt_theta(np.load(src / "policy.npy"), cfg.hidden, _saved_obs_names(src))
        policy.theta = theta
        log(f"starting {cfg.run_name()} from the driver trained in {src}/")
        return 0, {"theta": theta.copy(), "fitness": -np.inf}
    return gen, best


def practice_runs(cfg: TrainConfig, rng, gen: int) -> list:
    """This generation's practice runs (the same for every perturbation: a fair comparison)."""
    names = cfg.track_list()
    ep = cfg.episode
    common = dict(laps=99, lap_bonus=False, crash_speed_penalty=cfg.crash_speed_penalty,
                  time_bonus=cfg.time_bonus, realistic=ep.realistic, rubber=ep.rubber, phys_dt=ep.phys_dt)
    # The run from the line covers `laps` laps like the real test (rolling start, tyres off the
    # blankets): laps 2 and 3 start with different momentum and warmer tyres than lap 1.
    line_track = names[gen % len(names)]
    L = _track(line_track).length
    runs = [EpisodeConfig(track=line_track, max_time=ep.max_time, start_s=0.0, start_speed=ep.start_speed,
                          max_distance=cfg.laps * L, crash_penalty=cfg.crash_penalty_frac * cfg.laps * L,
                          fuel_kg=ep.fuel_kg, tyre_temp=ep.tyre_temp, brake_temp=ep.brake_temp, **common)]
    for _ in range(cfg.starts - 1):
        name = names[int(rng.integers(len(names)))]
        tr = _track(name)
        dist = min(cfg.start_distance, tr.length)
        corners = corner_entries(tr)
        if corners and rng.random() < cfg.corner_focus:  # practise braking for a corner
            c = corners[int(rng.integers(len(corners)))]
            start_s = float((c - rng.uniform(*cfg.corner_approach)) % tr.length)
            speed = float(rng.uniform(*cfg.corner_speed))
        else:
            start_s = float(rng.uniform(0, tr.length))
            speed = float(rng.uniform(*cfg.start_speed))
        runs.append(EpisodeConfig(track=name, max_time=dist / 8.0, start_s=start_s, start_speed=speed,
                                  max_distance=dist, crash_penalty=cfg.crash_penalty_frac * dist,
                                  tyre_temp=float(rng.uniform(*cfg.tyre_temp)),
                                  brake_temp=float(rng.uniform(*cfg.brake_temp)),
                                  fuel_kg=float(rng.uniform(*cfg.fuel_kg)), **common))
    return runs


def test_configs(cfg: TrainConfig) -> list:
    return [dataclasses.replace(cfg.episode, track=n, laps=cfg.laps) for n in cfg.track_list()]


def train(cfg: TrainConfig, workers: int = 2, generations: int | None = None, hours: float | None = None,
          log=print, init_from: str | None = None, carry_over: bool = False):
    """Train (or continue training) the driver for cfg's track(s). Ctrl+C stops after saving."""
    out = run_dir(cfg.run_name())
    rng = np.random.default_rng(cfg.seed)
    policy = Policy(N_OBS, cfg.hidden).init_random(rng)
    adam = Adam(policy.size, cfg.lr)
    gen, best = _load_start(out, cfg, policy, adam, log, init_from, carry_over)
    rng = np.random.default_rng(cfg.seed + gen)
    t_start = time.time()
    t_before = 0.0  # minutes trained in earlier runs (training time adds up across resumes)
    prog_file = out / "progress.csv"
    if gen > 0 and prog_file.exists():
        lines = prog_file.read_text().strip().splitlines()
        try:
            t_before = float(dict(zip(lines[0].split(","), lines[-1].split(",")))["minutes"])
        except (KeyError, ValueError, IndexError):
            pass
    half = cfg.population // 2
    tests = test_configs(cfg)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            while True:
                if generations is not None and gen >= generations:
                    break
                if hours is not None and time.time() - t_start > hours * 3600:
                    break
                eps = rng.standard_normal((half, policy.size))
                thetas = [policy.theta + cfg.sigma * e for e in eps] + [policy.theta - cfg.sigma * e for e in eps]
                runs = practice_runs(cfg, rng, gen)
                # Plus the real test: `laps` laps from the line on every track, with the current network.
                jobs = [(th, runs, cfg.hidden) for th in thetas] + [(policy.theta, t, cfg.hidden) for t in tests]
                results = list(ex.map(_eval_one, jobs))
                fit = np.array([r[0] for r in results[:len(thetas)]])
                test_res = results[len(thetas):]
                w = _ranks(fit)
                grad = (w[:half] - w[half:]) @ eps / (half * cfg.sigma)
                centre = policy.theta.copy()
                policy.theta = policy.theta + adam.step(grad - cfg.weight_decay * policy.theta)
                gen += 1
                # The best driver is judged on the real test (laps from the line), not the practice runs.
                centre_fit = float(sum(r[0] for r in test_res))
                if centre_fit > best["fitness"]:
                    best = {"theta": centre, "fitness": centre_fit}
                laps = [lt for r in test_res for lt in r[1]["laps"][:cfg.laps]]
                all_done = all(len(r[1]["laps"]) >= cfg.laps for r in test_res)
                best_lap = min(laps) if laps else float("nan")
                total = sum(laps) if all_done else float("nan")
                t_drive = sum(r[1]["time"] for r in test_res)
                steer_rate = sum(r[1]["steer_travel"] for r in test_res) / max(t_drive, 1e-9)
                prog = sum(r[1]["progress"] for r in test_res)
                mins = t_before + (time.time() - t_start) / 60
                line = (f"{gen},{fit.mean():.1f},{fit.max():.1f},{best['fitness']:.1f},{prog:.1f},"
                        f"{int(all_done)},{best_lap:.3f},{mins:.2f},{len(laps)},{total:.3f},{steer_rate:.3f}")
                save_checkpoint(out, policy.theta, adam, gen, best, cfg, line)
                parts = []
                for t, r in zip(tests, test_res):
                    lt = r[1]["laps"][:cfg.laps]
                    txt = " ".join(f"{x:.2f}" for x in lt) or "-"
                    if len(lt) < cfg.laps:
                        txt += f" then {r[1]['reason']} at {r[1]['progress']:.0f} m"
                    parts.append(f"{t.track}: {txt}")
                log(f"gen {gen:4d}  practice mean {fit.mean():6.0f} best {fit.max():6.0f}  |  test laps: "
                    f"{'; '.join(parts)}  |  best ever {best['fitness']:.0f}  {mins:5.1f} min")
    except KeyboardInterrupt:
        log("stopped; progress is saved — run the same command to continue")
    return best


def _run_hidden(path: Path):
    f = path / "config.json"
    if f.exists():
        try:
            return tuple(json.loads(f.read_text()).get("hidden", (32, 32)))
        except (ValueError, OSError):
            pass
    return (32, 32)


def load_policy(track: str):
    """The best driver of run `track` (a track name or a multi-track run name), in the current layout."""
    d = run_dir(track)
    p = d / "policy.npy"
    if not p.exists():
        raise FileNotFoundError(f"no trained driver for {track!r} yet: run python -m carsim.learn train --track {track}")
    return adapt_theta(np.load(p), _run_hidden(d), _saved_obs_names(d))[0]


class LearnedDriver:
    """Drives with a trained network (for the 3D server and comparisons). Reloads newer files.

    The car's own actuator limits (steering rate, pedal rates) apply exactly as in training."""

    def __init__(self, track, track_name: str, reload=True):
        self.track, self.name, self.reload = track, track_name, reload
        self.path = run_dir(track_name) / "policy.npy"
        self._mtime = 0.0
        self.hidden = _run_hidden(run_dir(track_name))
        self.policy = Policy(N_OBS, self.hidden)
        self._load()
        self.i = 0
        self._next_t = 0.0
        self._inp = Inputs()

    def _load(self):
        if self.path.exists():
            m = self.path.stat().st_mtime
            if m != self._mtime:
                self.hidden = _run_hidden(self.path.parent)
                self.policy = Policy(N_OBS, self.hidden, theta=load_policy(self.name))
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
    t.add_argument("--tracks", help="comma-separated tracks for one multi-track driver, e.g. short,test,hill")
    t.add_argument("--init-from", help="start a NEW run from the driver of this run (e.g. short)")
    t.add_argument("--carry-over", action="store_true",
                   help="if the saved run used older physics, keep its weights instead of starting fresh")
    t.add_argument("--workers", type=int, default=max(os.cpu_count() or 2, 1))
    t.add_argument("--generations", type=int)
    t.add_argument("--hours", type=float)
    t.add_argument("--population", type=int, default=32)
    t.add_argument("--laps", type=int, default=3, help="laps of the run from the line and of the test (default 3)")
    t.add_argument("--starts", type=int, default=4, help="practice runs per generation (1 from the line + random)")
    t.add_argument("--max-time", type=float, default=None, help="s per test run (default: ~70 s per lap of the short circuit)")
    t.add_argument("--legacy-car", action="store_true",
                   help="train on the old simplified car (no fuel, ideal tyres, no kerbs/grass)")
    for name in ("status", "drive"):
        p = sub.add_parser(name)
        p.add_argument("--track", default="short", help="track, or run name for status (e.g. multi_short-test)")
        p.add_argument("--run", help="drive: use the driver of this run (default: the track's own)")
        p.add_argument("--laps", type=int, default=3)
    a = ap.parse_args(argv)

    if a.cmd == "train":
        tracks = tuple(x.strip() for x in a.tracks.split(",")) if a.tracks else ()
        for n in tracks:
            if n not in TRACKS:
                ap.error(f"unknown track {n!r} (choose from {', '.join(TRACKS)})")
        per_lap = {"short": 70.0, "test": 120.0, "hill": 120.0}
        longest = max(per_lap[n] for n in (tracks or (a.track,)))
        ep = EpisodeConfig(track=a.track, laps=a.laps, max_time=a.max_time or longest * a.laps,
                           realistic=not a.legacy_car)
        cfg = TrainConfig(track=tracks[0] if tracks else a.track, tracks=tracks, population=a.population,
                          laps=a.laps, starts=a.starts, episode=ep)
        print(f"training a driver for {', '.join(cfg.track_list())} ({a.laps} laps) with {a.workers} workers; "
              f"checkpoints in {run_dir(cfg.run_name())}/")
        train(cfg, workers=a.workers, generations=a.generations, hours=a.hours, init_from=a.init_from,
              carry_over=a.carry_over)
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
        if a.track not in TRACKS:
            ap.error(f"unknown track {a.track!r}")
        cfg = EpisodeConfig(track=a.track, laps=a.laps, max_time=120.0 * a.laps)
        fit, info = run_episode(load_policy(a.run or a.track), cfg, hidden=_run_hidden(run_dir(a.run or a.track)))
        laps = " ".join(f"{x:.2f}" for x in info["laps"]) or "none"
        tail = "" if len(info["laps"]) >= a.laps else f" ({info['reason']} after {info['progress']:.0f} m)"
        print(f"learned driver on {a.track}: laps {laps}{tail}; steering {info['steer_travel'] / info['time']:.2f} rad/s average")


if __name__ == "__main__":
    main()
