"""The 2026 F1 calendar: 24 real circuits kept as a held-out (report-only) test set."""
import argparse

import numpy as np
import pytest

from carsim import learn
from carsim.circuits import TRACKS, get_track, monza, track_arg
from carsim.f1_calendar import BY_KEY, CALENDAR, DATA, is_calendar, track_names
from carsim.trackgen import track_stats

# Race direction of every circuit (Suzuka, a figure of eight, turns as much left as right: not checked).
ANTICLOCKWISE = {"jeddah", "miami", "baku", "singapore", "austin", "interlagos", "las_vegas", "yas_marina"}


def test_the_calendar_has_24_rounds_in_order():
    assert [c.round for c in CALENDAR] == list(range(1, 25))
    assert len(BY_KEY) == 24 and len(track_names()) == 24
    assert {c.key for c in CALENDAR if c.cancelled} == {"bahrain", "jeddah"}
    assert len(track_names(include_cancelled=False)) == 22


@pytest.mark.parametrize("c", CALENDAR, ids=lambda c: c.key)
def test_every_circuit_loads_with_its_real_length_width_and_direction(c):
    assert (DATA / ("monza.csv" if c.key == "monza" else f"{c.key}.csv")).exists()
    tr = get_track("f1:" + c.key)
    assert tr.length == pytest.approx(c.length_m, rel=0.012)  # all within 1.2 % of the official length
    w = tr.width_left + tr.width_right
    assert 7.0 < w.min() and w.max() < 30.0
    assert np.all(tr.width_left > 2.5) and np.all(tr.width_right > 2.5)
    st = track_stats(tr)
    assert 5.0 < st["min_radius_m"] < 40.0  # real hairpins, no kinks tighter than a few metres
    if c.key != "suzuka":
        assert st["direction"] == ("anticlockwise" if c.key in ANTICLOCKWISE else "clockwise")
    # the lap closes on itself smoothly: no jump between the last sample and the first
    gap = np.hypot(tr.x[0] - tr.x[-1], tr.y[0] - tr.y[-1])
    assert gap < 2 * tr.length / len(tr.s)


def test_calendar_monza_is_the_same_monza():
    a, b = get_track("f1:monza"), monza()
    assert np.array_equal(a.x, b.x) and np.array_equal(a.curvature, b.curvature)


def test_track_names_and_registry():
    assert track_arg("f1:suzuka") == "f1:suzuka" and "f1:suzuka" in TRACKS
    assert "f1:nowhere" not in TRACKS
    with pytest.raises(argparse.ArgumentTypeError):
        track_arg("f1:nowhere")
    with pytest.raises(KeyError):
        get_track("f1:nowhere")
    assert is_calendar("f1:spa") and not is_calendar("monza") and not is_calendar("random:3")
    # the hand-made / named circuits are untouched: the calendar is not one of them
    assert "suzuka" not in TRACKS


# ---- held out: never trained on, never used to pick the best driver -------------------------
def test_training_refuses_calendar_circuits(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "RUNS_DIR", tmp_path)
    for argv in (["train", "--track", "f1:suzuka"], ["train", "--tracks", "short,f1:spa"]):
        with pytest.raises(SystemExit):
            learn.main(argv)
    with pytest.raises(ValueError, match="held out"):
        learn.train(learn.TrainConfig(track="f1:monaco"), generations=1)
    with pytest.raises(ValueError, match="held out"):
        learn.train(learn.TrainConfig(track="random", validation=("random:0", "f1:monaco")), generations=1)
    assert not any(tmp_path.iterdir())  # refused before anything was written


def test_random_practice_never_draws_a_calendar_circuit():
    cfg = learn.TrainConfig(track="random", power_share=0.5, tracks_per_gen=3, starts=6)
    rng = np.random.default_rng(0)
    names = {r.track for g in range(30) for r in learn.practice_runs(cfg, rng, g)}
    names |= {t.track for t in learn.test_configs(cfg)}
    assert names and not any(is_calendar(n) for n in names)
