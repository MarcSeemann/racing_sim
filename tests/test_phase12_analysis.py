"""Phase 12: what-if studies, command line, lap viewer, live delta."""
import json
import re

import numpy as np
import pytest

from carsim.__main__ import main as cli, parse_change
from carsim.compare import LiveDelta
from carsim.track import short_circuit
from carsim.whatif import PRESETS, Scenario, apply_change, results_table, run_scenario, run_study


def test_parse_change_operators():
    assert parse_change("car.aero.ClA=*1.1") == ("mul", "car.aero.ClA", 1.1)
    assert parse_change("weather.air_temp_c=+5") == ("add", "weather.air_temp_c", 5.0)
    assert parse_change("weather.air_temp_c=-5") == ("add", "weather.air_temp_c", -5.0)
    assert parse_change("scenario.compound=wet") == ("set", "scenario.compound", "wet")
    assert parse_change("car.chassis.brake_front_bias=0.55") == ("set", "car.chassis.brake_front_bias", 0.55)


def test_apply_change_copies_and_targets_every_layer():
    base = Scenario()
    hot = apply_change(base, ("add", "weather.air_temp_c", 5.0))
    assert hot.weather[0].air_temp_c == base.weather[0].air_temp_c + 5.0
    aero = apply_change(base, ("mul", "car.aero.ClA", 1.1))
    assert aero.car.aero.ClA == pytest.approx(4.4)
    assert base.car is None  # the baseline is untouched
    sc = apply_change(base, ("set", "scenario.compound", "wet"))
    assert sc.compound == "wet"
    with pytest.raises(ValueError):
        apply_change(base, ("set", "nowhere.x", 1))


def test_every_preset_change_applies():
    for study in PRESETS.values():
        for ch in study.values():
            sc = Scenario()
            for c in (ch if isinstance(ch, list) else [ch]):
                sc = apply_change(sc, c)


@pytest.fixture(scope="module")
def downforce_study():
    base = Scenario(laps=2, tyre_thermals=False)
    return run_study(base, {"+10% downforce": ("mul", "car.aero.ClA", 1.10),
                            "-10% downforce": ("mul", "car.aero.ClA", 0.90)})


def test_more_downforce_is_faster_on_the_short_circuit(downforce_study):
    r = dict(downforce_study)
    assert r["+10% downforce"]["lap_time"] < r["baseline"]["lap_time"] < r["-10% downforce"]["lap_time"]
    assert r["+10% downforce"]["max_lat_g"] > r["-10% downforce"]["max_lat_g"]


def test_results_table_has_a_row_per_run(downforce_study):
    table = results_table(downforce_study)
    lines = table.splitlines()
    assert len(lines) == 2 + len(downforce_study)
    assert "+0.000" in lines[2]  # baseline delta


def test_cli_presets_and_compare_viewer(tmp_path, capsys):
    cli(["presets"])
    assert "hot-day" in capsys.readouterr().out
    out = tmp_path / "cmp.html"
    cli(["compare", "--laps", "2", "--no-thermals", "--change", "car.chassis.mass=+40", "--name", "+40 kg",
         "--out", str(out)])
    html = out.read_text()
    assert "<canvas" in html and "+40 kg" in html
    data = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))
    assert len(data["laps"]) == 2
    assert data["laps"][1]["lap_time"] > data["laps"][0]["lap_time"]  # heavier = slower


def test_live_delta_against_the_best_lap():
    tr = short_circuit()
    n = len(tr.s)
    live = LiveDelta(tr)
    t = 0.0
    for lap_time in (30.0, 28.0, 29.0):  # constant speed laps of different length in time
        for i in range(n):
            gap = live.update(i, t + lap_time * i / n)
        t += lap_time
    live.update(0, t)
    assert live.best_time == pytest.approx(28.0, abs=0.05)
    assert gap == pytest.approx(1.0 * (n - 1) / n, abs=0.05)  # the third lap ends ~1 s down
