"""Track realism: kerbs only in corners, mapped run-off surfaces, dirty tyres off the tarmac,
published elevation ranges, and the 3D view's surface message."""
import numpy as np
import pytest

from carsim import thermal
from carsim.circuits import PUBLISHED_RANGE_M, get_track, published_scale
from carsim.runoff import CODE, kerbs, mapped_profile
from carsim.server import _rle, surface_message
from carsim.track import SURFACES, Track


def _hairpins():
    return Track.from_segments("hairpin", [("straight", 400.0), ("arc", 30.0, 180.0), ("straight", 400.0),
                                           ("arc", 30.0, 180.0)])


def test_kerbs_line_the_corners_not_the_straights():
    from carsim.runoff import with_runoff
    tr = with_runoff(_hairpins())
    apex, straight = tr.index_at(400.0 + np.pi * 15), tr.index_at(200.0)
    assert tr.kerb_left[apex] and tr.kerb_right[apex]
    assert not tr.kerb_left[straight] and not tr.kerb_right[straight]
    w = tr.width_left[straight]
    # beside a straight the run-off starts right at the white line: no free kerb strip to drive on
    assert tr.surface_at(w + 0.5, straight) == "grass"
    assert tr.surface_at(tr.width_left[apex] + 0.5, apex) == "kerb"


def test_codes_match_the_track_surfaces():
    assert all(SURFACES[v] == k for k, v in CODE.items())


def test_mapped_circuits_have_gravel_traps_and_barriers():
    tr = get_track("f1:silverstone")
    assert mapped_profile("silverstone", len(tr.s)) is not None
    both = np.r_[tr.surf_left, tr.surf_right]
    assert (both == CODE["gravel"]).mean() > 0.05 and (both == CODE["barrier"]).mean() > 0.05
    assert mapped_profile("silverstone", len(tr.s) + 1) is None  # another sampling: not used


def test_dirty_tyres_lose_grip_and_clean_up_on_asphalt():
    tp = thermal.ThermalParams()
    st = thermal.TyreThermalState.from_blankets(tp)
    st.t_surface = st.t_core = tp.t_opt
    clean = thermal.grip_factor(tp, st)
    for _ in range(20):
        thermal.roll_surface(tp, st, "grass", 1.0)  # 20 m on the grass
    assert st.dirt > 0.95
    assert thermal.grip_factor(tp, st) == pytest.approx(clean * (1 - tp.dirt_grip_loss * st.dirt))
    thermal.roll_surface(tp, st, "track", 1000.0)  # a kilometre of asphalt
    assert st.dirt < 0.25
    thermal.roll_surface(tp, st, "kerb", 5000.0)
    assert st.dirt < 0.01


def test_elevation_follows_the_published_height_range():
    z = np.array([0.0, 17.4, 5.0])
    assert published_scale("monza", z) == pytest.approx(12.8 / 17.4)
    assert published_scale("spa", np.array([0.0, 105.1])) == 1.0  # within 10 %
    assert published_scale("monaco", np.zeros(3)) == 1.0  # no figure / kept flat
    tr = get_track("f1:melbourne")
    assert float(tr.z.max() - tr.z.min()) == pytest.approx(PUBLISHED_RANGE_M["melbourne"], rel=1e-6)


def test_surface_message_runs_add_up():
    assert _rle([0, 0, 1, 1, 1, 3]) == "A2G3B1"
    m = surface_message(get_track("f1:silverstone"))
    assert len(m["kerb_left"]) == len(m["surf_left"])
    for code in m["surf_left"][::500]:
        assert sum(int(x) for x in __import__("re").findall(r"\d+", code)) == 60
