"""Pit lanes (carsim.pitlane): mapped ones for the real circuits, an estimate for every other track."""
import numpy as np
import pytest

from carsim.circuits import _get_track
from carsim.f1_calendar import CALENDAR
from carsim.pitlane import DATA, edge_offset, estimated, get_pit_lane, load

NOT_MAPPED = {"melbourne"}  # Albert Park: public roads, no raceway in OpenStreetMap


def _beyond_edge(track, lane, n):
    """m beyond the white line on the lane's side at lane point n (negative: on the track)."""
    return abs(lane.offset[n]) - abs(edge_offset(track, lane.side, lane.s[n]))


@pytest.mark.parametrize("c", CALENDAR, ids=lambda c: c.key)
def test_every_circuit_has_a_drivable_pit_lane(c):
    track = _get_track("f1:" + c.key)
    lane = get_pit_lane("f1:" + c.key, track)
    assert lane.source == ("estimated" if c.key in NOT_MAPPED else "osm")
    assert (DATA / f"{c.key}.csv").exists() == (c.key not in NOT_MAPPED)
    assert 300 < lane.length < 1300  # real pit lanes: ~400 m (Baku) to ~1.2 km (Lusail, Interlagos with their roads)
    assert lane.entry_s < 0 < lane.exit_s  # leaves the track before the line, joins it after
    assert np.all(np.diff(lane.s) > -3.0)  # runs in the race direction (far from the track, s may wobble a metre)
    assert lane.min_radius() > 6.0  # no kinks: a road a car can drive at pit lane speed (Singapore's tightest: ~8 m)
    # it runs beside the track on its side, off the tarmac, and joins it at both ends
    beyond = np.array([_beyond_edge(track, lane, n) for n in range(len(lane.s))])
    assert np.median(beyond) > 1.0 and np.all(np.abs(lane.offset) < 110.0)
    beside = np.abs(lane.s) < 100  # alongside the start line it is on its side (Yas Marina's exit crosses under the track)
    assert np.all(np.sign(lane.offset[beside]) == lane.side)
    assert beyond[0] < 5.0 and beyond[-1] < 5.0
    # away from its ends it stays off our tarmac (a wall between), but where it crosses (Yas Marina's tunnel)
    own = np.where(lane.offset > 0, np.abs(edge_offset(track, 1, lane.s)), np.abs(edge_offset(track, -1, lane.s)))
    assert np.mean((np.abs(lane.offset) - own)[50:-50] < 0) < 0.03


def test_named_tracks_share_the_calendar_pit_lane():
    a, b = get_pit_lane("monza"), get_pit_lane("f1:monza")
    assert a.source == "osm" and np.array_equal(a.x, b.x)


def test_a_track_without_a_mapped_pit_lane_gets_one_beside_the_start_straight():
    track = _get_track("random:3")
    lane = get_pit_lane("random:3", track)
    assert lane.source == "estimated"
    assert lane.entry_s < 0 < lane.exit_s and 500 < lane.length < 700
    turning = np.sum(np.diff(np.unwrap(track.heading)))
    assert lane.side == (1 if turning > 0 else -1)  # on the inside of the lap
    assert abs(_beyond_edge(track, lane, 0)) < 1.0 and abs(_beyond_edge(track, lane, -1)) < 1.0
    mid = len(lane.s) // 2
    assert _beyond_edge(track, lane, mid) == pytest.approx(15.0, abs=1.0)


def test_a_pit_lane_file_reads_back_the_same(tmp_path):
    track = _get_track("random:1")
    lane = estimated(track, "random:1")
    lane.save(tmp_path / "x.csv")
    back = load(tmp_path / "x.csv", track, "random:1")
    assert back.side == lane.side and back.source == "estimated"
    assert np.allclose(back.s, lane.s, atol=0.01) and np.allclose(back.x, lane.x, atol=0.01)
