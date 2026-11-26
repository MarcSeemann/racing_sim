"""Every calendar circuit starts (s = 0) on its real start/finish line, which runs alongside the pit lane."""
import numpy as np
import pytest

from carsim.circuits import get_track
from carsim.f1_calendar import CALENDAR
from carsim.pitlane import get_pit_lane

SLACK_M = 100.0  # the line may lie a little beyond the mapped pit lane's ends


@pytest.mark.parametrize("c", CALENDAR, ids=lambda c: c.key)
def test_the_start_is_alongside_the_pit_lane(c):
    lane = get_pit_lane("f1:" + c.key)
    if lane.source == "estimated":
        pytest.skip("no pit lane mapped: the estimate is laid out around the start line, it proves nothing")
    assert lane.entry_s - SLACK_M <= 0.0 <= lane.exit_s + SLACK_M, \
        f"start outside the pit lane ({lane.entry_s:.0f}..{lane.exit_s:.0f} m)"


def _corners(tr, max_radius):
    """(s at the tightest point, radius, 'left'/'right') of every corner tighter than max_radius."""
    h = np.unwrap(np.arctan2(np.gradient(tr.y), np.gradient(tr.x)))
    k = np.gradient(h) / np.gradient(tr.s)
    tight = np.abs(k) > 1 / max_radius
    out, i = [], 0
    while i < len(k):
        if tight[i]:
            j = i
            while j < len(k) and tight[j]:
                j += 1
            m = i + int(np.argmax(np.abs(k[i:j])))
            out.append((float(tr.s[m]), 1 / abs(k[m]), "left" if k[m] > 0 else "right"))
            i = j
        else:
            i += 1
    return out


def test_monaco_starts_on_the_pit_straight():
    """The line is ~150 m before Sainte Devote, the hairpin ~1.2 km into the lap (it was 650 m too far on, at Casino)."""
    tr = get_track("f1:monaco")
    first = _corners(tr, 50.0)[0]
    assert 100 < first[0] < 250 and first[2] == "right" and first[1] < 15  # Sainte Devote
    hairpin = min((c for c in _corners(tr, 50.0) if c[0] < 1500), key=lambda c: c[1])
    assert 1100 < hairpin[0] < 1300 and hairpin[2] == "left"  # Grand Hotel hairpin
