"""Real-time server for the Godot 3D front end: messages, pacing, inputs, reset."""
import json
import socket
import threading
import time

import pytest

from carsim.server import Session, serve


@pytest.fixture(scope="module")
def session():
    return Session(track="short", weather="dry", compound="slick", fuel=10.0, ai=False)


def test_track_message_has_matching_arrays(session):
    msg = session.track_message()
    n = len(msg["x"])
    assert msg["type"] == "track" and n > 1000
    assert len(msg["y"]) == len(msg["z"]) == len(msg["heading"]) == len(msg["line_x"]) == n
    json.dumps(msg)  # serialisable


def test_state_message_after_stepping(session):
    session.device.throttle = 1.0
    session.step(500)
    st = session.state_message()
    assert st["type"] == "state" and st["t"] == pytest.approx(0.5, abs=1e-6)
    assert len(st["wheels"]) == 4 and len(st["tyres"]) == 4
    json.dumps(st)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_client_session_over_tcp():
    port = _free_port()
    sess = Session(track="short", fuel=10.0, ai=False)
    th = threading.Thread(target=serve, args=(sess,), kwargs={"port": port, "once": True, "max_seconds": 20},
                          daemon=True)
    th.start()
    for _ in range(100):
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
            break
        except OSError:
            time.sleep(0.1)
    f = sock.makefile("rwb")
    assert json.loads(f.readline())["type"] == "track"
    send = lambda d: (sock.sendall((json.dumps(d) + "\n").encode()))
    send({"type": "input", "throttle": 1.0, "steer": 0.0})
    t0 = time.perf_counter()
    states = []
    while time.perf_counter() - t0 < 2.0:
        states.append(json.loads(f.readline()))
    rate = len(states) / 2.0
    sim_elapsed = states[-1]["t"] - states[0]["t"]
    assert 40 < rate < 80  # ~60 Hz
    assert sim_elapsed == pytest.approx(2.0, abs=0.4)  # physics paced to the wall clock
    assert states[-1]["v"] > states[0]["v"]  # throttle input reached the car
    send({"type": "reset"})
    time.sleep(0.2)
    after = json.loads(f.readline())
    for _ in range(3):
        after = json.loads(f.readline())
    assert after["v"] < 15.0  # back on track, (almost) at rest
    send({"type": "quit"})
    th.join(timeout=5)
    sock.close()
