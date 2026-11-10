"""Drive the car yourself — real-time top-down view (roadmap phase 11, live input).

    pip install pygame
    python play.py                      # keyboard, short circuit, dry
    python play.py --track hill --weather shower --compound intermediate
    python play.py --demo               # watch the AI driver on the racing line
    python play.py --record lap.csv     # save your inputs; replay them on another setup later

Keyboard: arrows or WASD steer / throttle / brake, E or Shift = upshift, Q or Ctrl = downshift,
R = DRS, Esc = quit. A gamepad (if connected) works too: left stick steers, right/left
triggers are throttle/brake, A/B shift.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from carsim import Car2D, CarParams, race_car  # noqa: E402
from carsim.conditions import Conditions, Weather  # noqa: E402
from carsim.drivers import LapDriver  # noqa: E402
from carsim.compare import LiveDelta  # noqa: E402
from carsim.inputs import DeviceState, HumanDriver, Recorder  # noqa: E402
from carsim.track import hill_circuit, short_circuit  # noqa: E402
from carsim.track import test_circuit as long_circuit  # noqa: E402

DT = 0.001
TRACKS = {"short": short_circuit, "test": long_circuit, "hill": hill_circuit}
WEATHER = {"dry": lambda: Weather.dry(), "shower": lambda: Weather.shower(start=60, peak=180, end=360),
           "wet": lambda: Weather.constant(rain_mm_h=6.0, cloud=1.0, air_temp_c=16.0)}


class KeyboardPad:
    """Reads pygame keyboard / gamepad state into a DeviceState each frame."""

    def __init__(self, pygame):
        self.pg = pygame
        self.state = DeviceState()
        self.joy = None
        if pygame.joystick.get_count():
            self.joy = pygame.joystick.Joystick(0)
            self.joy.init()

    def poll(self):
        pg, k = self.pg, self.pg.key.get_pressed()
        st = DeviceState()
        st.steer = (1.0 if (k[pg.K_LEFT] or k[pg.K_a]) else 0.0) - (1.0 if (k[pg.K_RIGHT] or k[pg.K_d]) else 0.0)
        st.throttle = 1.0 if (k[pg.K_UP] or k[pg.K_w]) else 0.0
        st.brake = 1.0 if (k[pg.K_DOWN] or k[pg.K_s]) else 0.0
        st.shift_up = bool(k[pg.K_e] or k[pg.K_LSHIFT])
        st.shift_down = bool(k[pg.K_q] or k[pg.K_LCTRL])
        st.drs = bool(k[pg.K_r])
        if self.joy is not None:
            j = self.joy
            if abs(j.get_axis(0)) > 0.02:
                st.steer = -j.get_axis(0)
            if j.get_numaxes() > 5:
                st.throttle = max(st.throttle, (j.get_axis(5) + 1) / 2)
                st.brake = max(st.brake, (j.get_axis(4) + 1) / 2)
            st.shift_up |= bool(j.get_button(0))
            st.shift_down |= bool(j.get_button(1))
        self.state = st

    def __call__(self, t):
        return self.state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=TRACKS, default="short")
    ap.add_argument("--weather", choices=WEATHER, default="dry")
    ap.add_argument("--compound", default="slick", choices=("slick", "intermediate", "wet"))
    ap.add_argument("--fuel", type=float, default=30.0)
    ap.add_argument("--demo", action="store_true", help="AI drives the racing line")
    ap.add_argument("--record", help="save driver inputs to this CSV")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until Esc)")
    ap.add_argument("--screenshot", help="save the last frame to this PNG")
    a = ap.parse_args(argv)

    import pygame
    pygame.init()
    W, H = 1100, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("carsim")
    font = pygame.font.SysFont("dejavusans,arial", 16)
    big = pygame.font.SysFont("dejavusans,arial", 34, bold=True)

    track = TRACKS[a.track]()
    cond = Conditions(track, WEATHER[a.weather](), cars_on_track=20, rubber=2e6)
    car = Car2D(race_car(a.fuel), conditions=cond, compound=a.compound)
    car.set_speed(25.0)
    lap_driver = LapDriver(track, line="racing")  # also does lap timing in manual mode
    pad = KeyboardPad(pygame)
    driver = lap_driver if a.demo else HumanDriver(pad)
    if a.record:
        driver = Recorder(driver)

    # Track edges for drawing.
    half = track.width / 2
    left = [(x - half * math.sin(h), y + half * math.cos(h)) for x, y, h in zip(track.x, track.y, track.heading)]
    right = [(x + half * math.sin(h), y - half * math.cos(h)) for x, y, h in zip(track.x, track.y, track.heading)]
    line = list(zip(lap_driver.path.x, lap_driver.path.y))
    scale = 5.0  # px per m
    live = LiveDelta(track)
    gap = None
    # Minimap: whole track scaled into a box in the bottom-right corner.
    mx0, mx1, my0, my1 = min(track.x), max(track.x), min(track.y), max(track.y)
    mk = min(220 / (mx1 - mx0), 160 / (my1 - my0))
    mini = lambda x, y: (W - 240 + (x - mx0) * mk, H - 20 - (y - my0) * mk)
    mini_pts = [mini(x, y) for x, y in zip(track.x[::5], track.y[::5])]

    clock, frames, last_wall = pygame.time.Clock(), 0, time.perf_counter()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
        pad.poll()
        now = time.perf_counter()
        steps = min(int((now - last_wall) / DT), 50) if not a.frames else 17
        last_wall = now
        s = car.state
        for _ in range(max(steps, 1)):
            inp = driver(s, car)
            if not a.demo:
                lap_driver.i = track.nearest(s.x, s.y, lap_driver.i)  # keep lap timing in step
                s_now = track.s[lap_driver.i]
                if s_now < lap_driver._last_s - track.length / 2:
                    lap_driver.lap += 1
                    lap_driver.lap_times.append(s.t - lap_driver.lap_start)
                    lap_driver.lap_start = s.t
                lap_driver._last_s = s_now
            row = car.step(inp, DT)
        gap = live.update(car.cond.i, s.t)

        # ---- draw ------------------------------------------------------------
        cx, cy = s.x, s.y
        to_px = lambda p: (W / 2 + (p[0] - cx) * scale, H / 2 - (p[1] - cy) * scale)
        screen.fill((34, 74, 40))
        pygame.draw.polygon(screen, (70, 70, 74), [to_px(p) for p in left + right[::-1]])
        pygame.draw.lines(screen, (230, 230, 230), True, [to_px(p) for p in left], 2)
        pygame.draw.lines(screen, (230, 230, 230), True, [to_px(p) for p in right], 2)
        pygame.draw.lines(screen, (120, 160, 220), True, [to_px(p) for p in line], 1)
        L, Wd = 5.5, 2.0
        corners = [(L / 2, Wd / 2), (L / 2, -Wd / 2), (-L / 2, -Wd / 2), (-L / 2, Wd / 2)]
        c, sn = math.cos(s.psi), math.sin(s.psi)
        pygame.draw.polygon(screen, (230, 60, 40), [to_px((cx + px * c - py * sn, cy + px * sn + py * c)) for px, py in corners])

        # Minimap with the car's position.
        pygame.draw.rect(screen, (24, 50, 28), (W - 250, H - 200, 240, 190), border_radius=8)
        pygame.draw.lines(screen, (200, 200, 200), True, mini_pts, 3)
        pygame.draw.circle(screen, (230, 60, 40), [int(v) for v in mini(s.x, s.y)], 5)
        if gap is not None:
            col = (90, 220, 120) if gap < 0 else (240, 110, 90)
            screen.blit(big.render(f"{gap:+.2f}", True, col), (W - 170, 18))
            screen.blit(font.render(f"vs best {live.best_time:.2f} s", True, (240, 240, 240)), (W - 170, 60))

        hud = [f"{row['v'] * 3.6:5.0f} km/h   gear {row['gear']}   {row['rpm']:5.0f} rpm",
               f"lap {lap_driver.lap + 1}   last {lap_driver.lap_times[-1]:.2f} s" if lap_driver.lap_times else f"lap {lap_driver.lap + 1}",
               f"tyres {row['tsurf_fl']:.0f}/{row['tsurf_fr']:.0f}/{row['tsurf_rl']:.0f}/{row['tsurf_rr']:.0f} C   {a.compound}",
               f"fuel {row['fuel']:.1f} kg   track {row['track_temp']:.0f} C   rain {row['rain']:.1f} mm/h   water {row['water']:.2f} mm",
               "AI driving the racing line" if a.demo else "arrows/WASD drive, E/Q shift, R DRS, Esc quit"]
        screen.blit(big.render(f"{row['v'] * 3.6:.0f}", True, (255, 255, 255)), (24, 18))
        for i, txt in enumerate(hud):
            screen.blit(font.render(txt, True, (240, 240, 240)), (24, 70 + 22 * i))
        pygame.display.flip()
        frames += 1
        if a.frames and frames >= a.frames:
            running = False
        clock.tick(60)

    if a.screenshot:
        pygame.image.save(screen, a.screenshot)
    if a.record:
        driver.save(a.record)
        print(f"saved {len(driver.rows)} input samples to {a.record}")
    pygame.quit()
    return lap_driver.lap_times


if __name__ == "__main__":
    main()
