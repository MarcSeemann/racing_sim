# Circuit data

`monza.csv` — Autodromo Nazionale Monza (5.79 km, run clockwise), from the TUM racetrack
database: https://github.com/TUMFTM/racetrack-database (file `tracks/Monza.csv`, unchanged).
Columns: centreline x, y (m) and the distances from the centreline to the right and left
track edges (m). The centreline comes from OpenStreetMap GPS points (smoothed); the widths
were extracted from satellite images, so expect errors of the order of a metre. No elevation.

The data is licensed under the GNU LGPL v3 (see `LICENSE-TUMFTM`); it is redistributed here
unmodified. It is NOT covered by this repository's own (proprietary) licence: these data files
remain under the LGPL v3, whoever receives them may use, copy and modify them under its terms.

## The 2026 F1 calendar (`carsim/f1_calendar.py`, held-out test set)

Built by `tools/build_f1_2026.py` from two pinned sources:

- TUM racetrack database, commit `e59595d` (LGPL-3.0, `LICENSE-TUMFTM`): surveyed centreline and
  widths, layouts of about 2019-2020.
- bacinger/f1-circuits, commit `a78e292` (MIT, `LICENSE-f1-circuits`): centrelines traced on maps,
  current layouts, no widths.

| Files | Source |
| --- | --- |
| `shanghai`, `suzuka`, `bahrain`, `montreal`, `spielberg`, `silverstone`, `spa`, `hungaroring`, `austin`, `mexico`, `interlagos`, `sepang` (+ `monza.csv`) | TUM files copied unchanged (`Shanghai.csv`, ..., `SaoPaulo.csv`, `Sepang.csv`); each was checked to match the current layout within a few metres |
| `melbourne`, `barcelona`, `zandvoort`, `yas_marina` | modified TUM data (LGPL-3.0): the survey where the layout is unchanged (82-98 % of the lap), the f1-circuits trace where the circuit was rebuilt after the survey (Melbourne 2021, Barcelona 2023 final chicane, Zandvoort 2020 banking, Yas Marina 2021), widths from the survey (its median on the rebuilt parts) |
| `jeddah`, `miami`, `monaco`, `madring`, `baku`, `singapore`, `las_vegas`, `lusail` | f1-circuits trace only (MIT): splined, smoothed (2 m), constant nominal width 12 m (Monaco 9 m). Expect the corners to be less exact than surveyed ones: on the circuits both sources have, a traced version's estimated lap time differed from the surveyed one by about 3 % (no bias) |

Each file says in its header which source it came from. Coordinates are metres east/north of the
circuit's centre (TUM files: TUM's own frame). No elevation for any of them.
