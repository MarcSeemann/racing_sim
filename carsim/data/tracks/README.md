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

## Elevation (`elevation/<key>.csv`: s_m, z_m)

Height above sea level along the lap for every calendar circuit (and Monza), built by
`tools/build_elevation.py`: each centreline is fitted onto its f1-circuits GeoJSON trace (lon/lat),
the ground height is looked up every 10 m from OpenTopoData (EU-DEM 25 m in Europe, SRTM 30 m
elsewhere), then smoothed (running median 150 m, Gaussian 30 m). Terrain models measure the
surface, so expect a few metres of error, more under trees. The mean height also sets the air
pressure at the circuit (`Track.altitude_m`; Mexico City: ~24 % lower air density).

Where the terrain model plainly sees buildings, stands or a tunnel instead of the road (Singapore,
Monaco, Las Vegas, Mexico City), the circuit is kept flat at its mean height; each file's header
says which. EU-DEM is © European Union, Copernicus Land Monitoring Service; SRTM is public domain
(NASA/USGS).

The heights of the circuits Formula1.com publishes a height range for ("Highs and lows - which F1
track has the most elevation changes?", 2016) are scaled about their mean to that range when the
terrain model's range is more than 10 % off (`carsim.circuits.PUBLISHED_RANGE_M`): terrain models
see trees and stands (Melbourne came out at 9.6 m against 2.6 m published, Monza 17.4 against 12.8).

## Run-off surfaces and barriers (`runoff/<key>.npz`: arrays `left`, `right`)

What lies beyond the white line, metre by metre out to 60 m, at every centreline sample, on both
sides: codes 0 asphalt run-off, 1 grass, 2 gravel / sand, 3 barrier (and beyond). Built by
`tools/build_runoff.py` from OpenStreetMap (Overpass API): gravel traps (`natural=sand`,
`surface=gravel|sand`), grass (`landuse=grass|meadow`, `surface=grass`, ...), paved areas, and
barrier lines (`barrier=wall|guard_rail|fence|...`). Each centreline is fitted onto its f1-circuits
trace and then onto OSM's own raceway lines (median misfit about 1 m where they are mapped). A wide
unmapped band right beside the track is taken as paved run-off; samples where nothing is mapped keep
the estimate of `carsim/runoff.py`. Mapping quality varies by circuit (see `tools/build_runoff.py
--check` and notebook 12): read the files as "what OSM's mappers drew", not as a survey.

The files are a derived database of OpenStreetMap data, © OpenStreetMap contributors, available
under the Open Database License 1.0 (https://opendatacommons.org/licenses/odbl/1-0/). They are NOT
covered by this repository's own licence: they remain under the ODbL, and whoever receives them may
use, copy and adapt them under its terms (attribution, share-alike for adapted databases).

## Pit lanes (`pitlane/<key>.csv`: columns `s_m`, `offset_m`, `x_m`, `y_m`)

Each real circuit's pit lane, from where it leaves the track to where it joins it again, a point
every metre: `s_m` along our lap beside it (below 0: before the start line), `offset_m` from our
centreline (+ left), and the same points in the track frame. The second header line holds the side,
entry and exit as JSON. Built by `tools/build_pitlanes.py` from the OpenStreetMap answers that
`tools/build_runoff.py` keeps: `highway=raceway` ways named or tagged as the pit lane (with their
entry and exit roads), joined the way OSM connects them, placed with the same fit as the run-off.
Shanghai's pit lane is not named in OSM: there, the unnamed raceway lines beside the start straight.
Melbourne (Albert Park, public roads) has no raceway mapped: `carsim/pitlane.py` estimates one there,
as for every generated track. Nothing drives in them yet (meant for races: pit stops).

Like the run-off files, these are a derived database of OpenStreetMap data, © OpenStreetMap
contributors, under the Open Database License 1.0 (same terms as above).
