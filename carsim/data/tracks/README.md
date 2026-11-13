# Circuit data

`monza.csv` — Autodromo Nazionale Monza (5.79 km, run clockwise), from the TUM racetrack
database: https://github.com/TUMFTM/racetrack-database (file `tracks/Monza.csv`, unchanged).
Columns: centreline x, y (m) and the distances from the centreline to the right and left
track edges (m). The centreline comes from OpenStreetMap GPS points (smoothed); the widths
were extracted from satellite images, so expect errors of the order of a metre. No elevation.

The data is licensed under the GNU LGPL v3 (see `LICENSE-TUMFTM`); it is redistributed here
unmodified. It is NOT covered by this repository's own (proprietary) licence: these data files
remain under the LGPL v3, whoever receives them may use, copy and modify them under its terms.
