# Car-only lap vs real 2025 pole (before calibration)

Run 2026-11-24 on next-gen with `python tools/calibrate.py compare --year 2025 --all` (2025-spec car, 10 kg fuel,
racing line, no driver margin). Real poles: Jolpica-F1. Las Vegas 2025 qualifying was probably wet (to check), Miami has no
qualifying times in the data.

## Medium wing

```
melbourne    lap real  75.096 (NOR) sim   99.94 (+33.1%)   top sim  338   apex mean sim   86 km/h
shanghai     lap real  90.641 (PIA) sim  109.85 (+21.2%)   top sim  355   apex mean sim   82 km/h
suzuka       lap real  86.983 (VER) sim  104.51 (+20.2%)   top sim  351   apex mean sim  111 km/h
bahrain      lap real  89.841 (PIA) sim  102.95 (+14.6%)   top sim  353   apex mean sim   71 km/h
jeddah       lap real  87.294 (VER) sim  105.13 (+20.4%)   top sim  348   apex mean sim   99 km/h
miami        no 2025 qualifying result
montreal     lap real  70.899 (RUS) sim   84.44 (+19.1%)   top sim  353   apex mean sim   60 km/h
monaco       lap real  69.954 (NOR) sim   88.23 (+26.1%)   top sim  313   apex mean sim   58 km/h
barcelona    lap real  71.546 (PIA) sim   89.89 (+25.6%)   top sim  354   apex mean sim   95 km/h
spielberg    lap real  63.971 (NOR) sim   75.36 (+17.8%)   top sim  342   apex mean sim   85 km/h
silverstone  lap real  84.892 (VER) sim  108.08 (+27.3%)   top sim  347   apex mean sim  119 km/h
spa          lap real 100.562 (NOR) sim  120.20 (+19.5%)   top sim  355   apex mean sim  131 km/h
hungaroring  lap real  75.372 (LEC) sim   94.98 (+26.0%)   top sim  343   apex mean sim   89 km/h
zandvoort    lap real  68.662 (PIA) sim   90.52 (+31.8%)   top sim  336   apex mean sim   92 km/h
monza        lap real  78.792 (VER) sim   91.11 (+15.6%)   top sim  359   apex mean sim  112 km/h
madring      no 2025 qualifying result
baku         lap real 101.117 (VER) sim  111.97 (+10.7%)   top sim  359   apex mean sim   87 km/h
singapore    lap real  89.158 (RUS) sim  105.78 (+18.6%)   top sim  335   apex mean sim   65 km/h
austin       lap real  92.510 (VER) sim  116.01 (+25.4%)   top sim  355   apex mean sim   83 km/h
mexico       lap real  75.586 (NOR) sim   91.82 (+21.5%)   top sim  381   apex mean sim   79 km/h
interlagos   lap real  69.511 (NOR) sim   83.17 (+19.6%)   top sim  357   apex mean sim  100 km/h
las_vegas    lap real 107.934 (NOR) sim  100.00 ( -7.3%)   top sim  368   apex mean sim  112 km/h
lusail       lap real  79.387 (PIA) sim   99.49 (+25.3%)   top sim  352   apex mean sim  123 km/h
yas_marina   lap real  82.207 (VER) sim  105.52 (+28.4%)   top sim  349   apex mean sim   82 km/h
gap          mean +20.9%  min -7.3%  max +33.1%  (22 circuits)
```

## Low wing

```
melbourne    lap real  75.096 (NOR) sim  101.53 (+35.2%)   top sim  352   apex mean sim   84 km/h
shanghai     lap real  90.641 (PIA) sim  111.21 (+22.7%)   top sim  377   apex mean sim   79 km/h
suzuka       lap real  86.983 (VER) sim  106.64 (+22.6%)   top sim  369   apex mean sim  108 km/h
bahrain      lap real  89.841 (PIA) sim  104.19 (+16.0%)   top sim  373   apex mean sim   70 km/h
jeddah       lap real  87.294 (VER) sim  106.36 (+21.8%)   top sim  368   apex mean sim   96 km/h
miami        no 2025 qualifying result
montreal     lap real  70.899 (RUS) sim   85.04 (+19.9%)   top sim  373   apex mean sim   59 km/h
monaco       lap real  69.954 (NOR) sim   89.44 (+27.9%)   top sim  319   apex mean sim   72 km/h
barcelona    lap real  71.546 (PIA) sim   91.67 (+28.1%)   top sim  375   apex mean sim   93 km/h
spielberg    lap real  63.971 (NOR) sim   76.11 (+19.0%)   top sim  355   apex mean sim   84 km/h
silverstone  lap real  84.892 (VER) sim  110.11 (+29.7%)   top sim  363   apex mean sim  115 km/h
spa          lap real 100.562 (NOR) sim  122.50 (+21.8%)   top sim  375   apex mean sim  126 km/h
hungaroring  lap real  75.372 (LEC) sim   96.60 (+28.2%)   top sim  358   apex mean sim   87 km/h
zandvoort    lap real  68.662 (PIA) sim   92.17 (+34.2%)   top sim  350   apex mean sim   89 km/h
monza        lap real  78.792 (VER) sim   91.50 (+16.1%)   top sim  382   apex mean sim  107 km/h
madring      no 2025 qualifying result
baku         lap real 101.117 (VER) sim  112.82 (+11.6%)   top sim  384   apex mean sim   84 km/h
singapore    lap real  89.158 (RUS) sim  107.07 (+20.1%)   top sim  349   apex mean sim   64 km/h
austin       lap real  92.510 (VER) sim  117.84 (+27.4%)   top sim  375   apex mean sim   81 km/h
mexico       lap real  75.586 (NOR) sim   92.62 (+22.5%)   top sim  400   apex mean sim   77 km/h
interlagos   lap real  69.511 (NOR) sim   84.68 (+21.8%)   top sim  373   apex mean sim   95 km/h
las_vegas    lap real 107.934 (NOR) sim  100.53 ( -6.9%)   top sim  392   apex mean sim  106 km/h
lusail       lap real  79.387 (PIA) sim  101.71 (+28.1%)   top sim  373   apex mean sim  118 km/h
yas_marina   lap real  82.207 (VER) sim  107.30 (+30.5%)   top sim  367   apex mean sim   80 km/h
gap          mean +22.7%  min -6.9%  max +35.2%  (22 circuits)
```
