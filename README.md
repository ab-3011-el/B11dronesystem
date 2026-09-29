# Drone Flight Training Simulator

A transforming drone-car trainer built with **Python + Ursina**. Learn quadcopter
control (throttle, pitch, roll, yaw), fly training rings, land precisely on pads,
then transform into a ground vehicle and drive.

## Run

```
pip install -r requirements.txt
python "main (1).py"
```

Python 3.10 - 3.12 recommended. The simulator uses Ursina built-ins and requires
no external asset files.

## Controls

| Key | Action |
|---|---|
| SPACE | Throttle up (car mode: brake) |
| LEFT SHIFT | Throttle down |
| W / S | Pitch forward / backward (car: accelerate / reverse) |
| A / D | Roll left / right (car: steer) |
| Q / E | Yaw left / right |
| T | Transform drone <-> car (must be landed / stopped) |
| C | Cycle camera: Chase, FPV, Orbit |
| R | Reset vehicle to PAD A |
| H | Toggle help |
| ESC | Exit |
| Mouse drag / wheel | Orbit camera rotate / zoom (Orbit mode) |

Throttle is **persistent** like a real transmitter stick: hold SPACE to raise it,
LEFT SHIFT to lower it. Thrust is `throttle * 2g`, so **50% throttle hovers**.
Tilt the drone to accelerate horizontally; tilt returns to level when keys are released.

## Missions (in order)

1. Take off and climb above 10 m
2. Fly through all 8 rings (any order)
3. Reach all 3 checkpoints (cyan spheres)
4. Land on PAD A
5. Transform into CAR mode
6. Drive along the road to the green beacon
7. Transform back into DRONE mode
8. Take off and land on PAD B (end of the road)

## Scoring

| Event | Points |
|---|---|
| Ring cleared | +100 |
| Checkpoint | +75 |
| Precision pad landing | up to +400 (distance from centre + softness) |
| Mission step complete | +150 (+500 for finishing all) |
| Hard landing (> 3 m/s) | -50 |
| Crash | -100 |
| Building collision | extra -75 |
| Obstacle / tree / pylon impact | extra -50 |

## Crash detection

Crash when: hitting a building, obstacle or tree while flying; touching down faster
than 4.5 m/s vertically or 8 m/s horizontally; landing tilted more than 22 degrees;
driving into a building or obstacle above 8 m/s. `CRASH DETECTED` is shown and the
vehicle resets to PAD A after 3 seconds (score and mission progress are kept).

## Build a Windows executable

```
build.bat          (windowed)
build.bat debug    (with console, to see errors)
```

Output: `dist\DroneFlightSim.exe`.

## Tuning

Constants at the top of `main (1).py`: `MAX_TILT`, `TRANSFORM_TIME`, `CRASH_*`.
Drag and thrust coefficients are in `Vehicle.fly`; car handling is in `Vehicle.drive`.

## Troubleshooting

- **Ground or objects look dark / no shadows:** your Ursina version may differ in
  shadow API. Shadow setup is wrapped in `try/except`, so the sim still runs.
  Upgrade with `pip install -U ursina`.
- **Low FPS:** lower the shadow map size in `World.build_environment`, reduce
  the tree or building counts in `build_trees` / `build_buildings`.
