# tidepool

A toy box for [Jumper](https://github.com/KingKongRobotics/jumper), the open-source
crab robot. Write a few lines of YAML, get a `.map` you can drop the crab into.

```yaml
name: spiral-dominoes
look: studio
spawn: {at: [-0.6, 0], yaw: 0}
toys:
  - domino_run: {path: spiral, count: 50, start: [0.35, 0], height: 0.1}
```

```
tidepool build spiral.yaml    # -> out/spiral-dominoes/spiral-dominoes.map, verified
tidepool play spiral.yaml     # opens it in MuJoCo with the shipped app driving the crab
```

The `.map` is a plain kk-scene-package, so it uploads to beunlimited.me as is. Built for
the Crab Robot Open-source Challenge 2026. Everything runs in simulation.

## Setup

Clone the two official repositories next to this one, then install into one venv:

```
git clone https://github.com/KingKongRobotics/jumper
git clone https://github.com/KingKongRobotics/jumper-design
git clone https://github.com/BeBeBecon/tidepool
cd tidepool
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ../jumper -e "../jumper-design[sim]" -e .
```

jumper-design keeps its skins in git LFS; run `git lfs pull` there once. On macOS the
viewer needs `mjpython`, which the venv gets from MuJoCo. Set `TIDEPOOL_JUMPER` or
`TIDEPOOL_JUMPER_DESIGN` if the clones live somewhere else.

## Toys

`tidepool toys` lists them. So far:

- `domino_run`: dominoes along a `line`, `arc`, `spiral` or `zigzag`.

More on the way. A toy is one function in `tidepool/toys/` that returns mjscene objects;
see `domino_run.py` for the shape of it.

## Playing

In the viewer the crab runs the stock app from jumper: `W`/`S` walk, `A`/`D` sidestep,
`J`/`L` turn, Space jumps, `1`-`4` are gestures, Ctrl+`1`-`4` dances, `V`/`B` the claws.
Walk into the first domino.

## License

Apache-2.0. Jumper and jumper-design are copyright KingKong Robotics, Apache-2.0; this
project uses them as installed packages and ships nothing of theirs.
