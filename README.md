# Advanced Physics Simulation Playground

An interactive Python project for learning the physics that matters in games:

- vector math and steering
- projectile motion
- rigid-body style collisions
- spring systems and integration methods
- many-body gravity
- the practical limits of real-time simulation

This version is more advanced than the original:

- global simulation speed control
- configurable physics substeps
- a dedicated integrator lab
- a dedicated limits lab for tunneling
- a `numpy`-powered N-body gravity sandbox

## Requirements

- Python 3.14+
- `numpy`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Labs

### Vector Lab
Shows steering behavior, desired velocity, damping, projection, and lateral drift.

### Ballistics Lab
Shows gravity, launch angle, drag, wind, and a no-drag analytic reference path.

### Collision Lab
Shows momentum, restitution, gravity, and why discrete collision handling needs care.

### Integrator Lab
Runs the same spring system with Explicit Euler, Semi-Implicit Euler, and Velocity Verlet so you can compare stability and energy drift.

### N-Body Lab
Uses `numpy` to simulate multiple gravitating bodies at once. This is intentionally more expensive and helps show performance and stability tradeoffs.

### Limits Lab
Directly demonstrates tunneling: a fast-moving object can pass through a thin wall if your collision detection only checks overlap after motion.

## Global Controls

- `Pause / Resume`: stop and restart the simulation
- `Reset Lab`: reset the active lab
- `Speed`: slow the world down or speed it up
- `Substeps`: split each frame into smaller physics updates
- `Space`: keyboard shortcut for pause/resume

## What It Teaches

This project is aimed at helping you understand both the useful physics and the failure cases:

- why timestep size matters
- why some integrators are more stable than others
- why fast objects tunnel through thin geometry
- why many-body gravity gets expensive quickly
- why game physics is always a balance between realism, stability, and performance

## Files

- `main.py`: desktop app, labs, rendering, controls
- `physics_core.py`: vector math and shared physics helpers
- `requirements.txt`: Python dependency list
