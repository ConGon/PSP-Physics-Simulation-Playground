# Physics Simulation Playground

An interactive Python desktop app for learning the core physics ideas game developers use:

- vector math
- forces and acceleration
- gravity and projectile motion
- collisions and momentum
- springs and damping
- orbital attraction

The project is designed to be visual and educational. Each mode shows:

- the simulation itself
- the important values updating live
- the formulas behind the behavior
- controls to change parameters and immediately see the result

## Run

```bash
python main.py
```

No external packages are required. It uses the Python standard library with `tkinter`.

## Modes

### Vector Lab
Learn how position, velocity, acceleration, normalization, dot product, and scaling affect motion.

### Gravity Lab
See projectile motion with adjustable launch angle, speed, gravity, and drag.

### Collision Lab
Watch circular bodies collide and bounce using mass, restitution, and momentum.

### Spring Lab
Explore Hooke's law, damping, oscillation, and how springs stabilize motion.

### Orbit Lab
Experiment with Newton-style attraction and see how starting velocity changes the path.

## Controls

- Use the mode buttons at the top to switch simulations.
- Use the sliders on the right to change parameters.
- Click `Reset` to restore the current mode.
- In some modes, click inside the canvas to reposition or retarget the simulation.

## Suggested learning path

1. Start with `Vector Lab`.
2. Move to `Gravity Lab`.
3. Try `Collision Lab`.
4. Explore `Spring Lab`.
5. Finish with `Orbit Lab`.

## Files

- `main.py`: app entry point and UI
- `physics_core.py`: vector math and reusable simulation helpers
