from __future__ import annotations

import math
import random
import sys
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

LOCAL_DEPS = Path(__file__).with_name(".deps")
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np

from physics_core import (
    Body,
    Vec2,
    clamp,
    crosses_vertical_barrier,
    gravitational_potential,
    kinetic_energy,
    reflect_velocity,
    resolve_circle_collision,
    spring_force,
    spring_potential,
)


BG = "#0f172a"
PANEL = "#111827"
CANVAS_BG = "#eaf2f8"
GRID = "#cbd5e1"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
DARK_TEXT = "#0f172a"
TEAL = "#0f766e"
ORANGE = "#f97316"
BLUE = "#2563eb"
RED = "#dc2626"
GOLD = "#d4a017"
PURPLE = "#7c3aed"

WIDTH = 1320
HEIGHT = 820
CANVAS_W = 860
CANVAS_H = 760


def fmt(value: float) -> str:
    return f"{value:,.2f}"


class BaseMode:
    name = "Mode"
    description = ""
    formulas: tuple[str, ...] = ()
    hint = "Click in the canvas to interact with this lab."

    def __init__(self, app: "PhysicsApp") -> None:
        self.app = app
        self.info_lines: list[str] = []
        self.controls: list[tuple[str, float, float, float, float]] = []
        self.on_setup()

    def on_setup(self) -> None:
        pass

    def on_control_change(self, values: dict[str, float]) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def draw(self, canvas: tk.Canvas) -> None:
        pass

    def on_click(self, x: float, y: float) -> None:
        pass


class VectorMode(BaseMode):
    name = "Vector Lab"
    description = "Steering, projection, normalization, and force accumulation in one place."
    formulas = (
        "desired_velocity = normalize(target - position) * max_speed",
        "steering = desired_velocity - current_velocity",
        "acceleration = steering_force / mass",
        "projection = dot(v, n) * n",
    )
    hint = "Click anywhere to move the target. Watch the steering vector chase it."

    def on_setup(self) -> None:
        self.body = Body(Vec2(180, 330), Vec2(60, -40), mass=1.0, radius=16, restitution=0.92)
        self.target = Vec2(620, 260)
        self.controls = [
            ("Max Speed", 50, 360, 200, 1),
            ("Steering Gain", 0.5, 8.0, 3.0, 0.1),
            ("Damping", 0.0, 4.0, 0.6, 0.1),
            ("Mass", 0.5, 8.0, 1.0, 0.1),
        ]
        self.max_speed = 200.0
        self.steering_gain = 3.0
        self.damping = 0.6

    def on_control_change(self, values: dict[str, float]) -> None:
        self.max_speed = values["Max Speed"]
        self.steering_gain = values["Steering Gain"]
        self.damping = values["Damping"]
        self.body.mass = values["Mass"]

    def update(self, dt: float) -> None:
        to_target = self.target - self.body.position
        desired_velocity = to_target.normalized() * self.max_speed
        steering = (desired_velocity - self.body.velocity) * self.steering_gain
        damping_force = self.body.velocity * (-self.damping)
        total_force = steering + damping_force
        self.body.apply_force(total_force, dt)
        self.body.integrate(dt)

        if self.body.position.x < self.body.radius or self.body.position.x > CANVAS_W - self.body.radius:
            self.body.velocity = reflect_velocity(
                self.body.velocity, Vec2(1 if self.body.position.x < self.body.radius else -1, 0), 0.8
            )
            self.body.position.x = clamp(self.body.position.x, self.body.radius, CANVAS_W - self.body.radius)
        if self.body.position.y < self.body.radius or self.body.position.y > CANVAS_H - self.body.radius:
            self.body.velocity = reflect_velocity(
                self.body.velocity, Vec2(0, 1 if self.body.position.y < self.body.radius else -1), 0.8
            )
            self.body.position.y = clamp(self.body.position.y, self.body.radius, CANVAS_H - self.body.radius)

        target_dir = to_target.normalized()
        forward_projection = self.body.velocity.dot(target_dir)
        lateral = self.body.velocity - target_dir * forward_projection
        self.info_lines = [
            f"Position: ({fmt(self.body.position.x)}, {fmt(self.body.position.y)})",
            f"Velocity: ({fmt(self.body.velocity.x)}, {fmt(self.body.velocity.y)})",
            f"Steering force: ({fmt(total_force.x)}, {fmt(total_force.y)})",
            f"Speed toward target: {fmt(forward_projection)}",
            f"Lateral drift magnitude: {fmt(lateral.magnitude())}",
            "Game feel lives here: steering gain and damping shape how responsive motion feels.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        draw_target(canvas, self.target)
        draw_arrow(canvas, self.body.position, self.body.position + self.body.velocity * 0.45, BLUE, "velocity")
        to_target = self.target - self.body.position
        desired = to_target.normalized() * self.max_speed
        draw_arrow(canvas, self.body.position, self.body.position + desired * 0.32, ORANGE, "desired")
        canvas.create_oval(
            self.body.position.x - self.body.radius,
            self.body.position.y - self.body.radius,
            self.body.position.x + self.body.radius,
            self.body.position.y + self.body.radius,
            fill=TEAL,
            outline="",
        )
        canvas.create_line(self.body.position.x, self.body.position.y, self.target.x, self.target.y, fill=MUTED, dash=(5, 5))

    def on_click(self, x: float, y: float) -> None:
        self.target = Vec2(x, y)


class BallisticsMode(BaseMode):
    name = "Ballistics Lab"
    description = "Projectile motion with wind, drag, and an analytic no-drag reference path."
    formulas = (
        "vx = cos(theta) * speed",
        "vy = -sin(theta) * speed",
        "drag_force = -drag * velocity",
        "position_ideal = p0 + v0 * t + 0.5 * a * t^2",
    )
    hint = "Click to move the launcher. Changing sliders relaunches the projectile."

    def on_setup(self) -> None:
        self.controls = [
            ("Launch Speed", 60, 520, 240, 1),
            ("Launch Angle", 5, 85, 43, 1),
            ("Gravity", 50, 700, 280, 1),
            ("Drag", 0.0, 1.4, 0.06, 0.01),
            ("Wind", -200, 200, 0, 1),
        ]
        self.origin = Vec2(90, CANVAS_H - 70)
        self.body = Body(self.origin.copy(), Vec2(), mass=1.0, radius=12, restitution=0.42)
        self.elapsed = 0.0
        self.path: list[Vec2] = []
        self.ideal_path: list[Vec2] = []
        self.last_values: dict[str, float] = {}

    def relaunch(self, values: dict[str, float]) -> None:
        self.last_values = dict(values)
        speed = values["Launch Speed"]
        angle = math.radians(values["Launch Angle"])
        self.gravity = values["Gravity"]
        self.drag = values["Drag"]
        self.wind = values["Wind"]
        self.body.position = self.origin.copy()
        self.body.velocity = Vec2(math.cos(angle) * speed, -math.sin(angle) * speed)
        self.initial_velocity = self.body.velocity.copy()
        self.elapsed = 0.0
        self.path = [self.body.position.copy()]
        self.ideal_path = [self.body.position.copy()]

    def on_control_change(self, values: dict[str, float]) -> None:
        self.relaunch(values)

    def update(self, dt: float) -> None:
        self.elapsed += dt
        gravity_force = Vec2(0, self.gravity * self.body.mass)
        drag_force = self.body.velocity * (-self.drag)
        wind_force = Vec2(self.wind, 0)
        self.body.apply_force(gravity_force + drag_force + wind_force, dt)
        self.body.integrate(dt)

        self.path.append(self.body.position.copy())
        if len(self.path) > 300:
            self.path.pop(0)

        ideal = Vec2(
            self.origin.x + self.initial_velocity.x * self.elapsed,
            self.origin.y + self.initial_velocity.y * self.elapsed + 0.5 * self.gravity * self.elapsed * self.elapsed,
        )
        self.ideal_path.append(ideal)
        if len(self.ideal_path) > 300:
            self.ideal_path.pop(0)

        ground = CANVAS_H - 28
        if self.body.position.y > ground:
            self.body.position.y = ground
            self.body.velocity.y *= -self.body.restitution
            self.body.velocity.x *= 0.94
            if abs(self.body.velocity.y) < 9:
                self.body.velocity.y = 0

        error = (self.body.position - ideal).magnitude()
        self.info_lines = [
            f"Time: {fmt(self.elapsed)} s",
            f"Position: ({fmt(self.body.position.x)}, {fmt(self.body.position.y)})",
            f"Velocity: ({fmt(self.body.velocity.x)}, {fmt(self.body.velocity.y)})",
            f"No-drag reference error: {fmt(error)} px",
            f"Wind force: {fmt(self.wind)}   Drag coeff: {fmt(self.drag)}",
            "Set drag and wind to 0 to see how closely the discrete simulation matches the closed-form path.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        canvas.create_rectangle(0, CANVAS_H - 18, CANVAS_W, CANVAS_H, fill="#94a3b8", outline="")
        draw_path(canvas, self.ideal_path, GOLD, dashed=True)
        draw_path(canvas, self.path, ORANGE)
        draw_arrow(canvas, self.body.position, self.body.position + self.body.velocity * 0.28, BLUE, "velocity")
        draw_arrow(canvas, self.body.position, self.body.position + Vec2(self.wind * 0.2, self.gravity * 0.14), RED, "net a")
        canvas.create_oval(
            self.body.position.x - self.body.radius,
            self.body.position.y - self.body.radius,
            self.body.position.x + self.body.radius,
            self.body.position.y + self.body.radius,
            fill=TEAL,
            outline="",
        )
        canvas.create_polygon(
            self.origin.x - 16,
            self.origin.y + 8,
            self.origin.x - 16,
            self.origin.y - 28,
            self.origin.x + 10,
            self.origin.y - 10,
            fill="#475569",
            outline="",
        )

    def on_click(self, x: float, y: float) -> None:
        self.origin = Vec2(x, clamp(y, 70, CANVAS_H - 60))
        self.relaunch(self.last_values or self.app.current_control_values())


class CollisionMode(BaseMode):
    name = "Collision Lab"
    description = "A box of moving bodies with gravity, restitution, and impulse-based collisions."
    formulas = (
        "momentum = mass * velocity",
        "impulse = change in momentum",
        "restitution controls energy kept after impact",
        "discrete collisions can miss if dt is too large",
    )
    hint = "Click to blast nearby bodies outward with an impulse."

    def on_setup(self) -> None:
        self.controls = [
            ("Ball Count", 3, 9, 5, 1),
            ("Spawn Speed", 40, 240, 120, 1),
            ("Gravity", 0, 450, 120, 1),
            ("Restitution", 0.2, 1.0, 0.92, 0.01),
        ]
        self.bodies: list[Body] = []
        self.last_values: dict[str, float] = {}

    def respawn(self, values: dict[str, float]) -> None:
        self.last_values = dict(values)
        rng = random.Random(7)
        count = int(round(values["Ball Count"]))
        spawn_speed = values["Spawn Speed"]
        restitution = values["Restitution"]
        self.gravity = values["Gravity"]
        self.bodies = []
        for index in range(count):
            angle = (math.tau / count) * index
            radius = rng.uniform(16, 30)
            mass = radius * 0.14
            position = Vec2(
                CANVAS_W / 2 + math.cos(angle) * rng.uniform(80, 180),
                CANVAS_H / 2 + math.sin(angle) * rng.uniform(60, 130),
            )
            velocity = Vec2.from_angle_radians(angle + math.pi / 2, spawn_speed * rng.uniform(0.65, 1.2))
            self.bodies.append(Body(position, velocity, mass=mass, radius=radius, restitution=restitution))

    def on_control_change(self, values: dict[str, float]) -> None:
        should_respawn = not self.last_values or any(
            abs(values[name] - self.last_values.get(name, values[name])) > 1e-6 for name in ("Ball Count", "Spawn Speed")
        )
        self.gravity = values["Gravity"]
        if should_respawn:
            self.respawn(values)
        else:
            self.last_values = dict(values)
            for body in self.bodies:
                body.restitution = values["Restitution"]

    def update(self, dt: float) -> None:
        restitution = self.last_values.get("Restitution", 0.92)
        for body in self.bodies:
            body.restitution = restitution
            body.apply_force(Vec2(0, self.gravity * body.mass), dt)
            body.integrate(dt)
            if body.position.x < body.radius or body.position.x > CANVAS_W - body.radius:
                body.velocity = reflect_velocity(
                    body.velocity, Vec2(1 if body.position.x < body.radius else -1, 0), body.restitution
                )
                body.position.x = clamp(body.position.x, body.radius, CANVAS_W - body.radius)
            if body.position.y < body.radius or body.position.y > CANVAS_H - body.radius:
                body.velocity = reflect_velocity(
                    body.velocity, Vec2(0, 1 if body.position.y < body.radius else -1), body.restitution
                )
                body.position.y = clamp(body.position.y, body.radius, CANVAS_H - body.radius)

        for index in range(len(self.bodies)):
            for other in range(index + 1, len(self.bodies)):
                resolve_circle_collision(self.bodies[index], self.bodies[other])

        total_momentum = Vec2()
        total_ke = 0.0
        for body in self.bodies:
            total_momentum += body.velocity * body.mass
            total_ke += kinetic_energy(body)
        self.info_lines = [
            f"Body count: {len(self.bodies)}",
            f"Total momentum: ({fmt(total_momentum.x)}, {fmt(total_momentum.y)})",
            f"Total kinetic energy: {fmt(total_ke)}",
            f"Gravity: {fmt(self.gravity)}   Restitution: {fmt(restitution)}",
            "Low substeps and high speed can still create collision artifacts. That is a real engine problem, not just a demo problem.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        colors = [TEAL, ORANGE, BLUE, PURPLE, RED, "#14b8a6", "#eab308", "#2563eb", "#ef4444"]
        for index, body in enumerate(self.bodies):
            color = colors[index % len(colors)]
            canvas.create_oval(
                body.position.x - body.radius,
                body.position.y - body.radius,
                body.position.x + body.radius,
                body.position.y + body.radius,
                fill=color,
                outline="",
            )
            draw_arrow(canvas, body.position, body.position + body.velocity * 0.18, "#1e293b", "v")

    def on_click(self, x: float, y: float) -> None:
        blast_center = Vec2(x, y)
        for body in self.bodies:
            offset = body.position - blast_center
            distance = max(offset.magnitude(), 20)
            body.velocity += offset.normalized() * (2600 / distance)


class IntegratorMode(BaseMode):
    name = "Integrator Lab"
    description = "Three identical spring systems solved with different integrators to show stability and energy drift."
    formulas = (
        "Explicit Euler: x += v * dt, then v += a * dt",
        "Semi-Implicit Euler: v += a * dt, then x += v * dt",
        "Velocity Verlet: x += v * dt + 0.5 * a * dt^2",
        "Stiff systems punish large dt values",
    )
    hint = "Raise time scale or stiffness and watch the methods diverge."

    def on_setup(self) -> None:
        self.controls = [
            ("Stiffness", 0.5, 18.0, 6.0, 0.1),
            ("Damping", 0.0, 3.0, 0.1, 0.1),
            ("Mass", 0.5, 8.0, 1.0, 0.1),
            ("Rest Length", 80, 280, 180, 1),
            ("Amplitude", 10, 180, 100, 1),
        ]
        self.methods: list[dict[str, object]] = []
        self.energy_history: dict[str, list[float]] = {}
        self.last_values: dict[str, float] = {}

    def reset_systems(self, values: dict[str, float]) -> None:
        self.last_values = dict(values)
        self.stiffness = values["Stiffness"]
        self.damping = values["Damping"]
        self.mass = values["Mass"]
        self.rest_length = values["Rest Length"]
        amplitude = values["Amplitude"]
        anchors = [Vec2(120, 170), Vec2(120, 370), Vec2(120, 570)]
        labels = [("Explicit Euler", RED), ("Semi-Implicit", TEAL), ("Velocity Verlet", BLUE)]
        self.methods = []
        self.energy_history = {}
        for anchor, (label, color) in zip(anchors, labels):
            position = anchor.x + self.rest_length + amplitude
            velocity = 0.0
            extension = position - anchor.x - self.rest_length
            base_energy = 0.5 * self.mass * velocity * velocity + spring_potential(extension, self.stiffness)
            self.methods.append(
                {
                    "label": label,
                    "color": color,
                    "anchor": anchor,
                    "x": position,
                    "v": velocity,
                    "a": self.acceleration(position, velocity, anchor.x),
                    "base_energy": max(base_energy, 1e-6),
                }
            )
            self.energy_history[label] = []

    def acceleration(self, x: float, v: float, anchor_x: float) -> float:
        displacement = x - anchor_x
        extension = displacement - self.rest_length
        spring_accel = (-self.stiffness * extension) / self.mass
        damping_accel = (-self.damping * v) / self.mass
        return spring_accel + damping_accel

    def on_control_change(self, values: dict[str, float]) -> None:
        self.reset_systems(values)

    def update(self, dt: float) -> None:
        summaries: list[str] = []
        for state in self.methods:
            anchor_x = state["anchor"].x
            x = float(state["x"])
            v = float(state["v"])
            if state["label"] == "Explicit Euler":
                a = self.acceleration(x, v, anchor_x)
                x += v * dt
                v += a * dt
            elif state["label"] == "Semi-Implicit":
                a = self.acceleration(x, v, anchor_x)
                v += a * dt
                x += v * dt
            else:
                a0 = self.acceleration(x, v, anchor_x)
                x += v * dt + 0.5 * a0 * dt * dt
                a1 = self.acceleration(x, v + a0 * dt, anchor_x)
                v += 0.5 * (a0 + a1) * dt

            state["x"] = x
            state["v"] = v
            state["a"] = self.acceleration(x, v, anchor_x)
            extension = x - anchor_x - self.rest_length
            energy = 0.5 * self.mass * v * v + spring_potential(extension, self.stiffness)
            rel_error = (energy - float(state["base_energy"])) / float(state["base_energy"])
            history = self.energy_history[str(state["label"])]
            history.append(rel_error)
            if len(history) > 240:
                history.pop(0)
            summaries.append(f"{state['label']}: x={fmt(x)} v={fmt(v)} energy error={fmt(rel_error * 100)}%")

        self.info_lines = summaries + [
            f"dt per substep is controlled globally. Higher stiffness + larger dt is where many games run into instability.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        for state in self.methods:
            anchor = state["anchor"]
            position = Vec2(float(state["x"]), anchor.y)
            anchor_end = Vec2(anchor.x, anchor.y)
            draw_spring(canvas, anchor_end, position)
            canvas.create_oval(anchor.x - 8, anchor.y - 8, anchor.x + 8, anchor.y + 8, fill="#1d4ed8", outline="")
            canvas.create_oval(position.x - 18, position.y - 18, position.x + 18, position.y + 18, fill=state["color"], outline="")
            draw_arrow(canvas, position, position + Vec2(float(state["v"]) * 0.18, 0), "#1e293b", "v")
            canvas.create_text(120, anchor.y - 36, text=state["label"], fill=DARK_TEXT, anchor="w", font=("Segoe UI", 11, "bold"))
        draw_graph(
            canvas,
            x=520,
            y=80,
            width=290,
            height=220,
            title="Relative Energy Error",
            series=[(label, history, color) for label, history, color in (
                ("Explicit Euler", self.energy_history.get("Explicit Euler", []), RED),
                ("Semi-Implicit", self.energy_history.get("Semi-Implicit", []), TEAL),
                ("Velocity Verlet", self.energy_history.get("Velocity Verlet", []), BLUE),
            )],
            y_scale=0.8,
        )


class NBodyMode(BaseMode):
    name = "N-Body Lab"
    description = "Many-body gravity solved with numpy. Beautiful, expensive, and numerically sensitive."
    formulas = (
        "F = G * m1 * m2 / r^2",
        "Acceleration is the vector sum of every pairwise interaction",
        "This version is O(n^2), which becomes expensive as body count grows",
    )
    hint = "Click to reseed the swarm around a new center."

    def on_setup(self) -> None:
        self.controls = [
            ("Body Count", 6, 24, 12, 1),
            ("Gravity Constant", 3000, 28000, 12000, 100),
            ("Swirl Speed", 20, 220, 95, 1),
            ("Softening", 8, 48, 18, 1),
        ]
        self.center = np.array([CANVAS_W * 0.5, CANVAS_H * 0.5], dtype=np.float64)
        self.positions = np.zeros((0, 2), dtype=np.float64)
        self.velocities = np.zeros((0, 2), dtype=np.float64)
        self.masses = np.zeros(0, dtype=np.float64)
        self.trails: list[list[tuple[float, float]]] = []
        self.last_values: dict[str, float] = {}

    def reseed(self, values: dict[str, float]) -> None:
        self.last_values = dict(values)
        rng = np.random.default_rng(11)
        count = int(round(values["Body Count"]))
        self.gravity_constant = values["Gravity Constant"]
        self.softening = values["Softening"]
        swirl_speed = values["Swirl Speed"]

        positions = [self.center.copy()]
        velocities = [np.array([0.0, 0.0])]
        masses = [900.0]
        for _ in range(count - 1):
            angle = rng.uniform(0, math.tau)
            radius = rng.uniform(90.0, 260.0)
            position = self.center + np.array([math.cos(angle), math.sin(angle)]) * radius
            tangent = np.array([-math.sin(angle), math.cos(angle)])
            velocity = tangent * swirl_speed * rng.uniform(0.7, 1.45)
            positions.append(position)
            velocities.append(velocity)
            masses.append(rng.uniform(1.0, 8.0))

        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)
        self.masses = np.array(masses, dtype=np.float64)
        total_momentum = (self.velocities * self.masses[:, None]).sum(axis=0)
        self.velocities[0] = -total_momentum / self.masses[0]
        self.trails = [[] for _ in range(min(6, len(self.positions)))]

    def on_control_change(self, values: dict[str, float]) -> None:
        if not self.last_values or any(
            abs(values[name] - self.last_values.get(name, values[name])) > 1e-6
            for name in ("Body Count", "Swirl Speed", "Softening", "Gravity Constant")
        ):
            self.reseed(values)

    def update(self, dt: float) -> None:
        offsets = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        distance_sq = np.sum(offsets * offsets, axis=2) + self.softening * self.softening
        np.fill_diagonal(distance_sq, np.inf)
        inv_distance_cubed = distance_sq ** -1.5
        accelerations = self.gravity_constant * np.sum(
            offsets * self.masses[np.newaxis, :, None] * inv_distance_cubed[:, :, None],
            axis=1,
        )
        self.velocities += accelerations * dt
        self.positions += self.velocities * dt

        for index, trail in enumerate(self.trails):
            trail.append((float(self.positions[index, 0]), float(self.positions[index, 1])))
            if len(trail) > 180:
                trail.pop(0)

        speed = np.linalg.norm(self.velocities, axis=1)
        total_ke = 0.5 * np.sum(self.masses * speed * speed)
        pair_distance = np.sqrt(distance_sq)
        potential_matrix = -self.gravity_constant * (self.masses[:, None] * self.masses[None, :]) / pair_distance
        potential_matrix[np.isinf(potential_matrix)] = 0.0
        total_pe = float(np.sum(np.triu(potential_matrix, 1)))
        energy = total_ke + total_pe
        bounds_fraction = np.count_nonzero(
            (self.positions[:, 0] < -120)
            | (self.positions[:, 0] > CANVAS_W + 120)
            | (self.positions[:, 1] < -120)
            | (self.positions[:, 1] > CANVAS_H + 120)
        ) / max(len(self.positions), 1)

        self.info_lines = [
            f"Bodies: {len(self.positions)}  Pair checks per step: {len(self.positions) * (len(self.positions) - 1) // 2}",
            f"Total kinetic energy: {fmt(total_ke)}",
            f"Total potential energy: {fmt(total_pe)}",
            f"Total mechanical energy: {fmt(energy)}",
            f"Bodies outside safe view: {fmt(bounds_fraction * 100)}%",
            "This is where numerical softness, timestep choice, and computational cost all start fighting each other.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        for trail, color in zip(self.trails, [TEAL, ORANGE, BLUE, PURPLE, RED, GOLD]):
            draw_tuple_path(canvas, trail, color)
        for index, position in enumerate(self.positions):
            radius = 24 if index == 0 else clamp(self.masses[index] * 1.7, 4, 12)
            color = TEAL if index == 0 else ORANGE
            canvas.create_oval(
                position[0] - radius,
                position[1] - radius,
                position[0] + radius,
                position[1] + radius,
                fill=color,
                outline="",
            )
        canvas.create_text(18, CANVAS_H - 24, anchor="w", text="Powered by numpy pairwise gravity", fill=DARK_TEXT, font=("Segoe UI", 10, "bold"))

    def on_click(self, x: float, y: float) -> None:
        self.center = np.array([x, y], dtype=np.float64)
        self.reseed(self.last_values or self.app.current_control_values())


class LimitsMode(BaseMode):
    name = "Limits Lab"
    description = "A direct look at tunneling: one projectile uses naive discrete collision, the other uses a swept test."
    formulas = (
        "Large dt means objects move farther between checks",
        "If an object jumps past a thin wall, overlap tests never fire",
        "Swept tests or more substeps reduce tunneling",
    )
    hint = "Increase speed or simulation speed and compare the top lane to the bottom lane."

    def on_setup(self) -> None:
        self.controls = [
            ("Bullet Speed", 120, 4200, 1800, 10),
            ("Wall X", 300, 760, 520, 1),
            ("Wall Thickness", 4, 26, 10, 1),
            ("Bullet Radius", 4, 18, 8, 1),
        ]
        self.last_values: dict[str, float] = {}
        self.reset_demo({"Bullet Speed": 1800, "Wall X": 520, "Wall Thickness": 10, "Bullet Radius": 8})

    def reset_demo(self, values: dict[str, float]) -> None:
        self.last_values = dict(values)
        self.speed = values["Bullet Speed"]
        self.wall_x = values["Wall X"]
        self.wall_thickness = values["Wall Thickness"]
        self.bullet_radius = values["Bullet Radius"]
        self.top = {"pos": 80.0, "prev": 80.0, "hit": False, "escaped": False}
        self.bottom = {"pos": 80.0, "prev": 80.0, "hit": False, "escaped": False}

    def on_control_change(self, values: dict[str, float]) -> None:
        self.reset_demo(values)

    def update_lane(self, lane: dict[str, float | bool], dt: float, use_swept_test: bool) -> None:
        if lane["hit"] or lane["escaped"]:
            return
        lane["prev"] = lane["pos"]
        lane["pos"] += self.speed * dt
        wall_left = self.wall_x - self.wall_thickness * 0.5
        wall_right = self.wall_x + self.wall_thickness * 0.5

        if use_swept_test:
            crossed_left = crosses_vertical_barrier(float(lane["prev"]), float(lane["pos"]), wall_left, self.bullet_radius)
            crossed_right = crosses_vertical_barrier(float(lane["prev"]), float(lane["pos"]), wall_right, self.bullet_radius)
            if crossed_left or crossed_right or (
                float(lane["pos"]) + self.bullet_radius >= wall_left and float(lane["pos"]) - self.bullet_radius <= wall_right
            ):
                lane["pos"] = wall_left - self.bullet_radius
                lane["hit"] = True
        else:
            if float(lane["pos"]) + self.bullet_radius >= wall_left and float(lane["pos"]) - self.bullet_radius <= wall_right:
                lane["pos"] = wall_left - self.bullet_radius
                lane["hit"] = True

        if float(lane["pos"]) > CANVAS_W + 60:
            lane["escaped"] = True

    def update(self, dt: float) -> None:
        self.update_lane(self.top, dt, use_swept_test=False)
        self.update_lane(self.bottom, dt, use_swept_test=True)
        frame_distance = self.speed * dt
        self.info_lines = [
            f"Distance moved per substep: {fmt(frame_distance)} px",
            f"Naive lane: {'stopped at wall' if self.top['hit'] else 'tunneled through' if self.top['escaped'] else 'in flight'}",
            f"Swept lane: {'stopped at wall' if self.bottom['hit'] else 'escaped' if self.bottom['escaped'] else 'in flight'}",
            f"Wall thickness: {fmt(self.wall_thickness)}   Bullet diameter: {fmt(self.bullet_radius * 2)}",
            "This is one of the classic limits of real-time game physics. More substeps help, but robust collision methods matter too.",
        ]

    def draw_lane(self, canvas: tk.Canvas, y: float, lane: dict[str, float | bool], label: str, color: str) -> None:
        wall_left = self.wall_x - self.wall_thickness * 0.5
        wall_right = self.wall_x + self.wall_thickness * 0.5
        canvas.create_text(36, y - 42, text=label, anchor="w", fill=DARK_TEXT, font=("Segoe UI", 11, "bold"))
        canvas.create_line(30, y, CANVAS_W - 30, y, fill="#94a3b8", width=2)
        canvas.create_rectangle(wall_left, y - 34, wall_right, y + 34, fill="#475569", outline="")
        x = float(lane["pos"])
        canvas.create_oval(x - self.bullet_radius, y - self.bullet_radius, x + self.bullet_radius, y + self.bullet_radius, fill=color, outline="")
        if lane["escaped"]:
            canvas.create_text(CANVAS_W - 150, y - 42, text="escaped past wall", fill=RED, font=("Segoe UI", 11, "bold"))

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        self.draw_lane(canvas, 240, self.top, "Naive discrete overlap check", ORANGE)
        self.draw_lane(canvas, 480, self.bottom, "Swept / continuous wall crossing check", TEAL)
        canvas.create_text(
            40,
            680,
            anchor="w",
            text="Tip: set a high global time scale and low substeps to exaggerate the failure mode.",
            fill=DARK_TEXT,
            font=("Segoe UI", 10, "bold"),
        )


def draw_grid(canvas: tk.Canvas) -> None:
    for x in range(0, CANVAS_W, 40):
        canvas.create_line(x, 0, x, CANVAS_H, fill=GRID)
    for y in range(0, CANVAS_H, 40):
        canvas.create_line(0, y, CANVAS_W, y, fill=GRID)


def draw_target(canvas: tk.Canvas, target: Vec2) -> None:
    canvas.create_oval(target.x - 10, target.y - 10, target.x + 10, target.y + 10, outline=ORANGE, width=2)
    canvas.create_line(target.x - 15, target.y, target.x + 15, target.y, fill=ORANGE, width=2)
    canvas.create_line(target.x, target.y - 15, target.x, target.y + 15, fill=ORANGE, width=2)


def draw_arrow(canvas: tk.Canvas, start: Vec2, end: Vec2, color: str, label: str) -> None:
    canvas.create_line(start.x, start.y, end.x, end.y, fill=color, width=3, arrow=tk.LAST, arrowshape=(12, 14, 5))
    canvas.create_text(end.x + 18, end.y, text=label, fill=color, font=("Segoe UI", 10, "bold"))


def draw_spring(canvas: tk.Canvas, start: Vec2, end: Vec2) -> None:
    direction = end - start
    length = max(direction.magnitude(), 1)
    unit = direction / length
    normal = unit.perpendicular()
    points = [start]
    for step in range(1, 13):
        t = step / 13
        offset = math.sin(t * math.pi * 8) * 12
        points.append(start + direction * t + normal * offset)
    points.append(end)
    flat: list[float] = []
    for point in points:
        flat.extend((point.x, point.y))
    canvas.create_line(*flat, fill="#334155", width=3, smooth=True)


def draw_path(canvas: tk.Canvas, points: list[Vec2], color: str, dashed: bool = False) -> None:
    if len(points) < 2:
        return
    flat: list[float] = []
    for point in points:
        flat.extend((point.x, point.y))
    canvas.create_line(*flat, fill=color, width=2, smooth=True, dash=(4, 4) if dashed else ())


def draw_tuple_path(canvas: tk.Canvas, points: list[tuple[float, float]], color: str) -> None:
    if len(points) < 2:
        return
    flat: list[float] = []
    for x, y in points:
        flat.extend((x, y))
    canvas.create_line(*flat, fill=color, width=1.6)


def draw_graph(
    canvas: tk.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    series: list[tuple[str, list[float], str]],
    y_scale: float,
) -> None:
    canvas.create_rectangle(x, y, x + width, y + height, fill="#f8fafc", outline="#94a3b8", width=2)
    canvas.create_text(x + 12, y + 14, text=title, anchor="w", fill=DARK_TEXT, font=("Segoe UI", 11, "bold"))
    mid_y = y + height / 2
    canvas.create_line(x + 10, mid_y, x + width - 10, mid_y, fill="#cbd5e1", dash=(4, 4))
    for index, (label, values, color) in enumerate(series):
        canvas.create_text(x + 12, y + 34 + index * 18, text=label, anchor="w", fill=color, font=("Segoe UI", 9, "bold"))
        if len(values) < 2:
            continue
        points: list[float] = []
        for i, value in enumerate(values):
            px = x + 8 + (i / max(len(values) - 1, 1)) * (width - 16)
            py = mid_y - clamp(value / y_scale, -1.0, 1.0) * (height * 0.36)
            points.extend((px, py))
        canvas.create_line(*points, fill=color, width=2, smooth=True)


class PhysicsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Advanced Physics Simulation Playground")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")
        self.root.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 13, "bold"))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

        self.mode_classes = [VectorMode, BallisticsMode, CollisionMode, IntegratorMode, NBodyMode, LimitsMode]
        self.control_vars: dict[str, tk.DoubleVar] = {}
        self.speed_var = tk.DoubleVar(value=1.0)
        self.substeps_var = tk.DoubleVar(value=2.0)
        self.paused = False
        self.last_time = time.perf_counter()

        self.build_layout()
        self.switch_mode(VectorMode)
        self.root.bind("<space>", lambda _event: self.toggle_pause())
        self.tick()

    def build_layout(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top, text="Advanced Physics Simulation Playground", style="Title.TLabel").pack(side=tk.LEFT)

        controls = ttk.Frame(top)
        controls.pack(side=tk.RIGHT)

        ttk.Button(controls, text="Pause / Resume", command=self.toggle_pause).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reset Lab", command=self.reset_current_mode).pack(side=tk.LEFT, padx=4)
        tk.Label(controls, text="Speed", bg=BG, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        tk.Scale(
            controls,
            from_=0.0,
            to=4.0,
            resolution=0.1,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            length=150,
            bg=BG,
            fg=TEXT,
            troughcolor="#243042",
            activebackground=TEAL,
            highlightthickness=0,
        ).pack(side=tk.LEFT)
        tk.Label(controls, text="Substeps", bg=BG, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        tk.Scale(
            controls,
            from_=1,
            to=12,
            resolution=1,
            variable=self.substeps_var,
            orient=tk.HORIZONTAL,
            length=120,
            bg=BG,
            fg=TEXT,
            troughcolor="#243042",
            activebackground=ORANGE,
            highlightthickness=0,
        ).pack(side=tk.LEFT)

        button_row = ttk.Frame(outer)
        button_row.pack(fill=tk.X, pady=(0, 10))
        for mode_cls in self.mode_classes:
            ttk.Button(button_row, text=mode_cls.name.replace(" Lab", ""), command=lambda cls=mode_cls: self.switch_mode(cls)).pack(
                side=tk.LEFT, padx=4
            )

        content = ttk.Frame(outer)
        content.pack(fill=tk.BOTH, expand=True)

        canvas_panel = ttk.Frame(content, style="Panel.TFrame")
        canvas_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.canvas = tk.Canvas(canvas_panel, width=CANVAS_W, height=CANVAS_H, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(padx=(0, 14))
        self.canvas.bind("<Button-1>", self.handle_click)

        self.side_panel = ttk.Frame(content, style="Panel.TFrame")
        self.side_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.mode_label = ttk.Label(self.side_panel, text="", style="Title.TLabel")
        self.mode_label.pack(anchor="w", padx=14, pady=(14, 4))
        self.description_label = ttk.Label(self.side_panel, text="", wraplength=380, style="Muted.TLabel")
        self.description_label.pack(anchor="w", padx=14, pady=(0, 10))
        self.status_label = ttk.Label(self.side_panel, text="", wraplength=380)
        self.status_label.pack(anchor="w", padx=14, pady=(0, 10))
        self.controls_frame = ttk.Frame(self.side_panel, style="Panel.TFrame")
        self.controls_frame.pack(fill=tk.X, padx=14, pady=(0, 10))
        self.info_label = ttk.Label(self.side_panel, text="", justify=tk.LEFT, wraplength=380)
        self.info_label.pack(anchor="w", padx=14, pady=(0, 10))
        self.formula_label = ttk.Label(self.side_panel, text="", justify=tk.LEFT, wraplength=380, style="Muted.TLabel")
        self.formula_label.pack(anchor="w", padx=14, pady=(0, 10))

    def current_control_values(self) -> dict[str, float]:
        return {name: var.get() for name, var in self.control_vars.items()}

    def switch_mode(self, mode_cls: type[BaseMode]) -> None:
        self.current_mode = mode_cls(self)
        self.render_controls()
        self.refresh_side_panel()

    def reset_current_mode(self) -> None:
        mode_cls = type(self.current_mode)
        self.switch_mode(mode_cls)

    def render_controls(self) -> None:
        for child in self.controls_frame.winfo_children():
            child.destroy()
        self.control_vars.clear()

        for name, minimum, maximum, default, resolution in self.current_mode.controls:
            ttk.Label(self.controls_frame, text=name).pack(anchor="w")
            variable = tk.DoubleVar(value=default)
            scale = tk.Scale(
                self.controls_frame,
                from_=minimum,
                to=maximum,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                bg=PANEL,
                fg=TEXT,
                troughcolor="#243042",
                activebackground=TEAL,
                highlightthickness=0,
                font=("Segoe UI", 9),
                variable=variable,
                command=lambda _value, n=name: self.on_control_change(n),
            )
            scale.pack(fill=tk.X, pady=(0, 8))
            self.control_vars[name] = variable

        self.apply_control_values()

    def on_control_change(self, _name: str) -> None:
        self.apply_control_values()

    def apply_control_values(self) -> None:
        self.current_mode.on_control_change(self.current_control_values())

    def refresh_side_panel(self) -> None:
        self.mode_label.configure(text=self.current_mode.name)
        self.description_label.configure(text=self.current_mode.description)

    def handle_click(self, event: tk.Event) -> None:
        self.current_mode.on_click(event.x, event.y)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def tick(self) -> None:
        now = time.perf_counter()
        frame_dt = min(now - self.last_time, 0.05)
        self.last_time = now

        if not self.paused:
            sim_dt = frame_dt * self.speed_var.get()
            substeps = max(1, int(round(self.substeps_var.get())))
            step_dt = sim_dt / substeps if sim_dt > 0 else 0.0
            for _ in range(substeps):
                self.current_mode.update(step_dt)

        self.draw(frame_dt)
        self.root.after(16, self.tick)

    def draw(self, frame_dt: float) -> None:
        self.canvas.delete("all")
        self.current_mode.draw(self.canvas)
        self.info_label.configure(text="\n".join(self.current_mode.info_lines))
        self.formula_label.configure(text="Core formulas:\n" + "\n".join(f"- {line}" for line in self.current_mode.formulas))
        fps = 0.0 if frame_dt <= 0 else 1.0 / frame_dt
        status = "Paused" if self.paused else "Running"
        self.status_label.configure(
            text=f"Status: {status}\nTime Scale: {fmt(self.speed_var.get())}x\nSubsteps: {int(round(self.substeps_var.get()))}\nFrame Rate: {fmt(fps)} fps"
        )
        self.canvas.create_text(12, 12, anchor="nw", text=self.current_mode.hint, fill=DARK_TEXT, font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(
            12,
            36,
            anchor="nw",
            text=f"Space toggles pause. Speed and substeps are global controls.",
            fill=DARK_TEXT,
            font=("Segoe UI", 9),
        )


def main() -> None:
    root = tk.Tk()
    PhysicsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
