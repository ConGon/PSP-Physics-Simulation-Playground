from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

from physics_core import Body, Vec2, reflect_velocity, resolve_circle_collision, spring_force


BG = "#0f172a"
PANEL = "#111827"
CANVAS_BG = "#e5eef7"
ACCENT = "#0f766e"
ACCENT_2 = "#f97316"
ACCENT_3 = "#2563eb"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
DARK_TEXT = "#0f172a"
GRID = "#cbd5e1"

WIDTH = 980
HEIGHT = 680
CANVAS_W = 660
CANVAS_H = 620


def fmt(num: float) -> str:
    return f"{num:,.2f}"


class BaseMode:
    name = "Mode"
    description = ""
    formulas = ()

    def __init__(self, app: "PhysicsApp") -> None:
        self.app = app
        self.info_lines: list[str] = []
        self.controls: list[tuple[str, float, float, float, float]] = []
        self.on_setup()

    def on_setup(self) -> None:
        pass

    def reset(self) -> None:
        self.on_setup()

    def update(self, dt: float) -> None:
        pass

    def draw(self, canvas: tk.Canvas) -> None:
        pass

    def on_click(self, x: float, y: float) -> None:
        pass


class VectorMode(BaseMode):
    name = "Vector Lab"
    description = "Control velocity and acceleration vectors to see how motion is built from math."
    formulas = (
        "velocity = position change / time",
        "acceleration = force / mass",
        "next_velocity = velocity + acceleration * dt",
        "next_position = position + velocity * dt",
        "dot(a, b) = |a| |b| cos(theta)",
    )

    def on_setup(self) -> None:
        self.position = Vec2(180, 280)
        self.velocity = Vec2(140, -30)
        self.acceleration = Vec2(0, 0)
        self.target = Vec2(500, 240)
        self.controls = [
            ("Velocity X", -250, 250, self.velocity.x, 1),
            ("Velocity Y", -250, 250, self.velocity.y, 1),
            ("Accel X", -200, 200, self.acceleration.x, 1),
            ("Accel Y", -200, 200, self.acceleration.y, 1),
        ]

    def sync_from_controls(self, values: dict[str, float]) -> None:
        self.velocity = Vec2(values["Velocity X"], values["Velocity Y"])
        self.acceleration = Vec2(values["Accel X"], values["Accel Y"])

    def update(self, dt: float) -> None:
        self.velocity = self.velocity + self.acceleration * dt
        self.position = self.position + self.velocity * dt

        if self.position.x < 20 or self.position.x > CANVAS_W - 20:
            self.velocity = reflect_velocity(self.velocity, Vec2(1 if self.position.x < 20 else -1, 0), 0.9)
            self.position.x = min(max(self.position.x, 20), CANVAS_W - 20)
        if self.position.y < 20 or self.position.y > CANVAS_H - 20:
            self.velocity = reflect_velocity(self.velocity, Vec2(0, 1 if self.position.y < 20 else -1), 0.9)
            self.position.y = min(max(self.position.y, 20), CANVAS_H - 20)

        to_target = self.target - self.position
        dot = self.velocity.dot(to_target)
        self.info_lines = [
            f"Position: ({fmt(self.position.x)}, {fmt(self.position.y)})",
            f"Velocity: ({fmt(self.velocity.x)}, {fmt(self.velocity.y)})  |v| = {fmt(self.velocity.magnitude())}",
            f"Acceleration: ({fmt(self.acceleration.x)}, {fmt(self.acceleration.y)})",
            f"To target: ({fmt(to_target.x)}, {fmt(to_target.y)})",
            f"Dot(velocity, target) = {fmt(dot)}",
            "Positive dot means motion is mostly toward the target.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        draw_target(canvas, self.target)
        draw_arrow(canvas, self.position, self.position + self.velocity * 0.4, ACCENT_3, "velocity")
        draw_arrow(canvas, self.position, self.position + self.acceleration * 0.8, ACCENT_2, "accel")
        canvas.create_oval(
            self.position.x - 15,
            self.position.y - 15,
            self.position.x + 15,
            self.position.y + 15,
            fill=ACCENT,
            outline="",
        )

    def on_click(self, x: float, y: float) -> None:
        self.target = Vec2(x, y)


class GravityMode(BaseMode):
    name = "Gravity Lab"
    description = "Launch a projectile and study gravity, launch angle, speed, and drag."
    formulas = (
        "vx = cos(theta) * speed",
        "vy = -sin(theta) * speed",
        "gravity adds downward acceleration every frame",
        "drag force is opposite velocity",
        "range depends on speed, angle, and gravity",
    )

    def on_setup(self) -> None:
        self.origin = Vec2(90, CANVAS_H - 70)
        self.body = Body(self.origin.copy(), Vec2(), mass=1.0, radius=12.0, restitution=0.65)
        self.path: list[Vec2] = []
        self.elapsed = 0.0
        self.controls = [
            ("Launch Speed", 50, 450, 220, 1),
            ("Launch Angle", 5, 85, 45, 1),
            ("Gravity", 50, 700, 280, 1),
            ("Drag", 0, 2.0, 0.1, 0.01),
        ]
        self.launch(self.app.current_control_values())

    def launch(self, values: dict[str, float]) -> None:
        speed = values.get("Launch Speed", 220)
        angle = math.radians(values.get("Launch Angle", 45))
        self.gravity = values.get("Gravity", 280)
        self.drag = values.get("Drag", 0.1)
        self.body.position = self.origin.copy()
        self.body.velocity = Vec2(math.cos(angle) * speed, -math.sin(angle) * speed)
        self.path = [self.body.position.copy()]
        self.elapsed = 0.0

    def sync_from_controls(self, values: dict[str, float]) -> None:
        self.gravity = values["Gravity"]
        self.drag = values["Drag"]

    def update(self, dt: float) -> None:
        self.elapsed += dt
        gravity_force = Vec2(0, self.gravity * self.body.mass)
        drag_force = self.body.velocity * (-self.drag)
        self.body.apply_force(gravity_force + drag_force, dt)
        self.body.integrate(dt)
        self.path.append(self.body.position.copy())
        if len(self.path) > 250:
            self.path.pop(0)

        if self.body.position.y >= CANVAS_H - 30:
            self.body.position.y = CANVAS_H - 30
            self.body.velocity.y *= -self.body.restitution
            self.body.velocity.x *= 0.92
            if abs(self.body.velocity.y) < 12:
                self.body.velocity.y = 0

        self.info_lines = [
            f"Time: {fmt(self.elapsed)} s",
            f"Position: ({fmt(self.body.position.x)}, {fmt(self.body.position.y)})",
            f"Velocity: ({fmt(self.body.velocity.x)}, {fmt(self.body.velocity.y)})",
            f"Speed: {fmt(self.body.velocity.magnitude())}",
            f"Gravity accel: {fmt(self.gravity)} px/s^2",
            f"Drag strength: {fmt(self.drag)}",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        canvas.create_rectangle(0, CANVAS_H - 20, CANVAS_W, CANVAS_H, fill="#94a3b8", outline="")
        if len(self.path) > 1:
            points = []
            for p in self.path:
                points.extend((p.x, p.y))
            canvas.create_line(*points, fill=ACCENT_2, width=2, smooth=True)
        draw_arrow(canvas, self.body.position, self.body.position + self.body.velocity * 0.3, ACCENT_3, "velocity")
        draw_arrow(canvas, self.body.position, self.body.position + Vec2(0, self.gravity * 0.12), "#dc2626", "gravity")
        canvas.create_oval(
            self.body.position.x - self.body.radius,
            self.body.position.y - self.body.radius,
            self.body.position.x + self.body.radius,
            self.body.position.y + self.body.radius,
            fill=ACCENT,
            outline="",
        )

    def on_click(self, x: float, y: float) -> None:
        self.origin = Vec2(x, min(y, CANVAS_H - 60))
        self.launch(self.app.current_control_values())


class CollisionMode(BaseMode):
    name = "Collision Lab"
    description = "See velocity exchange, momentum, mass, and restitution during bounces."
    formulas = (
        "momentum = mass * velocity",
        "relative_velocity determines collision response",
        "impulse changes velocity instantly",
        "restitution controls bounciness",
    )

    def on_setup(self) -> None:
        self.a = Body(Vec2(180, 260), Vec2(160, 20), mass=2.0, radius=24.0, restitution=0.92)
        self.b = Body(Vec2(470, 310), Vec2(-90, -30), mass=3.5, radius=34.0, restitution=0.92)
        self.controls = [
            ("Mass A", 0.5, 10.0, self.a.mass, 0.1),
            ("Mass B", 0.5, 10.0, self.b.mass, 0.1),
            ("Restitution", 0.1, 1.0, 0.92, 0.01),
        ]

    def sync_from_controls(self, values: dict[str, float]) -> None:
        self.a.mass = values["Mass A"]
        self.b.mass = values["Mass B"]
        self.a.restitution = values["Restitution"]
        self.b.restitution = values["Restitution"]

    def update(self, dt: float) -> None:
        for body in (self.a, self.b):
            body.integrate(dt)
            if body.position.x < body.radius or body.position.x > CANVAS_W - body.radius:
                normal = Vec2(1 if body.position.x < body.radius else -1, 0)
                body.velocity = reflect_velocity(body.velocity, normal, body.restitution)
                body.position.x = min(max(body.position.x, body.radius), CANVAS_W - body.radius)
            if body.position.y < body.radius or body.position.y > CANVAS_H - body.radius:
                normal = Vec2(0, 1 if body.position.y < body.radius else -1)
                body.velocity = reflect_velocity(body.velocity, normal, body.restitution)
                body.position.y = min(max(body.position.y, body.radius), CANVAS_H - body.radius)

        resolve_circle_collision(self.a, self.b)

        momentum_a = self.a.velocity * self.a.mass
        momentum_b = self.b.velocity * self.b.mass
        total = momentum_a + momentum_b
        self.info_lines = [
            f"Body A velocity: ({fmt(self.a.velocity.x)}, {fmt(self.a.velocity.y)})",
            f"Body B velocity: ({fmt(self.b.velocity.x)}, {fmt(self.b.velocity.y)})",
            f"Body A momentum: ({fmt(momentum_a.x)}, {fmt(momentum_a.y)})",
            f"Body B momentum: ({fmt(momentum_b.x)}, {fmt(momentum_b.y)})",
            f"Total momentum: ({fmt(total.x)}, {fmt(total.y)})",
            "Energy is not fully conserved because restitution can be less than 1.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        for body, color, label in ((self.a, ACCENT, "A"), (self.b, ACCENT_2, "B")):
            canvas.create_oval(
                body.position.x - body.radius,
                body.position.y - body.radius,
                body.position.x + body.radius,
                body.position.y + body.radius,
                fill=color,
                outline="",
            )
            canvas.create_text(body.position.x, body.position.y, text=label, fill="white", font=("Segoe UI", 16, "bold"))
            draw_arrow(canvas, body.position, body.position + body.velocity * 0.35, ACCENT_3, "v")


class SpringMode(BaseMode):
    name = "Spring Lab"
    description = "Explore Hooke's law, damping, overshoot, and stable spring motion."
    formulas = (
        "spring_force = -k * extension * direction",
        "damping_force = -velocity * damping",
        "net_force = spring_force + damping_force",
        "oscillation happens when energy swaps between motion and stretch",
    )

    def on_setup(self) -> None:
        self.anchor = Vec2(CANVAS_W / 2, 90)
        self.body = Body(Vec2(CANVAS_W / 2 + 160, 280), Vec2(), mass=1.0, radius=18.0, restitution=0.4)
        self.controls = [
            ("Stiffness", 0.5, 12.0, 3.5, 0.1),
            ("Damping", 0.0, 6.0, 1.0, 0.1),
            ("Rest Length", 40, 260, 150, 1),
            ("Mass", 0.5, 8.0, 1.0, 0.1),
        ]

    def sync_from_controls(self, values: dict[str, float]) -> None:
        self.stiffness = values["Stiffness"]
        self.damping = values["Damping"]
        self.rest_length = values["Rest Length"]
        self.body.mass = values["Mass"]

    def update(self, dt: float) -> None:
        spring = spring_force(self.body.position, self.anchor, self.rest_length, self.stiffness)
        damping = self.body.velocity * (-self.damping)
        gravity = Vec2(0, 120 * self.body.mass)
        net = spring + damping + gravity
        self.body.apply_force(net, dt)
        self.body.integrate(dt)

        if self.body.position.y > CANVAS_H - self.body.radius:
            self.body.position.y = CANVAS_H - self.body.radius
            self.body.velocity.y *= -0.3

        extension = (self.body.position - self.anchor).magnitude() - self.rest_length
        self.info_lines = [
            f"Mass position: ({fmt(self.body.position.x)}, {fmt(self.body.position.y)})",
            f"Velocity: ({fmt(self.body.velocity.x)}, {fmt(self.body.velocity.y)})",
            f"Extension: {fmt(extension)}",
            f"Spring force magnitude: {fmt(spring.magnitude())}",
            f"Damping force magnitude: {fmt(damping.magnitude())}",
            "Higher damping removes oscillation faster.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        canvas.create_line(self.anchor.x, 0, self.anchor.x, self.anchor.y, fill=MUTED, width=3)
        draw_spring(canvas, self.anchor, self.body.position)
        canvas.create_oval(self.anchor.x - 8, self.anchor.y - 8, self.anchor.x + 8, self.anchor.y + 8, fill="#1d4ed8", outline="")
        canvas.create_oval(
            self.body.position.x - self.body.radius,
            self.body.position.y - self.body.radius,
            self.body.position.x + self.body.radius,
            self.body.position.y + self.body.radius,
            fill=ACCENT_2,
            outline="",
        )
        draw_arrow(canvas, self.body.position, self.body.position + self.body.velocity * 0.35, ACCENT_3, "velocity")

    def on_click(self, x: float, y: float) -> None:
        self.body.position = Vec2(x, y)
        self.body.velocity = Vec2()


class OrbitMode(BaseMode):
    name = "Orbit Lab"
    description = "Use attraction and starting velocity to compare falling, escaping, and orbiting."
    formulas = (
        "force magnitude = G * m1 * m2 / r^2",
        "force direction points toward the attracting body",
        "sideways velocity can turn a fall into an orbit",
    )

    def on_setup(self) -> None:
        self.center = Vec2(CANVAS_W / 2, CANVAS_H / 2)
        self.body = Body(Vec2(self.center.x + 180, self.center.y), Vec2(0, -150), mass=1.0, radius=10.0, restitution=0.9)
        self.path: list[Vec2] = []
        self.controls = [
            ("Gravity Constant", 5000, 60000, 18000, 100),
            ("Start Speed", 20, 320, 150, 1),
            ("Start Distance", 80, 240, 180, 1),
        ]
        self.reset_orbit(self.app.current_control_values())

    def reset_orbit(self, values: dict[str, float]) -> None:
        self.gravity_constant = values.get("Gravity Constant", 18000)
        distance = values.get("Start Distance", 180)
        speed = values.get("Start Speed", 150)
        self.body.position = Vec2(self.center.x + distance, self.center.y)
        self.body.velocity = Vec2(0, -speed)
        self.path = [self.body.position.copy()]

    def sync_from_controls(self, values: dict[str, float]) -> None:
        self.gravity_constant = values["Gravity Constant"]

    def update(self, dt: float) -> None:
        offset = self.center - self.body.position
        r2 = max(offset.magnitude_squared(), 100)
        gravity = offset.normalized() * (self.gravity_constant / r2)
        self.body.apply_force(gravity * self.body.mass, dt)
        self.body.integrate(dt)
        self.path.append(self.body.position.copy())
        if len(self.path) > 500:
            self.path.pop(0)

        distance = (self.body.position - self.center).magnitude()
        self.info_lines = [
            f"Position: ({fmt(self.body.position.x)}, {fmt(self.body.position.y)})",
            f"Velocity: ({fmt(self.body.velocity.x)}, {fmt(self.body.velocity.y)})",
            f"Distance to center: {fmt(distance)}",
            f"Gravity accel magnitude: {fmt(gravity.magnitude())}",
            "Try changing start speed to see a crash, orbit, or escape.",
        ]

    def draw(self, canvas: tk.Canvas) -> None:
        draw_grid(canvas)
        if len(self.path) > 1:
            points = []
            for p in self.path:
                points.extend((p.x, p.y))
            canvas.create_line(*points, fill=ACCENT_3, width=2)
        canvas.create_oval(self.center.x - 28, self.center.y - 28, self.center.x + 28, self.center.y + 28, fill=ACCENT, outline="")
        canvas.create_oval(
            self.body.position.x - self.body.radius,
            self.body.position.y - self.body.radius,
            self.body.position.x + self.body.radius,
            self.body.position.y + self.body.radius,
            fill=ACCENT_2,
            outline="",
        )
        draw_arrow(canvas, self.body.position, self.body.position + (self.center - self.body.position).normalized() * 80, "#dc2626", "gravity")
        draw_arrow(canvas, self.body.position, self.body.position + self.body.velocity * 0.4, ACCENT_3, "velocity")

    def on_click(self, x: float, y: float) -> None:
        self.body.position = Vec2(x, y)
        tangent = (self.body.position - self.center).perpendicular().normalized()
        speed = self.app.current_control_values().get("Start Speed", 150)
        self.body.velocity = tangent * speed
        self.path = [self.body.position.copy()]


def draw_grid(canvas: tk.Canvas) -> None:
    for x in range(0, CANVAS_W, 40):
        canvas.create_line(x, 0, x, CANVAS_H, fill=GRID)
    for y in range(0, CANVAS_H, 40):
        canvas.create_line(0, y, CANVAS_W, y, fill=GRID)


def draw_target(canvas: tk.Canvas, target: Vec2) -> None:
    canvas.create_oval(target.x - 10, target.y - 10, target.x + 10, target.y + 10, outline=ACCENT_2, width=2)
    canvas.create_line(target.x - 14, target.y, target.x + 14, target.y, fill=ACCENT_2, width=2)
    canvas.create_line(target.x, target.y - 14, target.x, target.y + 14, fill=ACCENT_2, width=2)


def draw_arrow(canvas: tk.Canvas, start: Vec2, end: Vec2, color: str, label: str) -> None:
    canvas.create_line(start.x, start.y, end.x, end.y, fill=color, width=3, arrow=tk.LAST, arrowshape=(12, 14, 5))
    canvas.create_text(end.x + 18, end.y, text=label, fill=color, font=("Segoe UI", 10, "bold"))


def draw_spring(canvas: tk.Canvas, start: Vec2, end: Vec2) -> None:
    direction = end - start
    length = max(direction.magnitude(), 1)
    unit = direction / length
    normal = unit.perpendicular()
    points = [start]
    segments = 12
    for i in range(1, segments):
        t = i / segments
        offset = math.sin(t * math.pi * 8) * 12
        points.append(start + direction * t + normal * offset)
    points.append(end)
    flat = []
    for p in points:
        flat.extend((p.x, p.y))
    canvas.create_line(*flat, fill="#334155", width=3, smooth=True)


class PhysicsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Physics Simulation Playground")
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

        self.mode_classes = [VectorMode, GravityMode, CollisionMode, SpringMode, OrbitMode]
        self.control_vars: dict[str, tk.DoubleVar] = {}

        self.build_layout()
        self.switch_mode(VectorMode)
        self.tick()

    def build_layout(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(top, text="Physics Simulation Playground", style="Title.TLabel").pack(side=tk.LEFT)

        self.mode_button_frame = ttk.Frame(top)
        self.mode_button_frame.pack(side=tk.RIGHT)

        for mode_cls in self.mode_classes:
            ttk.Button(
                self.mode_button_frame,
                text=mode_cls.name.replace(" Lab", ""),
                command=lambda cls=mode_cls: self.switch_mode(cls),
            ).pack(side=tk.LEFT, padx=4)

        content = ttk.Frame(outer)
        content.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(content, style="Panel.TFrame")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_W, height=CANVAS_H, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(padx=(0, 14))
        self.canvas.bind("<Button-1>", self.handle_click)

        self.side_panel = ttk.Frame(content, style="Panel.TFrame")
        self.side_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.mode_label = ttk.Label(self.side_panel, text="", style="Title.TLabel")
        self.mode_label.pack(anchor="w", padx=14, pady=(12, 4))

        self.description_label = ttk.Label(self.side_panel, text="", wraplength=260, style="Muted.TLabel")
        self.description_label.pack(anchor="w", padx=14, pady=(0, 10))

        self.controls_frame = ttk.Frame(self.side_panel, style="Panel.TFrame")
        self.controls_frame.pack(fill=tk.X, padx=14, pady=(0, 10))

        ttk.Button(self.side_panel, text="Reset", command=self.reset_current_mode).pack(anchor="w", padx=14, pady=(0, 10))

        self.info_label = ttk.Label(self.side_panel, text="", justify=tk.LEFT, wraplength=260)
        self.info_label.pack(anchor="w", padx=14, pady=(0, 10))

        self.formula_label = ttk.Label(self.side_panel, text="", justify=tk.LEFT, wraplength=260, style="Muted.TLabel")
        self.formula_label.pack(anchor="w", padx=14, pady=(0, 10))

    def current_control_values(self) -> dict[str, float]:
        return {name: var.get() for name, var in self.control_vars.items()}

    def switch_mode(self, mode_cls: type[BaseMode]) -> None:
        self.current_mode = mode_cls(self)
        self.render_controls()
        self.refresh_side_panel()

    def reset_current_mode(self) -> None:
        self.current_mode.reset()
        self.render_controls()
        self.refresh_side_panel()

    def render_controls(self) -> None:
        for child in self.controls_frame.winfo_children():
            child.destroy()
        self.control_vars.clear()

        for name, min_value, max_value, default, resolution in self.current_mode.controls:
            ttk.Label(self.controls_frame, text=name).pack(anchor="w")
            var = tk.DoubleVar(value=default)
            scale = tk.Scale(
                self.controls_frame,
                from_=min_value,
                to=max_value,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                bg=PANEL,
                fg=TEXT,
                highlightthickness=0,
                troughcolor="#243042",
                activebackground=ACCENT,
                font=("Segoe UI", 9),
                command=lambda _value, n=name: self.on_control_change(n),
                variable=var,
            )
            scale.pack(fill=tk.X, pady=(0, 8))
            self.control_vars[name] = var

        self.apply_control_values()

    def on_control_change(self, _name: str) -> None:
        self.apply_control_values()

    def apply_control_values(self) -> None:
        values = self.current_control_values()
        sync = getattr(self.current_mode, "sync_from_controls", None)
        if sync:
            sync(values)
        if isinstance(self.current_mode, GravityMode):
            self.current_mode.launch(values)
        elif isinstance(self.current_mode, OrbitMode):
            self.current_mode.reset_orbit(values)

    def refresh_side_panel(self) -> None:
        self.mode_label.configure(text=self.current_mode.name)
        self.description_label.configure(text=self.current_mode.description)

    def handle_click(self, event: tk.Event) -> None:
        self.current_mode.on_click(event.x, event.y)

    def tick(self) -> None:
        dt = 1 / 60
        self.current_mode.update(dt)
        self.draw()
        self.root.after(16, self.tick)

    def draw(self) -> None:
        self.canvas.delete("all")
        self.current_mode.draw(self.canvas)
        self.info_label.configure(text="\n".join(self.current_mode.info_lines))
        self.formula_label.configure(
            text="Core formulas:\n" + "\n".join(f"- {formula}" for formula in self.current_mode.formulas)
        )
        self.canvas.create_text(
            12,
            12,
            anchor="nw",
            text="Click the canvas to interact with the current lab.",
            fill=DARK_TEXT,
            font=("Segoe UI", 10, "bold"),
        )


def main() -> None:
    root = tk.Tk()
    app = PhysicsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
