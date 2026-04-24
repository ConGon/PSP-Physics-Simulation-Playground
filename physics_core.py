from __future__ import annotations

from dataclasses import dataclass
import math


EPSILON = 1e-9


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        if abs(scalar) < EPSILON:
            return Vec2()
        return Vec2(self.x / scalar, self.y / scalar)

    def copy(self) -> "Vec2":
        return Vec2(self.x, self.y)

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec2":
        mag = self.magnitude()
        if mag < EPSILON:
            return Vec2()
        return self / mag

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def perpendicular(self) -> "Vec2":
        return Vec2(-self.y, self.x)

    def clamp_magnitude(self, max_magnitude: float) -> "Vec2":
        mag = self.magnitude()
        if mag <= max_magnitude or mag < EPSILON:
            return self.copy()
        return self.normalized() * max_magnitude

    @staticmethod
    def from_angle_radians(angle: float, length: float = 1.0) -> "Vec2":
        return Vec2(math.cos(angle) * length, math.sin(angle) * length)


@dataclass
class Body:
    position: Vec2
    velocity: Vec2
    mass: float = 1.0
    radius: float = 12.0
    restitution: float = 0.85

    def apply_force(self, force: Vec2, dt: float) -> None:
        acceleration = force / max(self.mass, EPSILON)
        self.velocity = self.velocity + acceleration * dt

    def integrate(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt


def reflect_velocity(velocity: Vec2, normal: Vec2, restitution: float) -> Vec2:
    unit_normal = normal.normalized()
    separating_speed = velocity.dot(unit_normal)
    if separating_speed >= 0:
        return velocity
    return velocity - unit_normal * ((1.0 + restitution) * separating_speed)


def resolve_circle_collision(a: Body, b: Body) -> None:
    offset = b.position - a.position
    distance = offset.magnitude()
    min_distance = a.radius + b.radius
    if distance <= 0 or distance >= min_distance:
        return

    normal = offset / distance
    overlap = min_distance - distance
    total_mass = a.mass + b.mass
    if total_mass < EPSILON:
        return

    a.position = a.position - normal * (overlap * (b.mass / total_mass))
    b.position = b.position + normal * (overlap * (a.mass / total_mass))

    relative_velocity = b.velocity - a.velocity
    separating_speed = relative_velocity.dot(normal)
    if separating_speed > 0:
        return

    restitution = min(a.restitution, b.restitution)
    impulse_magnitude = -(1.0 + restitution) * separating_speed
    impulse_magnitude /= (1.0 / max(a.mass, EPSILON)) + (1.0 / max(b.mass, EPSILON))
    impulse = normal * impulse_magnitude

    a.velocity = a.velocity - impulse / max(a.mass, EPSILON)
    b.velocity = b.velocity + impulse / max(b.mass, EPSILON)


def spring_force(current: Vec2, anchor: Vec2, rest_length: float, stiffness: float) -> Vec2:
    displacement = current - anchor
    distance = displacement.magnitude()
    if distance < EPSILON:
        return Vec2()
    extension = distance - rest_length
    direction = displacement / distance
    return direction * (-stiffness * extension)
