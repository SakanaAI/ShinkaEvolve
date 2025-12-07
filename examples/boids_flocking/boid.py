"""
Boid class implementing separation, alignment, and cohesion behaviors.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Vector2D:
    """Simple 2D vector for boid physics."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        if scalar == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / scalar, self.y / scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> "Vector2D":
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return self / mag

    def limit(self, max_val: float) -> "Vector2D":
        mag = self.magnitude()
        if mag > max_val:
            return self.normalize() * max_val
        return Vector2D(self.x, self.y)

    def distance_to(self, other: "Vector2D") -> float:
        return (self - other).magnitude()


@dataclass
class Boid:
    """A single boid in the flock."""
    position: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    acceleration: Vector2D = field(default_factory=lambda: Vector2D(0, 0))

    # Behavior weights (SUBOPTIMAL: these could be evolved)
    separation_weight: float = 1.0
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0

    # Physical parameters
    max_speed: float = 4.0
    max_force: float = 0.1
    perception_radius: float = 50.0
    separation_radius: float = 25.0

    def apply_force(self, force: Vector2D) -> None:
        """Apply a steering force to the boid."""
        self.acceleration = self.acceleration + force

    def update(self) -> None:
        """Update velocity and position."""
        self.velocity = self.velocity + self.acceleration
        self.velocity = self.velocity.limit(self.max_speed)
        self.position = self.position + self.velocity
        self.acceleration = Vector2D(0, 0)

    def seek(self, target: Vector2D) -> Vector2D:
        """Calculate steering force toward a target."""
        desired = target - self.position
        desired = desired.normalize() * self.max_speed
        steer = desired - self.velocity
        return steer.limit(self.max_force)

    def separation(self, neighbors: List["Boid"]) -> Vector2D:
        """Steer to avoid crowding local flockmates."""
        steer = Vector2D(0, 0)
        count = 0

        for other in neighbors:
            d = self.position.distance_to(other.position)
            if 0 < d < self.separation_radius:
                diff = self.position - other.position
                diff = diff.normalize()
                # SUBOPTIMAL: Simple inverse weighting (could use inverse square)
                diff = diff / d
                steer = steer + diff
                count += 1

        if count > 0:
            steer = steer / count
            if steer.magnitude() > 0:
                steer = steer.normalize() * self.max_speed
                steer = steer - self.velocity
                steer = steer.limit(self.max_force)

        return steer * self.separation_weight

    def alignment(self, neighbors: List["Boid"]) -> Vector2D:
        """Steer towards the average heading of local flockmates."""
        avg_velocity = Vector2D(0, 0)
        count = 0

        for other in neighbors:
            d = self.position.distance_to(other.position)
            if 0 < d < self.perception_radius:
                avg_velocity = avg_velocity + other.velocity
                count += 1

        if count > 0:
            avg_velocity = avg_velocity / count
            avg_velocity = avg_velocity.normalize() * self.max_speed
            steer = avg_velocity - self.velocity
            steer = steer.limit(self.max_force)
            return steer * self.alignment_weight

        return Vector2D(0, 0)

    def cohesion(self, neighbors: List["Boid"]) -> Vector2D:
        """Steer to move toward the average position of local flockmates."""
        center = Vector2D(0, 0)
        count = 0

        for other in neighbors:
            d = self.position.distance_to(other.position)
            if 0 < d < self.perception_radius:
                center = center + other.position
                count += 1

        if count > 0:
            center = center / count
            return self.seek(center) * self.cohesion_weight

        return Vector2D(0, 0)

    def flock(self, boids: List["Boid"]) -> None:
        """Apply all three flocking behaviors."""
        # Filter out self from neighbors
        neighbors = [b for b in boids if b is not self]

        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)

        self.apply_force(sep)
        self.apply_force(ali)
        self.apply_force(coh)

    def wrap_edges(self, width: float, height: float) -> None:
        """Wrap boid around screen edges."""
        if self.position.x > width:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = width

        if self.position.y > height:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = height
