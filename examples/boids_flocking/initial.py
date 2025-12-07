#!/usr/bin/env python3
"""
Initial (SUBOPTIMAL) implementation of Boids Flocking Simulation.

This file serves as the starting point for evolutionary optimization.
The implementation is deliberately suboptimal to allow room for improvement.

Known issues to evolve:
1. Behavior weights are not well-tuned
2. Simple linear distance weighting for separation
3. Basic collision threshold
4. Naive scoring function
5. No adaptive parameters

Target fitness: ~40-50 (should evolve to 85+)
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any


# ============================================================================
# Vector2D - Basic 2D vector operations
# ============================================================================

@dataclass
class Vector2D:
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


# ============================================================================
# Boid - Individual flocking agent
# ============================================================================

@dataclass
class Boid:
    position: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    acceleration: Vector2D = field(default_factory=lambda: Vector2D(0, 0))

    # SUBOPTIMAL: These weights could be much better tuned
    separation_weight: float = 1.5  # Too aggressive
    alignment_weight: float = 1.0   # Could be higher
    cohesion_weight: float = 1.0    # Could be higher

    max_speed: float = 4.0
    max_force: float = 0.1
    perception_radius: float = 50.0
    separation_radius: float = 25.0

    def apply_force(self, force: Vector2D) -> None:
        self.acceleration = self.acceleration + force

    def update(self) -> None:
        self.velocity = self.velocity + self.acceleration
        self.velocity = self.velocity.limit(self.max_speed)
        self.position = self.position + self.velocity
        self.acceleration = Vector2D(0, 0)

    def seek(self, target: Vector2D) -> Vector2D:
        desired = target - self.position
        desired = desired.normalize() * self.max_speed
        steer = desired - self.velocity
        return steer.limit(self.max_force)

    def separation(self, neighbors: List["Boid"]) -> Vector2D:
        """SUBOPTIMAL: Simple inverse distance weighting."""
        steer = Vector2D(0, 0)
        count = 0

        for other in neighbors:
            d = self.position.distance_to(other.position)
            if 0 < d < self.separation_radius:
                diff = self.position - other.position
                diff = diff.normalize()
                # SUBOPTIMAL: Linear inverse (should be inverse square)
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
        neighbors = [b for b in boids if b is not self]
        self.apply_force(self.separation(neighbors))
        self.apply_force(self.alignment(neighbors))
        self.apply_force(self.cohesion(neighbors))

    def wrap_edges(self, width: float, height: float) -> None:
        if self.position.x > width:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = width
        if self.position.y > height:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = height


# ============================================================================
# Simulation
# ============================================================================

class Simulation:
    def __init__(
        self,
        width: float = 800,
        height: float = 600,
        num_boids: int = 50
    ):
        self.width = width
        self.height = height
        self.boids: List[Boid] = []
        self.collision_count = 0
        self.step_count = 0

        # Initialize flock
        for _ in range(num_boids):
            position = Vector2D(
                random.uniform(0, width),
                random.uniform(0, height)
            )
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 4)
            velocity = Vector2D(
                math.cos(angle) * speed,
                math.sin(angle) * speed
            )
            self.boids.append(Boid(position=position, velocity=velocity))

    def step(self) -> None:
        for boid in self.boids:
            boid.flock(self.boids)

        for boid in self.boids:
            boid.update()
            boid.wrap_edges(self.width, self.height)

        # SUBOPTIMAL: Simple collision counting
        collision_threshold = 10.0
        for i, b1 in enumerate(self.boids):
            for b2 in self.boids[i + 1:]:
                if b1.position.distance_to(b2.position) < collision_threshold:
                    self.collision_count += 1

        self.step_count += 1

    def get_metrics(self) -> Dict[str, float]:
        # Average separation
        separations = []
        for boid in self.boids:
            min_dist = float("inf")
            for other in self.boids:
                if other is not boid:
                    dist = boid.position.distance_to(other.position)
                    min_dist = min(min_dist, dist)
            if min_dist != float("inf"):
                separations.append(min_dist)
        avg_separation = sum(separations) / len(separations) if separations else 0

        # Alignment score
        alignment_scores = []
        for boid in self.boids:
            neighbors = [
                b for b in self.boids
                if b is not boid and boid.position.distance_to(b.position) < 50
            ]
            if neighbors:
                avg_vx = sum(n.velocity.x for n in neighbors) / len(neighbors)
                avg_vy = sum(n.velocity.y for n in neighbors) / len(neighbors)
                avg_vel = Vector2D(avg_vx, avg_vy)
                if boid.velocity.magnitude() > 0 and avg_vel.magnitude() > 0:
                    dot = boid.velocity.x * avg_vel.x + boid.velocity.y * avg_vel.y
                    alignment = dot / (boid.velocity.magnitude() * avg_vel.magnitude())
                    alignment_scores.append((alignment + 1) / 2)
        alignment_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5

        # Cohesion score
        center_x = sum(b.position.x for b in self.boids) / len(self.boids)
        center_y = sum(b.position.y for b in self.boids) / len(self.boids)
        center = Vector2D(center_x, center_y)
        distances = [b.position.distance_to(center) for b in self.boids]
        avg_dist = sum(distances) / len(distances)
        max_dist = math.sqrt(self.width**2 + self.height**2) / 4
        cohesion_score = max(0, 1 - avg_dist / max_dist)

        return {
            "avg_separation": avg_separation,
            "alignment_score": alignment_score,
            "cohesion_score": cohesion_score,
            "total_collisions": self.collision_count,
            "collision_rate": self.collision_count / self.step_count if self.step_count > 0 else 0
        }


def calculate_score(metrics: Dict[str, float]) -> float:
    """SUBOPTIMAL scoring function."""
    separation_penalty = abs(metrics["avg_separation"] - 30) / 30
    separation_score = max(0, 1 - separation_penalty)
    collision_penalty = min(1, metrics["collision_rate"] * 10)

    combined = (
        0.25 * separation_score +
        0.25 * metrics["alignment_score"] +
        0.25 * metrics["cohesion_score"] +
        0.25 * (1 - collision_penalty)
    )

    return max(0, min(100, combined * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--boids", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("BOIDS FLOCKING SIMULATION (Initial Version)")
    print("=" * 60)

    sim = Simulation(num_boids=args.boids)

    for step in range(args.steps):
        sim.step()
        if (step + 1) % 100 == 0:
            m = sim.get_metrics()
            print(f"Step {step + 1}: collisions={m['total_collisions']}, "
                  f"align={m['alignment_score']:.3f}, coh={m['cohesion_score']:.3f}")

    metrics = sim.get_metrics()
    score = calculate_score(metrics)
    correct = score >= 40

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Avg Separation: {metrics['avg_separation']:.2f}")
    print(f"Alignment: {metrics['alignment_score']:.3f}")
    print(f"Cohesion: {metrics['cohesion_score']:.3f}")
    print(f"Collisions: {metrics['total_collisions']}")
    print(f"Score: {score:.2f}")
    print(f"Correct: {correct}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "correct.json", "w") as f:
        json.dump({"correct": correct}, f)

    return 0 if correct else 1


if __name__ == "__main__":
    sys.exit(main())
