"""
Simulation environment for managing a flock of boids.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from boid import Boid, Vector2D


@dataclass
class SimulationConfig:
    """Configuration for the boids simulation."""
    width: float = 800.0
    height: float = 600.0
    num_boids: int = 50
    max_steps: int = 1000

    # Boid parameters (SUBOPTIMAL: could be evolved)
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    max_speed: float = 4.0
    max_force: float = 0.1
    perception_radius: float = 50.0
    separation_radius: float = 25.0


class SimulationEnvironment:
    """Manages a flock of boids and runs the simulation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.boids: List[Boid] = []
        self.step_count: int = 0
        self.collision_count: int = 0
        self.metrics_history: List[Dict[str, float]] = []
        self._initialize_flock()

    def _initialize_flock(self) -> None:
        """Create the initial flock with random positions and velocities."""
        for _ in range(self.config.num_boids):
            position = Vector2D(
                random.uniform(0, self.config.width),
                random.uniform(0, self.config.height)
            )
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, self.config.max_speed)
            velocity = Vector2D(
                math.cos(angle) * speed,
                math.sin(angle) * speed
            )

            boid = Boid(
                position=position,
                velocity=velocity,
                separation_weight=self.config.separation_weight,
                alignment_weight=self.config.alignment_weight,
                cohesion_weight=self.config.cohesion_weight,
                max_speed=self.config.max_speed,
                max_force=self.config.max_force,
                perception_radius=self.config.perception_radius,
                separation_radius=self.config.separation_radius
            )
            self.boids.append(boid)

    def step(self) -> Dict[str, float]:
        """Run one simulation step and return current metrics."""
        # Apply flocking behavior to each boid
        for boid in self.boids:
            boid.flock(self.boids)

        # Update positions and wrap edges
        for boid in self.boids:
            boid.update()
            boid.wrap_edges(self.config.width, self.config.height)

        # Count collisions (boids too close together)
        step_collisions = self._count_collisions()
        self.collision_count += step_collisions

        # Calculate metrics
        metrics = self._calculate_metrics()
        metrics["step_collisions"] = step_collisions
        self.metrics_history.append(metrics)

        self.step_count += 1
        return metrics

    def _count_collisions(self) -> int:
        """Count pairs of boids that are too close (collision)."""
        collision_threshold = 10.0  # Minimum safe distance
        collisions = 0

        for i, boid1 in enumerate(self.boids):
            for boid2 in self.boids[i + 1:]:
                distance = boid1.position.distance_to(boid2.position)
                if distance < collision_threshold:
                    collisions += 1

        return collisions

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate current flock metrics."""
        if not self.boids:
            return {"avg_separation": 0, "alignment_score": 0, "cohesion_score": 0}

        # Average separation (distance to nearest neighbor)
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

        # Alignment score (how similar are velocity directions)
        alignment_scores = []
        for boid in self.boids:
            neighbors = [
                b for b in self.boids
                if b is not boid and boid.position.distance_to(b.position) < boid.perception_radius
            ]
            if neighbors:
                # Calculate average velocity direction
                avg_vx = sum(n.velocity.x for n in neighbors) / len(neighbors)
                avg_vy = sum(n.velocity.y for n in neighbors) / len(neighbors)
                avg_vel = Vector2D(avg_vx, avg_vy)

                if boid.velocity.magnitude() > 0 and avg_vel.magnitude() > 0:
                    # Dot product normalized (1 = perfect alignment)
                    dot = (boid.velocity.x * avg_vel.x + boid.velocity.y * avg_vel.y)
                    alignment = dot / (boid.velocity.magnitude() * avg_vel.magnitude())
                    alignment_scores.append((alignment + 1) / 2)  # Normalize to 0-1

        alignment_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5

        # Cohesion score (how close are boids to the flock center)
        center_x = sum(b.position.x for b in self.boids) / len(self.boids)
        center_y = sum(b.position.y for b in self.boids) / len(self.boids)
        center = Vector2D(center_x, center_y)

        distances_to_center = [b.position.distance_to(center) for b in self.boids]
        avg_distance = sum(distances_to_center) / len(distances_to_center)

        # Normalize cohesion (lower distance = better cohesion)
        max_expected_distance = math.sqrt(self.config.width**2 + self.config.height**2) / 4
        cohesion_score = max(0, 1 - avg_distance / max_expected_distance)

        return {
            "avg_separation": avg_separation,
            "alignment_score": alignment_score,
            "cohesion_score": cohesion_score,
            "avg_distance_to_center": avg_distance
        }

    def run(self, steps: int = None) -> Dict[str, Any]:
        """Run simulation for specified steps and return final metrics."""
        steps = steps or self.config.max_steps

        for _ in range(steps):
            self.step()

        return self.get_final_metrics()

    def get_final_metrics(self) -> Dict[str, Any]:
        """Get final aggregated metrics."""
        if not self.metrics_history:
            return {}

        # Average over last 100 steps for stability
        recent = self.metrics_history[-100:] if len(self.metrics_history) >= 100 else self.metrics_history

        return {
            "avg_separation": sum(m["avg_separation"] for m in recent) / len(recent),
            "alignment_score": sum(m["alignment_score"] for m in recent) / len(recent),
            "cohesion_score": sum(m["cohesion_score"] for m in recent) / len(recent),
            "total_collisions": self.collision_count,
            "collision_rate": self.collision_count / self.step_count if self.step_count > 0 else 0,
            "steps_completed": self.step_count
        }

    def get_boid_positions(self) -> List[Tuple[float, float]]:
        """Get current positions of all boids for rendering."""
        return [(b.position.x, b.position.y) for b in self.boids]

    def get_boid_velocities(self) -> List[Tuple[float, float]]:
        """Get current velocities of all boids for rendering."""
        return [(b.velocity.x, b.velocity.y) for b in self.boids]
