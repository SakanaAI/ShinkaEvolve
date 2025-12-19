#!/usr/bin/env python3
"""
Boids Flocking Simulation - Main Entry Point

This simulation evolves flocking behavior by optimizing separation, alignment,
and cohesion weights to minimize collisions while maintaining tight grouping.

Usage:
    python main.py                    # Run with visualization
    python main.py --headless         # Run without visualization
    python main.py --steps 500        # Run for specific number of steps
"""

import argparse
import json
import sys
from pathlib import Path

from render import create_renderer
from simulation import SimulationConfig, SimulationEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Boids Flocking Simulation")
    parser.add_argument(
        "--headless", action="store_true", help="Run without graphical output"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Run with graphical output (opposite of --headless)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps (default: 1000)",
    )
    parser.add_argument(
        "--boids",
        type=int,
        default=50,
        help="Number of boids in the simulation (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory for output files"
    )
    # For framework compatibility (--results_dir is passed by shinka legacy evaluator)
    parser.add_argument(
        "--results_dir", type=str, default=None, help="Alias for --output-dir (framework compat)"
    )
    parser.add_argument(
        "--program_path", type=str, default=None, help="Ignored (framework compat)"
    )
    return parser.parse_args()


def calculate_combined_score(metrics: dict) -> float:
    """
    Calculate a combined fitness score from the simulation metrics.

    SUBOPTIMAL SCORING (room for evolution):
    - Simple weighted average
    - Doesn't account for trade-offs between metrics
    - Could use more sophisticated aggregation
    """
    # Extract key metrics
    avg_separation = metrics.get("avg_separation", 0)
    alignment_score = metrics.get("alignment_score", 0.5)
    cohesion_score = metrics.get("cohesion_score", 0)
    collision_rate = metrics.get("collision_rate", 1)

    # SUBOPTIMAL: Simple weighting scheme
    # Ideal separation is around 20-40 (not too close, not too far)
    separation_penalty = abs(avg_separation - 30) / 30
    separation_score = max(0, 1 - separation_penalty)

    # Penalize collisions heavily
    collision_penalty = min(1, collision_rate * 10)

    # Combined score (higher is better)
    combined = (
        0.25 * separation_score
        + 0.25 * alignment_score
        + 0.25 * cohesion_score
        + 0.25 * (1 - collision_penalty)
    )

    return max(0, min(100, combined * 100))


def evaluate_simulation(args) -> dict:
    """Run simulation and return evaluation results."""
    # Create simulation config
    config = SimulationConfig(
        num_boids=args.boids,
        max_steps=args.steps,
        # SUBOPTIMAL weights (evolution should improve these)
        separation_weight=1.5,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        max_speed=4.0,
        max_force=0.1,
        perception_radius=50.0,
        separation_radius=25.0,
    )

    # Create and run simulation
    sim = SimulationEnvironment(config)

    # Create renderer if --gui is set (default is headless for framework eval)
    renderer = None
    headless = args.headless or not args.gui  # Default to headless unless --gui is set
    if not headless:
        try:
            renderer = create_renderer(
                headless=False, width=config.width, height=config.height
            )
        except Exception as e:
            print(f"Warning: Could not create graphical renderer: {e}")
            print("Falling back to headless mode.")

    # Run simulation
    for step in range(args.steps):
        sim.step()

        # Render if available
        if renderer and hasattr(renderer, "render"):
            try:
                positions = sim.get_boid_positions()
                velocities = sim.get_boid_velocities()
                renderer.render(positions, velocities, step)
            except Exception:
                pass  # Continue even if rendering fails

        # Progress output every 100 steps
        if (step + 1) % 100 == 0:
            metrics = sim.get_final_metrics()
            print(
                f"Step {step + 1}/{args.steps}: "
                f"collisions={metrics.get('total_collisions', 0)}, "
                f"alignment={metrics.get('alignment_score', 0):.3f}, "
                f"cohesion={metrics.get('cohesion_score', 0):.3f}"
            )

    # Close renderer
    if renderer and hasattr(renderer, "close"):
        renderer.close()

    # Get final metrics
    final_metrics = sim.get_final_metrics()
    combined_score = calculate_combined_score(final_metrics)

    return {
        "metrics": final_metrics,
        "combined_score": combined_score,
        "correct": combined_score >= 40,  # SUBOPTIMAL threshold (should be higher)
    }


def main():
    """Main entry point."""
    args = parse_args()
    # Use --results_dir if provided (framework compat), otherwise --output-dir
    output_dir = Path(args.results_dir if args.results_dir else args.output_dir)

    print("=" * 60)
    print("BOIDS FLOCKING SIMULATION")
    print("=" * 60)
    print(f"Boids: {args.boids}")
    print(f"Steps: {args.steps}")
    headless = args.headless or not args.gui  # Default to headless unless --gui
    print(f"Mode: {'Headless' if headless else 'Graphical'}")
    print("=" * 60)

    # Run evaluation
    result = evaluate_simulation(args)

    # Print results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    metrics = result["metrics"]
    print(f"Average Separation: {metrics.get('avg_separation', 0):.2f}")
    print(f"Alignment Score: {metrics.get('alignment_score', 0):.3f}")
    print(f"Cohesion Score: {metrics.get('cohesion_score', 0):.3f}")
    print(f"Total Collisions: {metrics.get('total_collisions', 0)}")
    print(f"Collision Rate: {metrics.get('collision_rate', 0):.4f}")
    print(f"Combined Score: {result['combined_score']:.2f}")
    print(f"Correct: {result['correct']}")
    print("=" * 60)

    # Write output files
    metrics_file = output_dir / "metrics.json"
    correct_file = output_dir / "correct.json"

    # Write full evaluation results including combined_score
    eval_output = {
        **metrics,
        "combined_score": result["combined_score"],
        "correct": result["correct"],
        "details": f"Collisions: {metrics.get('total_collisions', 0)}, "
                   f"Alignment: {metrics.get('alignment_score', 0):.3f}, "
                   f"Cohesion: {metrics.get('cohesion_score', 0):.3f}"
    }
    with open(metrics_file, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"Metrics written to: {metrics_file}")

    with open(correct_file, "w") as f:
        json.dump({"correct": result["correct"]}, f)
    print(f"Correctness written to: {correct_file}")

    return 0 if result["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
