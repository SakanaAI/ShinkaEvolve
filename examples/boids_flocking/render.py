"""
Renderer for visualizing the boids simulation.
Supports both matplotlib (graphical) and terminal (headless) output.
"""

import math
from typing import List, Tuple, Optional


class TerminalRenderer:
    """Simple ASCII renderer for headless mode."""

    def __init__(self, width: int = 80, height: int = 24, sim_width: float = 800, sim_height: float = 600):
        self.width = width
        self.height = height
        self.sim_width = sim_width
        self.sim_height = sim_height

    def render(
        self,
        positions: List[Tuple[float, float]],
        velocities: List[Tuple[float, float]],
        step: int = 0
    ) -> None:
        """Render boids to ASCII art and print to terminal."""
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        for x, y in positions:
            # Map simulation coords to terminal coords
            tx = int((x / self.sim_width) * (self.width - 1))
            ty = int((y / self.sim_height) * (self.height - 1))

            # Clamp to bounds
            tx = max(0, min(self.width - 1, tx))
            ty = max(0, min(self.height - 1, ty))

            grid[ty][tx] = "*"

        # Build output string
        output = f"Step: {step}\n"
        output += "+" + "-" * self.width + "+\n"
        for row in grid:
            output += "|" + "".join(row) + "|\n"
        output += "+" + "-" * self.width + "+"

        print(output)

    def close(self) -> None:
        """No cleanup needed for terminal renderer."""
        pass


class MatplotlibRenderer:
    """Matplotlib-based renderer for graphical output."""

    def __init__(self, width: float = 800, height: float = 600):
        self.width = width
        self.height = height
        self.fig = None
        self.ax = None
        self.scatter = None
        self.quiver = None

    def initialize(self) -> None:
        """Initialize matplotlib figure."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect("equal")
            self.ax.set_facecolor("#1a1a2e")
            self.fig.patch.set_facecolor("#1a1a2e")
            self.ax.axis("off")

        except ImportError:
            raise RuntimeError("matplotlib not available for graphical rendering")

    def render(
        self,
        positions: List[Tuple[float, float]],
        velocities: List[Tuple[float, float]],
        step: int = 0
    ) -> None:
        """Render current frame."""
        import matplotlib.pyplot as plt

        if self.fig is None:
            self.initialize()

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_facecolor("#1a1a2e")
        self.ax.axis("off")

        if positions:
            xs, ys = zip(*positions)
            vxs, vys = zip(*velocities) if velocities else (None, None)

            # Draw boids as points
            self.ax.scatter(xs, ys, c="#00d9ff", s=30, alpha=0.8)

            # Draw velocity vectors
            if vxs and vys:
                # Normalize velocities for arrow display
                scale = 5.0
                self.ax.quiver(
                    xs, ys, vxs, vys,
                    color="#ff6b6b",
                    alpha=0.5,
                    scale=50,
                    width=0.003
                )

        self.ax.set_title(f"Step: {step}", color="white", fontsize=12)
        plt.pause(0.001)

    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        if self.fig:
            self.fig.savefig(filename, dpi=100, facecolor="#1a1a2e")

    def close(self) -> None:
        """Close the renderer."""
        if self.fig:
            import matplotlib.pyplot as plt
            plt.close(self.fig)


def create_renderer(headless: bool = False, width: float = 800, height: float = 600, **kwargs) -> Optional[object]:
    """Factory function to create appropriate renderer."""
    if headless:
        return TerminalRenderer(sim_width=width, sim_height=height, **kwargs)
    else:
        renderer = MatplotlibRenderer(width=width, height=height, **kwargs)
        try:
            renderer.initialize()
            return renderer
        except RuntimeError:
            # Fall back to terminal if matplotlib not available
            return TerminalRenderer(sim_width=width, sim_height=height)
