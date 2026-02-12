import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.quiver import Quiver
from matplotlib.backend_bases import MouseEvent, CloseEvent
from numpy.typing import NDArray
from enum import Enum


class PathRenderMode(Enum):
    """Rendering mode for trajectory visualization."""

    STRAIGHT = "straight"
    ARC = "arc"
    BOTH = "both"


class TrajectoryVisualizer:
    """Interactive visualizer for Timed Elastic Band (TEB) trajectory optimization.

    Provides a matplotlib-based interface for visualizing robot trajectories with
    interactive obstacle placement. Users can add obstacles with left-click and
    remove them with right-click.

    Attributes:
        fig: Matplotlib figure object.
        ax: Matplotlib axes object for plotting.
        obstacles: List of obstacles as [x, y, radius] arrays.
        obstacle_radius: Fixed radius for all obstacles.
        start_pos: Start position as [x, y, theta] numpy array.
        goal_pos: Goal position as [x, y, theta] numpy array.
        path_line: Line2D object for trajectory visualization.
        start_scatter: PathCollection for start position marker.
        goal_scatter: PathCollection for goal position marker.
        obstacle_patches: List of Circle patches for obstacle visualization.
        quiver: Quiver object for orientation arrows.
        cid: Connection ID for mouse click callback.
        interactive_obstacles: If True, enables mouse interaction. If False,
                               obstacles must be set via set_obstacles().
    """

    fig: Figure
    ax: Axes
    obstacles: list[list[float]]
    obstacle_radius: float
    start_pos: NDArray[np.floating] | None
    goal_pos: NDArray[np.floating] | None
    path_line: Line2D
    path_line_straight: Line2D | None
    start_scatter: PathCollection
    goal_scatter: PathCollection
    obstacle_patches: list[patches.Circle]
    quiver: Quiver | None
    cid: int
    path_render_mode: PathRenderMode
    interactive_obstacles: bool

    def __init__(
        self,
        x_lim: tuple[float, float] = (0, 10),
        y_lim: tuple[float, float] = (0, 10),
        title: str = "TEB Optimization",
        path_render_mode: PathRenderMode | str = PathRenderMode.STRAIGHT,
        interactive_obstacles: bool = True,
    ) -> None:
        """Initialize the visualizer with specified plot dimensions.

        Args:
            x_lim: X-axis limits as (min, max) tuple.
            y_lim: Y-axis limits as (min, max) tuple.
            title: Title for the plot window.
            path_render_mode: How to render trajectory paths - "straight", "arc", or "both".
            interactive_obstacles: Enable/Disable mouse click interaction.
        """
        self._is_open = True
        self.interactive_obstacles = interactive_obstacles

        if isinstance(path_render_mode, str):
            path_render_mode = PathRenderMode(path_render_mode)
        self.path_render_mode = path_render_mode

        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        if self.interactive_obstacles:
            controls = "\nLeft Click: Add Obstacle | Right Click: Remove Obstacle"
        else:
            controls = "\n(System Controlled Obstacles)"

        self.ax.set_title(f"{title}{controls}")
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

        self.obstacles = []
        self.obstacle_radius = 0.5
        self.start_pos = None
        self.goal_pos = None

        if self.path_render_mode == PathRenderMode.STRAIGHT:
            line_style = "b.-"
        else:
            line_style = "b-"

        (self.path_line,) = self.ax.plot(
            [], [], line_style, alpha=1.0, linewidth=1, label="Trajectory"
        )
        self.path_line_straight = None
        if self.path_render_mode == PathRenderMode.BOTH:
            (self.path_line_straight,) = self.ax.plot(
                [], [], "b:", alpha=0.7, linewidth=1
            )

        self.start_scatter = self.ax.scatter(
            [], [], color="green", marker="s", s=100, label="Start", zorder=5
        )
        self.goal_scatter = self.ax.scatter(
            [], [], color="red", marker="*", s=150, label="Goal", zorder=5
        )
        self.obstacle_patches = []
        self.quiver = None

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._close_cid = self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.ax.legend(loc="upper right")
        plt.ion()
        plt.show()

    @property
    def is_open(self) -> bool:
        """Check if the visualization window is still open."""
        return self._is_open

    def _on_close(self, event: CloseEvent) -> None:
        """Handle window close event."""
        self._is_open = False

    def set_start_goal(
        self,
        start: NDArray[np.floating] | list[float],
        goal: NDArray[np.floating] | list[float],
    ) -> None:
        """Set and display start and goal positions.

        Args:
            start: Start position as [x, y, theta] array-like.
            goal: Goal position as [x, y, theta] array-like.
        """
        self.start_pos = np.array(start)
        self.goal_pos = np.array(goal)

        self.start_scatter.set_offsets([self.start_pos[:2]])
        self.goal_scatter.set_offsets([self.goal_pos[:2]])

    def set_obstacles(
        self, obstacles: list[list[float]] | NDArray[np.floating]
    ) -> None:
        """Programmatically set the list of obstacles.

        Args:
            obstacles: List of [x, y] or [x, y, radius].
                      If radius is omitted, uses self.obstacle_radius.
        """
        if not self._is_open:
            return

        for patch in self.obstacle_patches:
            patch.remove()
        self.obstacle_patches.clear()
        self.obstacles.clear()

        for obs in obstacles:
            x, y = float(obs[0]), float(obs[1])
            if len(obs) >= 3:
                r = float(obs[2])
            else:
                r = self.obstacle_radius

            self.obstacles.append([x, y, r])
            self._add_obstacle_patch(x, y, r)

    def update_trajectory(self, poses: NDArray[np.floating] | None) -> None:
        """Update the displayed trajectory with new poses.
        Args:
            poses: Nx4 numpy array where each row is [x, y, theta, dt].
                  Returns early if None or empty.
        """
        if poses is None or len(poses) == 0:
            return

        if self.path_render_mode == PathRenderMode.STRAIGHT:
            self.path_line.set_data(poses[:, 0], poses[:, 1])
        elif self.path_render_mode == PathRenderMode.ARC:
            arc_x, arc_y = self._compute_arc_path(poses)
            self.path_line.set_data(arc_x, arc_y)
        else:
            self.path_line_straight.set_data(poses[:, 0], poses[:, 1])
            arc_x, arc_y = self._compute_arc_path(poses)
            self.path_line.set_data(arc_x, arc_y)

        if self.quiver:
            self.quiver.remove()

        u = np.cos(poses[:, 2])
        v = np.sin(poses[:, 2])
        self.quiver = self.ax.quiver(
            poses[:, 0],
            poses[:, 1],
            u,
            v,
            color="blue",
            scale=20,
            width=0.003,
            alpha=0.5,
            zorder=4,
        )

    def _compute_arc_path(
        self, poses: NDArray[np.floating], points_per_arc: int = 20
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute arc path that respects heading constraints between consecutive poses.

        Args:
            poses: Nx4 array of [x, y, theta, dt] poses.
            points_per_arc: Number of points to use for each arc segment.

        Returns:
            Tuple of (x_coords, y_coords) arrays for the complete arc path.
        """
        if len(poses) < 2:
            return poses[:, 0], poses[:, 1]

        all_x = []
        all_y = []

        for i in range(len(poses) - 1):
            x1, y1, theta1, _ = poses[i]
            x2, y2, theta2, _ = poses[i + 1]

            arc_x, arc_y = self._compute_single_arc(
                x1, y1, theta1, x2, y2, theta2, points_per_arc
            )
            all_x.extend(arc_x)
            all_y.extend(arc_y)

        return np.array(all_x), np.array(all_y)

    def _compute_single_arc(
        self,
        x1: float,
        y1: float,
        theta1: float,
        x2: float,
        y2: float,
        theta2: float,
        num_points: int,
    ) -> tuple[list[float], list[float]]:
        """Compute a single arc segment between two poses using cubic Bezier curve.

        Args:
            x1, y1, theta1: Start pose coordinates and heading.
            x2, y2, theta2: End pose coordinates and heading.
            num_points: Number of points to generate along the arc.

        Returns:
            Tuple of (x_coords, y_coords) for the arc segment.
        """
        distance = np.hypot(x2 - x1, y2 - y1)
        control_distance = distance / 3.0

        control1_x = x1 + control_distance * np.cos(theta1)
        control1_y = y1 + control_distance * np.sin(theta1)

        control2_x = x2 - control_distance * np.cos(theta2)
        control2_y = y2 - control_distance * np.sin(theta2)

        t = np.linspace(0, 1, num_points)
        t = t.reshape(-1, 1)

        bezier_matrix = (
            np.array([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3])
            .squeeze()
            .T
        )

        control_points_x = np.array([x1, control1_x, control2_x, x2])
        control_points_y = np.array([y1, control1_y, control2_y, y2])

        arc_x = (bezier_matrix @ control_points_x).tolist()
        arc_y = (bezier_matrix @ control_points_y).tolist()

        return arc_x, arc_y

    def get_obstacles(self) -> NDArray[np.floating]:
        """Get all obstacles in the environment.

        Returns:
            Nx3 numpy array where each row is [x, y, radius].
            Returns empty (0, 3) array if no obstacles exist.
        """
        if not self.obstacles:
            return np.empty((0, 3))
        return np.array(self.obstacles)

    def draw(self, pause_time: float = 0.01) -> None:
        """Refresh the plot display.

        Args:
            pause_time: Time in seconds to pause for rendering.
        """
        if not self._is_open:
            return

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(pause_time)

    def _on_click(self, event: MouseEvent) -> None:
        """Handle mouse click events for obstacle management.

        Args:
            event: MouseEvent from matplotlib containing click information.
        """
        if not self.interactive_obstacles:
            return

        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:
            self.obstacles.append([x, y, self.obstacle_radius])
            self._add_obstacle_patch(x, y, self.obstacle_radius)

        elif event.button == 3:
            self._remove_nearest_obstacle(x, y)

    def _add_obstacle_patch(self, x: float, y: float, r: float) -> None:
        """Create and display a circular obstacle patch.

        Args:
            x: X-coordinate of obstacle center.
            y: Y-coordinate of obstacle center.
            r: Radius of obstacle.
        """
        circle = patches.Circle((x, y), r, color="black", alpha=0.5)
        self.ax.add_patch(circle)
        self.obstacle_patches.append(circle)

    def _remove_nearest_obstacle(self, x: float, y: float) -> None:
        """Remove the nearest obstacle to a clicked position.

        Args:
            x: X-coordinate of click position.
            y: Y-coordinate of click position.
        """
        if not self.obstacles:
            return

        obs_arr = np.array(self.obstacles)
        dists = np.hypot(obs_arr[:, 0] - x, obs_arr[:, 1] - y)
        idx = np.argmin(dists)

        if dists[idx] < 1.0:
            self.obstacles.pop(idx)
            patch = self.obstacle_patches.pop(idx)
            patch.remove()
