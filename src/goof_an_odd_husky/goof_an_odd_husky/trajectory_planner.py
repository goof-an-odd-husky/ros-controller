from goof_an_odd_husky.obstacles import Obstacle
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class TrajectoryPlanner(ABC):
    """Abstract base class for trajectory planning algorithms.

    Defines the interface for planners that compute collision-free trajectories
    from a start pose to a goal pose while avoiding obstacles.

    Attributes:
        start_pose: Start position as [x, y, theta] numpy array.
        goal_pose: Goal position as [x, y, theta] numpy array.
        obstacles: List of Obstacle objects (Points, Circles, Lines).
    """

    start_pose: NDArray[np.floating]
    goal_pose: NDArray[np.floating]
    obstacles: list[Obstacle]

    def setup_poses(
        self,
        start_pose: NDArray[np.floating] | list[float],
        goal_pose: NDArray[np.floating] | list[float],
    ) -> None:
        """Initialize the trajectory planner with start and goal poses.

        Args:
            start_pose: Start position as [x, y, theta] array-like.
            goal_pose: Goal position as [x, y, theta] array-like.
        """
        self.start_pose = np.array(start_pose)
        self.goal_pose = np.array(goal_pose)
        self.obstacles = []

    def update_obstacles(self, obstacles: list[Obstacle]) -> None:
        """Update the obstacle configuration in the environment.

        Args:
            obstacles: A list of Obstacle objects.
        """
        self.obstacles = obstacles

    @abstractmethod
    def plan(self) -> NDArray[np.floating] | None:
        """Compute a trajectory from start to goal avoiding obstacles.

        Returns:
            Nx4 numpy array where each row is [x, y, theta, dt] representing
            the planned trajectory, or None if no valid path exists.
            dt represents the time duration to reach the next point.
        """
        ...

    @abstractmethod
    def get_distance_goal(self) -> float:
        """Return the distance to the goal.

        Returns:
            The straight line distance to the goal.
        """
        ...

    def transform_trajectory(self, dx: float, dy: float, dtheta: float):
        """
        Transforms the internal trajectory guess to account for robot motion.
        This keeps the trajectory aligned with the world while the robot frame moves.

        Args:
            dx, dy: Change in robot position relative to the previous frame's frame.
            dtheta: Change in robot heading.
        """
        ...

    def move_goal(
        self,
        new_xy: tuple[float, float],
        new_theta: tuple[float],
    ) -> None:
        """
        Move the goal point.

        Args:
            new_xy: a tuple of x and y coordinates (relative) of the goal.
            new_theta: a tuple of a theta - angle (relative) of the vehicle when positioned at the goal
        """
        ...

    def refine(
        self,
        iterations: int = 1,
        current_velocity: float = 0,
        current_omega: float = 0,
    ) -> bool:
        """Refine the current trajectory (optional for some planners).

        Default implementation does nothing. Override for optimization-based planners.

        Args:
            iterations: how many iterations to take during a single refinement.
            current_velocity: current velocity (in m/s).
            current_omega: current angular velocity (in rad/s).
        """
        return True

    def get_trajectory(self) -> NDArray[np.floating] | None:
        """Get the current planned trajectory.

        Returns:
            Nx4 numpy array where each row is [x, y, theta, dt], or None if
            no trajectory has been computed yet.
        """
        return self.plan()

    @abstractmethod
    def get_length(self) -> int:
        """Get the length of the current trajectory.

        Returns:
            The length of the current trajectory
        """
        ...

    def set_start_pose(self, pose: NDArray[np.floating] | list[float]) -> None:
        """Update the start pose.

        Args:
            pose: New start position as [x, y, theta] array-like.
        """
        self.start_pose = np.array(pose)

    def set_goal_pose(self, pose: NDArray[np.floating] | list[float]) -> None:
        """Update the goal pose.

        Args:
            pose: New goal position as [x, y, theta] array-like.
        """
        self.goal_pose = np.array(pose)
