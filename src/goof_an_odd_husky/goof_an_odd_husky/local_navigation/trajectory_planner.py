from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from goof_an_odd_husky_common.types import Trajectory
from goof_an_odd_husky_common.obstacles import Obstacle


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
    def plan(self) -> Trajectory | None:
        """Compute a trajectory from start to goal avoiding obstacles.

        Returns:
            Trajectory | None: Nx4 array [x, y, theta, dt] or None if no valid path exists.
        """
        ...

    @abstractmethod
    def get_distance_goal(self) -> float:
        """Return the distance to the goal.

        Returns:
            float: The straight line distance to the goal.
        """
        ...

    def transform_trajectory(self, dx: float, dy: float, dtheta: float) -> None:
        """Transforms the internal trajectory guess to account for robot motion.

        This keeps the trajectory aligned with the world while the robot frame moves.

        Args:
            dx: Change in robot position X relative to the previous frame's frame.
            dy: Change in robot position Y.
            dtheta: Change in robot heading.
        """
        ...

    def move_goal(
        self,
        new_xy: tuple[float, float],
        new_theta: tuple[float],
    ) -> None:
        """Move the goal point.

        Args:
            new_xy: A tuple of x and y coordinates (relative) of the goal.
            new_theta: A tuple containing the orientation of the vehicle at the goal.
        """
        ...

    def refine(
        self,
        iterations: int = 1,
        current_velocity: float = 0.0,
        current_omega: float = 0.0,
    ) -> bool:
        """Refine the current trajectory (optional for some planners).

        Default implementation does nothing. Override for optimization-based planners.

        Args:
            iterations: How many iterations to take during a single refinement.
            current_velocity: Current linear velocity (in m/s).
            current_omega: Current angular velocity (in rad/s).

        Returns:
            bool: Success status of the refinement process.
        """
        return True

    def get_trajectory(self) -> Trajectory | None:
        """Get the current planned trajectory.

        Returns:
            Trajectory | None: Nx4 array [x, y, theta, dt], or None if not yet computed.
        """
        return self.plan()

    @abstractmethod
    def get_length(self) -> int:
        """Get the length of the current trajectory.

        Returns:
            int: The number of waypoints in the current trajectory.
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
