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
        obstacles: Nx3 numpy array where each row is [x, y, radius].
    """

    start_pose: NDArray[np.floating]
    goal_pose: NDArray[np.floating]
    obstacles: NDArray[np.floating]

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
        self.obstacles = np.empty((0, 3))

    def update_obstacles(self, obstacles: NDArray[np.floating]) -> None:
        """Update the obstacle configuration in the environment.

        Args:
            obstacles: Nx3 numpy array where each row is [x, y, radius].
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

    def refine(self, iterations: int = 1) -> bool:
        """Refine the current trajectory (optional for some planners).

        Default implementation does nothing. Override for optimization-based planners.

        Args:
            iterations: how many iterations to take during a single refinement.
        """
        return True

    def get_trajectory(self) -> NDArray[np.floating] | None:
        """Get the current planned trajectory.

        Returns:
            Nx4 numpy array where each row is [x, y, theta, dt], or None if
            no trajectory has been computed yet.
        """
        return self.plan()

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
