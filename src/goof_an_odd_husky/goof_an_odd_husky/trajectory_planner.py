from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import LaserScan


class Obstacle(ABC):
    """Abstract base class for all obstacle types."""

    pass


@dataclass
class CircleObstacle(Obstacle):
    """Represents a circular obstacle."""

    x: float
    y: float
    radius: float


@dataclass
class LineObstacle(Obstacle):
    """Represents a line segment obstacle, defined by two endpoints."""

    x1: float
    y1: float
    x2: float
    y2: float


class ObstacleExtractor(ABC):
    """Abstract base class for extracting shapes from grouped clusters."""

    @abstractmethod
    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]:
        """Convert a list of point clusters into a list of Obstacles.

        Args:
            clusters: A list of Nx2 numpy arrays representing grouped points.

        Returns:
            A list of instantiated Obstacle objects.
        """
        ...


class CircleExtractor(ObstacleExtractor):
    """Extracts circular obstacles from point clusters, biased away from the vehicle."""

    def __init__(self, min_radius: float = 0.1, bias_factor: float = 0.2):
        self.min_radius = min_radius
        self.bias_factor = bias_factor

    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]:
        """
        Fits a bounding circle to each cluster, biasing the center away from
        the vehicle (origin) to better cover the unobserved backside of obstacles.
        """
        obstacles = []
        for cluster in clusters:
            if len(cluster) == 0:
                continue

            center = np.mean(cluster, axis=0)
            if len(cluster) == 1:
                radius = self.min_radius
                obstacles.append(
                    CircleObstacle(x=center[0], y=center[1], radius=radius)
                )
                continue

            dists = np.linalg.norm(cluster - center, axis=1)
            radius = float(np.max(dists))

            dist_to_center = np.linalg.norm(center)
            if dist_to_center > 1e-6:
                direction = center / dist_to_center
                shift_amount = radius * self.bias_factor
                center = center + direction * shift_amount

            final_dists = np.linalg.norm(cluster - center, axis=1)
            final_radius = float(np.max(final_dists))

            obstacles.append(
                CircleObstacle(
                    x=center[0], y=center[1], radius=max(final_radius, self.min_radius)
                )
            )

        return obstacles


class LineExtractor(ObstacleExtractor):
    """Extracts line obstacles from point clusters using recursive split-and-merge."""

    def __init__(self, max_distance: float = 0.5):
        self.max_distance = max_distance

    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]:
        """
        Fits lines to clusters by recursively splitting segments
        where the point-to-line distance exceeds max_distance.
        """
        obstacles = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue

            obstacles.extend(self._fit_recursive(cluster))

        return obstacles

    def _fit_recursive(self, points: NDArray[np.floating]) -> list[Obstacle]:
        if len(points) < 2:
            return []

        p1 = points[0]
        p2 = points[-1]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return []

        vecs = points - p1
        cross_products = vecs[:, 0] * line_vec[1] - vecs[:, 1] * line_vec[0]
        distances = np.abs(cross_products) / line_len

        max_idx = np.argmax(distances)
        max_dist = float(distances[max_idx])

        if max_dist > self.max_distance and len(points) > 2:
            left_lines = self._fit_recursive(points[: max_idx + 1])
            right_lines = self._fit_recursive(points[max_idx:])
            return left_lines + right_lines
        else:
            return [LineObstacle(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])]


class ObstaclePipeline:
    """Orchestrates the conversion of raw LaserScans into geometric obstacles."""

    def __init__(
        self,
        cluster_break_distance: float = 1.5,
        geometry_split_threshold: float = 1.5,
        step: int = 1,
    ):
        """
        Args:
            cluster_break_distance: Max distance between adjacent LIDAR beams to be in the same cluster.
            geometry_split_threshold: The max internal distance (X meters) to split circles vs lines.
            step: Downsampling step for the raw LIDAR data.
        """
        self.cluster_break_distance = cluster_break_distance
        self.geometry_split_threshold = geometry_split_threshold
        self.step = step

        self.circle_extractor = CircleExtractor()
        self.line_extractor = LineExtractor()

    def process(self, scan_msg: LaserScan) -> list[Obstacle]:
        """Main pipeline: Scan -> Cartesian -> Clusters -> Split -> Extract."""
        points = self._scan_to_cartesian(scan_msg)
        if len(points) == 0:
            return []

        clusters = self._extract_clusters_sequential(points)

        circle_clusters = []
        line_clusters = []

        for cluster in clusters:
            if len(cluster) < 2:
                circle_clusters.append(cluster)
                continue

            diff_matrix = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
            sq_dists = np.sum(diff_matrix**2, axis=-1)
            max_dist = float(np.sqrt(np.max(sq_dists)))

            if max_dist < self.geometry_split_threshold:
                circle_clusters.append(cluster)
            else:
                line_clusters.append(cluster)

        obstacles: list[Obstacle] = []
        if circle_clusters:
            obstacles.extend(self.circle_extractor.extract(circle_clusters))
        if line_clusters:
            obstacles.extend(self.line_extractor.extract(line_clusters))

        return obstacles

    def _scan_to_cartesian(self, scan_msg: LaserScan) -> NDArray[np.floating]:
        """Converts polar LIDAR scan to Cartesian points efficiently."""
        ranges = np.array(scan_msg.ranges)

        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment

        if self.step > 1:
            ranges = ranges[:: self.step]
            angles = angles[:: self.step]

        valid_indices = np.isfinite(ranges)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]

        xs = valid_ranges * np.cos(valid_angles)
        ys = valid_ranges * np.sin(valid_angles)

        return np.column_stack((xs, ys))

    def _extract_clusters_sequential(
        self, points: NDArray[np.floating]
    ) -> list[NDArray[np.floating]]:
        """
        Exploits the sequential nature of LIDAR data to cluster in O(N) time.
        Splits the array wherever the distance between adjacent points exceeds the threshold.
        """
        if len(points) < 2:
            return [points]

        diffs = np.linalg.norm(points[1:] - points[:-1], axis=1)

        breakpoints = np.where(diffs > self.cluster_break_distance)[0] + 1

        return np.split(points, breakpoints)


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
