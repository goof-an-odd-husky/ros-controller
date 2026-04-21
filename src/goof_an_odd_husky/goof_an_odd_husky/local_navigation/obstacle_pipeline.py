from typing import override
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import LaserScan
from scipy.ndimage import median_filter

from goof_an_odd_husky_common.helpers import point_segment_distance, segments_intersect
from goof_an_odd_husky_common.obstacles import Obstacle, CircleObstacle, LineObstacle


class ObstacleExtractor(ABC):
    """Abstract base class for extracting geometric objects from point clusters."""

    @abstractmethod
    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]:
        """Extract obstacles from a list of point clusters.

        Args:
            clusters: A list of numpy arrays representing clustered point clouds.

        Returns:
            list[Obstacle]: Extracted geometry obstacles.
        """
        ...


class CircleExtractor(ObstacleExtractor):
    """Fits circle primitives to point clusters using the Xavier method.

    Assumes incoming clusters exhibit circular geometry. Uses algebraically
    stable circumcenter calculation. Falls back to Chord-Width estimation
    only for mathematical anomalies.

     Attributes:
        min_radius: The minimum acceptable radius for a circular obstacle.
        max_radius: The maximum acceptable radius for a circular obstacle.
    """

    min_radius: float
    max_radius: float

    def __init__(self, min_radius: float, max_radius: float):
        """Initializes the CircleExtractor.

        Args:
            min_radius: Minimum radius of the fitted circles.
            max_radius: Maximum radius of the fitted circles.
        """
        self.min_radius = min_radius
        self.max_radius = max_radius

    @override
    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]:
        obstacles = []

        for cluster in clusters:
            n_points = len(cluster)

            if n_points == 0:
                continue
            if n_points == 1:
                obstacles.append(
                    CircleObstacle(
                        x=cluster[0][0], y=cluster[0][1], radius=self.min_radius
                    )
                )
                continue
            obstacles.append(self.cluster_to_obstacle(cluster))

        return obstacles

    def cluster_to_obstacle(self, cluster: NDArray[np.floating]) -> CircleObstacle:
        """Robust method for circle extraction.

        Args:
            cluster: A numpy array representing clustered point cloud.

        Returns:
            CircleObstacle: An extracted geometry obstacle.
        """
        diffs = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=-1)
        cluster_width = float(np.max(distances))

        estimated_radius = max(cluster_width / 2.0, self.min_radius)
        estimated_radius = min(estimated_radius, self.max_radius)

        visible_center = cluster[:][len(cluster) // 2]
        dist_to_origin = np.linalg.norm(visible_center)

        if dist_to_origin > 1e-6:
            direction = visible_center / dist_to_origin
            true_center = visible_center + (direction * estimated_radius)
        else:
            true_center = visible_center

        final_dists = np.linalg.norm(cluster - true_center, axis=1)
        final_radius = max(float(np.max(final_dists)), estimated_radius)

        return CircleObstacle(x=true_center[0], y=true_center[1], radius=final_radius)


class LineExtractor(ObstacleExtractor):
    """Fits line primitives to point clusters recursively.

    Attributes:
        max_distance: Max deviation distance to accept a point as belonging to a line segment.
    """

    max_distance: float

    def __init__(self, max_distance: float):
        """Initializes the LineExtractor.

        Args:
            max_distance: Tolerance distance from the mathematical line.
        """
        self.max_distance = max_distance

    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]:
        obstacles = []
        for cluster in clusters:
            if len(cluster) >= 2:
                obstacles.extend(self._fit_recursive(cluster))
        return obstacles

    def _fit_recursive(self, points: NDArray[np.floating]) -> list[Obstacle]:
        if len(points) < 2:
            return []

        p1, p2 = points[0], points[-1]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return []

        vecs = points - p1
        distances = (
            np.abs(vecs[:, 0] * line_vec[1] - vecs[:, 1] * line_vec[0]) / line_len
        )

        max_idx = int(np.argmax(distances))
        if distances[max_idx] > self.max_distance and len(points) > 2:
            return self._fit_recursive(points[: max_idx + 1]) + self._fit_recursive(
                points[max_idx:]
            )

        return [LineObstacle(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])]


class ObstaclePipeline:
    """End-to-end pipeline processing raw LaserScans into geometric primitives.

    Attributes:
        cluster_break_distance: Gap threshold in meters to slice scans into clusters.
        geometry_split_threshold: Threshold to decide whether to map a cluster as a Line or Circle.
        min_range: Minimum range in meters; points closer than this are discarded.
        median_filter_size: The window size for median filter run on a scan.
        circle_extractor: Instance of CircleExtractor.
        line_extractor: Instance of LineExtractor.
    """

    cluster_break_distance: float
    geometry_split_threshold: float
    min_range: float
    circle_extractor: CircleExtractor
    line_extractor: LineExtractor
    median_filter_size: int

    def __init__(
        self,
        cluster_break_distance: float,
        geometry_split_threshold: float,
        min_range: float,
        median_filter_size: int,
        min_circle_radius: float,
        max_circle_radius: float,
        max_line_distance: float,
    ):
        """Initialize pipeline parameters.

        Args:
            cluster_break_distance: Points farther apart than this start a new cluster.
            geometry_split_threshold: Clusters larger than this are fitted as lines.
            min_range: Minimum distance from sensor; points closer than this value (in meters) are filtered out.
            median_filter_size: The window size for median filter run on a scan.
            min_circle_radius: The lower bound for the circle obstacles.
            max_circle_radius: The upper bound for the circle obstacles.
            max_line_distance: The upper bound before a line is split in two.
        """
        self.cluster_break_distance = cluster_break_distance
        self.geometry_split_threshold = geometry_split_threshold
        self.min_range = min_range
        self.median_filter_size = median_filter_size
        self.circle_extractor = CircleExtractor(min_circle_radius, max_circle_radius)
        self.line_extractor = LineExtractor(max_line_distance)

    def process(self, scan_msg: LaserScan, is_sim: bool = True) -> list[Obstacle]:
        points = self._scan_to_cartesian(scan_msg, not is_sim)
        if len(points) == 0:
            return []
        points = self._median_filter(points)

        clusters = self._cluster_sequential(points)

        circle_clusters, line_clusters = [], []
        for cluster in clusters:
            n_points = len(cluster)

            if n_points < 3:
                circle_clusters.append(cluster)
                continue

            diff = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
            max_dist = float(np.sqrt(np.max(np.sum(diff**2, axis=-1))))

            if max_dist >= self.geometry_split_threshold:
                line_clusters.append(cluster)
                continue

            p1 = cluster[0]
            p3 = cluster[-1]
            p2 = cluster[n_points // 2]

            chord_vector = p3 - p1
            chord_len = float(np.linalg.norm(chord_vector))

            if chord_len < 1e-6:
                circle_clusters.append(cluster)
                continue

            cross_prod = (p3[0] - p1[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (
                p3[1] - p1[1]
            )
            arc_depth = abs(cross_prod) / chord_len

            if 0.1 * chord_len < arc_depth < 0.7 * chord_len:
                circle_clusters.append(cluster)
            else:
                line_clusters.append(cluster)

        return (
            self.circle_extractor.extract(circle_clusters) if circle_clusters else []
        ) + (self.line_extractor.extract(line_clusters) if line_clusters else [])

    def _scan_to_cartesian(
        self, scan_msg: LaserScan, invert: bool = False
    ) -> NDArray[np.floating]:
        ranges = np.array(scan_msg.ranges)
        if invert:
            ranges = ranges[::-1]
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment

        valid = np.isfinite(ranges) & (ranges >= self.min_range)
        return np.column_stack(
            (
                ranges[valid] * np.cos(angles[valid]),
                ranges[valid] * np.sin(angles[valid]),
            )
        )

    def _cluster_sequential(
        self, points: NDArray[np.floating]
    ) -> list[NDArray[np.floating]]:
        if len(points) < 2:
            return [points]
        breakpoints = (
            np.where(
                np.linalg.norm(points[1:] - points[:-1], axis=1)
                > self.cluster_break_distance
            )[0]
            + 1
        )
        return np.split(points, breakpoints)

    def _median_filter(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        if len(points) < self.median_filter_size:
            return points

        return median_filter(points, size=(self.median_filter_size, 1))


class ObstacleFilter:
    """Pre-filters geometric obstacles to supply only relevant ones to the TEB optimization step.

    Attributes:
        safety_radius: Dist margin applied to obstacle checks.
        circle_obstacles: All known circles.
        line_obstacles: All known lines.
        circ_x, circ_y, circ_r: Numpy arrays representing circle parameters.
        line_cx, line_cy, line_dx, line_dy: Numpy arrays representing line segment parameters.
    """

    safety_radius: float
    circle_obstacles: list[CircleObstacle]
    line_obstacles: list[LineObstacle]

    circ_x: NDArray[np.float64]
    circ_y: NDArray[np.float64]
    circ_r: NDArray[np.float64]

    line_cx: NDArray[np.float64]
    line_cy: NDArray[np.float64]
    line_dx: NDArray[np.float64]
    line_dy: NDArray[np.float64]

    def __init__(
        self,
        circle_obstacles: list[CircleObstacle],
        line_obstacles: list[LineObstacle],
        safety_radius: float,
    ):
        """Initializes the filter with precomputed numpy vectors for fast distance checking.

        Args:
            circle_obstacles: The full list of circular obstacles.
            line_obstacles: The full list of line segment obstacles.
            safety_radius: Safety padding radius.
        """
        self.safety_radius = safety_radius
        self.circle_obstacles = circle_obstacles
        self.line_obstacles = line_obstacles

        if circle_obstacles:
            self.circ_x = np.array([o.x for o in circle_obstacles], dtype=np.float64)
            self.circ_y = np.array([o.y for o in circle_obstacles], dtype=np.float64)
            self.circ_r = np.array(
                [o.radius for o in circle_obstacles], dtype=np.float64
            )

        if line_obstacles:
            self.line_cx = np.array([o.x1 for o in line_obstacles], dtype=np.float64)
            self.line_cy = np.array([o.y1 for o in line_obstacles], dtype=np.float64)
            self.line_dx = np.array([o.x2 for o in line_obstacles], dtype=np.float64)
            self.line_dy = np.array([o.y2 for o in line_obstacles], dtype=np.float64)

    def get_close_circles(
        self, A_x: float, A_y: float, B_x: float, B_y: float
    ) -> list[CircleObstacle]:
        """Filters circular obstacles intersecting a segment or positioned near it.

        Args:
            A_x, A_y: Start point of segment.
            B_x, B_y: End point of segment.

        Returns:
            list[CircleObstacle]: Filtered subset of close circles.
        """
        if not self.circle_obstacles:
            return []

        AB_x, AB_y = B_x - A_x, B_y - A_y
        AB_len_sq = max(AB_x**2 + AB_y**2, 1e-10)

        AO_x, AO_y = self.circ_x - A_x, self.circ_y - A_y
        t = np.clip((AO_x * AB_x + AO_y * AB_y) / AB_len_sq, 0.0, 1.0)

        dist = np.sqrt((AO_x - t * AB_x) ** 2 + (AO_y - t * AB_y) ** 2)
        mask = dist <= (self.circ_r + self.safety_radius + 0.1)

        return [obs for i, obs in enumerate(self.circle_obstacles) if mask[i]]

    def get_close_lines(
        self, A_x: float, A_y: float, B_x: float, B_y: float
    ) -> list[LineObstacle]:
        """Filters line obstacles intersecting a segment or positioned near it.

        Args:
            A_x, A_y: Start point of segment.
            B_x, B_y: End point of segment.

        Returns:
            list[LineObstacle]: Filtered subset of close lines.
        """
        if not self.line_obstacles:
            return []

        d1, *_ = point_segment_distance(
            A_x, A_y, self.line_cx, self.line_cy, self.line_dx, self.line_dy
        )
        d2, *_ = point_segment_distance(
            B_x, B_y, self.line_cx, self.line_cy, self.line_dx, self.line_dy
        )
        d3, *_ = point_segment_distance(self.line_cx, self.line_cy, A_x, A_y, B_x, B_y)
        d4, *_ = point_segment_distance(self.line_dx, self.line_dy, A_x, A_y, B_x, B_y)

        min_dist = np.min(np.vstack([d1, d2, d3, d4]), axis=0)
        intersect = segments_intersect(
            A_x, A_y, B_x, B_y, self.line_cx, self.line_cy, self.line_dx, self.line_dy
        )
        mask = intersect | (min_dist <= (self.safety_radius + 0.1))

        return [obs for i, obs in enumerate(self.line_obstacles) if mask[i]]
