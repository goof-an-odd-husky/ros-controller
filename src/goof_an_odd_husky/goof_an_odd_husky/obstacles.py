from goof_an_odd_husky.helpers import point_segment_distance
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


class ObstacleFilter:
    def __init__(
        self,
        circle_obstacles: list[CircleObstacle],
        line_obstacles: list[LineObstacle],
        safety_radius: float,
    ):
        self.safety_radius = safety_radius

        self.circle_obstacles = circle_obstacles
        self.line_obstacles = line_obstacles

        if self.circle_obstacles:
            self.circ_x = np.array([o.x for o in circle_obstacles], dtype=np.float64)
            self.circ_y = np.array([o.y for o in circle_obstacles], dtype=np.float64)
            self.circ_r = np.array(
                [o.radius for o in circle_obstacles], dtype=np.float64
            )

        if self.line_obstacles:
            self.line_cx = np.array([o.x1 for o in line_obstacles], dtype=np.float64)
            self.line_cy = np.array([o.y1 for o in line_obstacles], dtype=np.float64)
            self.line_dx = np.array([o.x2 for o in line_obstacles], dtype=np.float64)
            self.line_dy = np.array([o.y2 for o in line_obstacles], dtype=np.float64)

    def get_close_circles(self, A_x: float, A_y: float, B_x: float, B_y: float):
        if not self.circle_obstacles:
            return []

        AB_x = B_x - A_x
        AB_y = B_y - A_y
        AB_len_sq = max(AB_x**2 + AB_y**2, 1e-10)

        AO_x = self.circ_x - A_x
        AO_y = self.circ_y - A_y

        t = (AO_x * AB_x + AO_y * AB_y) / AB_len_sq
        t = np.clip(t, 0.0, 1.0)

        O1O_x = AO_x - t * AB_x
        O1O_y = AO_y - t * AB_y

        dist = np.sqrt(O1O_x**2 + O1O_y**2)

        mask = dist <= (self.circ_r + self.safety_radius + 0.1)

        return [obs for i, obs in enumerate(self.circle_obstacles) if mask[i]]

    def get_close_lines(self, A_x: float, A_y: float, B_x: float, B_y: float):
        if not self.line_obstacles:
            return []

        d1, _, _, _ = point_segment_distance(
            A_x, A_y, self.line_cx, self.line_cy, self.line_dx, self.line_dy
        )
        d2, _, _, _ = point_segment_distance(
            B_x, B_y, self.line_cx, self.line_cy, self.line_dx, self.line_dy
        )
        d3, _, _, _ = point_segment_distance(
            self.line_cx, self.line_cy, A_x, A_y, B_x, B_y
        )
        d4, _, _, _ = point_segment_distance(
            self.line_dx, self.line_dy, A_x, A_y, B_x, B_y
        )

        all_d = np.vstack([d1, d2, d3, d4])
        min_dist = np.min(all_d, axis=0)

        CD_x, CD_y = self.line_dx - self.line_cx, self.line_dy - self.line_cy
        CA_x, CA_y = A_x - self.line_cx, A_y - self.line_cy
        CB_x, CB_y = B_x - self.line_cx, B_y - self.line_cy

        cp1 = CD_x * CA_y - CD_y * CA_x
        cp2 = CD_x * CB_y - CD_y * CB_x
        diff_side_CD = (cp1 * cp2) <= 0.0

        AB_x, AB_y = B_x - A_x, B_y - A_y
        AC_x, AC_y = self.line_cx - A_x, self.line_cy - A_y
        AD_x, AD_y = self.line_dx - A_x, self.line_dy - A_y

        cp3 = AB_x * AC_y - AB_y * AC_x
        cp4 = AB_x * AD_y - AB_y * AD_x
        diff_side_AB = (cp3 * cp4) <= 0.0

        intersect = diff_side_CD & diff_side_AB

        mask = intersect | (min_dist <= (self.safety_radius + 0.1))

        return [obs for i, obs in enumerate(self.line_obstacles) if mask[i]]
