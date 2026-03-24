from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import LaserScan

from goof_an_odd_husky.helpers import point_segment_distance
from goof_an_odd_husky_common.obstacles import Obstacle, CircleObstacle, LineObstacle


class ObstacleExtractor(ABC):
    @abstractmethod
    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]: ...


class CircleExtractor(ObstacleExtractor):
    def __init__(self, min_radius: float = 0.1, bias_factor: float = 0.2):
        self.min_radius = min_radius
        self.bias_factor = bias_factor

    def extract(self, clusters: list[NDArray[np.floating]]) -> list[Obstacle]:
        obstacles = []
        for cluster in clusters:
            if len(cluster) == 0:
                continue

            center = np.mean(cluster, axis=0)
            if len(cluster) == 1:
                obstacles.append(
                    CircleObstacle(x=center[0], y=center[1], radius=self.min_radius)
                )
                continue

            dists = np.linalg.norm(cluster - center, axis=1)
            radius = float(np.max(dists))

            dist_to_origin = np.linalg.norm(center)
            if dist_to_origin > 1e-6:
                direction = center / dist_to_origin
                center = center + direction * radius * self.bias_factor

            final_radius = float(np.max(np.linalg.norm(cluster - center, axis=1)))
            obstacles.append(
                CircleObstacle(
                    x=center[0], y=center[1], radius=max(final_radius, self.min_radius)
                )
            )

        return obstacles


class LineExtractor(ObstacleExtractor):
    def __init__(self, max_distance: float = 0.5):
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
    def __init__(
        self,
        cluster_break_distance: float = 1.5,
        geometry_split_threshold: float = 1.5,
        step: int = 1,
    ):
        self.cluster_break_distance = cluster_break_distance
        self.geometry_split_threshold = geometry_split_threshold
        self.step = step
        self.circle_extractor = CircleExtractor()
        self.line_extractor = LineExtractor()

    def process(self, scan_msg: LaserScan) -> list[Obstacle]:
        points = self._scan_to_cartesian(scan_msg)
        if len(points) == 0:
            return []

        clusters = self._cluster_sequential(points)

        circle_clusters, line_clusters = [], []
        for cluster in clusters:
            if len(cluster) < 2:
                circle_clusters.append(cluster)
                continue
            diff = cluster[:, np.newaxis, :] - cluster[np.newaxis, :, :]
            max_dist = float(np.sqrt(np.max(np.sum(diff**2, axis=-1))))
            (
                line_clusters
                if max_dist >= self.geometry_split_threshold
                else circle_clusters
            ).append(cluster)

        return (
            self.circle_extractor.extract(circle_clusters) if circle_clusters else []
        ) + (self.line_extractor.extract(line_clusters) if line_clusters else [])

    def _scan_to_cartesian(self, scan_msg: LaserScan) -> NDArray[np.floating]:
        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment

        if self.step > 1:
            ranges = ranges[:: self.step]
            angles = angles[:: self.step]

        valid = np.isfinite(ranges)
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

        CD_x, CD_y = self.line_dx - self.line_cx, self.line_dy - self.line_cy
        cp1 = CD_x * (A_y - self.line_cy) - CD_y * (A_x - self.line_cx)
        cp2 = CD_x * (B_y - self.line_cy) - CD_y * (B_x - self.line_cx)
        diff_side_CD = (cp1 * cp2) <= 0.0

        AB_x, AB_y = B_x - A_x, B_y - A_y
        cp3 = AB_x * (self.line_cy - A_y) - AB_y * (self.line_cx - A_x)
        cp4 = AB_x * (self.line_dy - A_y) - AB_y * (self.line_dx - A_x)
        diff_side_AB = (cp3 * cp4) <= 0.0

        mask = (diff_side_CD & diff_side_AB) | (min_dist <= (self.safety_radius + 0.1))

        return [obs for i, obs in enumerate(self.line_obstacles) if mask[i]]
