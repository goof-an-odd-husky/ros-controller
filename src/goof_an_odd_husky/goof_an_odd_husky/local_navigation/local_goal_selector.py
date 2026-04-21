import math
import numpy as np

from rclpy.impl.rcutils_logger import RcutilsLogger
from goof_an_odd_husky.local_navigation.safety import is_point_safe
from goof_an_odd_husky_common.obstacles import Obstacle, LineObstacle
from goof_an_odd_husky_common.config import SAFETY_RADIUS
from shapely.geometry import LineString, Point


class LocalGoalSelector:
    """Finds a safe, immediate local goal point along the global path for the TEB planner.

    Attributes:
        max_trajectory_distance: The maximum horizon distance to plan the trajectory ahead.
        logger: The ROS 2 logger instance.
    """

    max_trajectory_distance: float
    logger: RcutilsLogger

    def __init__(self, max_trajectory_distance: float, logger: RcutilsLogger) -> None:
        """Initialize the LocalGoalSelector.

        Args:
            max_trajectory_distance: Dist in meters defining the lookahead horizon.
            logger: ROS 2 logger instance.
        """
        self.max_trajectory_distance = max_trajectory_distance
        self.logger = logger

    def _transform_point_to_local(
        self,
        gx: float,
        gy: float,
        vehicle_x: float,
        vehicle_y: float,
        yaw: float,
    ) -> tuple[float, float]:
        """Transform a single point from global to robot's local frame.

        Args:
            gx: Global x coordinate.
            gy: Global y coordinate.
            vehicle_x: Vehicle's x position (global frame).
            vehicle_y: Vehicle's y position (global frame).
            yaw: Vehicle's yaw orientation.

        Returns:
            tuple[float, float]: The (lx, ly) coordinates in the robot's local frame.
        """
        c, s = np.cos(-yaw), np.sin(-yaw)
        dx = gx - vehicle_x
        dy = gy - vehicle_y
        return dx * c - dy * s, dx * s + dy * c

    def _transform_coords_to_local(
        self,
        coords: list[tuple[float, float]],
        vehicle_x: float,
        vehicle_y: float,
        yaw: float,
    ) -> list[tuple[float, float]]:
        """Transform a list of coordinates from global to robot's local frame.

        Args:
            coords: List of (x, y) coordinates in global frame.
            vehicle_x: Vehicle's x position (global frame).
            vehicle_y: Vehicle's y position (global frame).
            yaw: Vehicle's yaw orientation.

        Returns:
            list[tuple[float, float]]: List of (lx, ly) coordinates in the robot's local frame.
        """
        return [
            self._transform_point_to_local(gx, gy, vehicle_x, vehicle_y, yaw)
            for gx, gy in coords
        ]

    def select_local_goal(
        self,
        path: list[tuple[float, float]],
        start_index: int,
        vehicle_x: float,
        vehicle_y: float,
        yaw: float,
        detected_obstacles: list[Obstacle],
    ) -> tuple[tuple[float, float], int, int] | None:
        """Select a safe local goal from the global path.

        Args:
            path: The global path in local coordinates.
            start_index: The index to start searching from (cached closest index).
            vehicle_x: Vehicle's x position.
            vehicle_y: Vehicle's y position.
            yaw: Vehicle's yaw orientation.
            detected_obstacles: List of detected obstacles in local frame.

        Returns:
            tuple[tuple[float, float], int, int] | None: A tuple containing the safe
            local goal (x, y), the new closest index and the target index. None if no valid goal could be found.
        """
        if not path:
            return None

        min_dist = float("inf")
        search_window_start = max(start_index - 3, 0)
        search_window_end = min(len(path), start_index + 8)
        closest_idx = search_window_start

        for i in range(search_window_start, search_window_end):
            px, py = path[i]
            dist = math.hypot(px - vehicle_x, py - vehicle_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        accumulated_dist = 0.0

        for i in range(closest_idx, len(path) - 1):
            px1, py1 = path[i]
            px2, py2 = path[i + 1]
            segment_dist = math.hypot(px2 - px1, py2 - py1)
            accumulated_dist += segment_dist

            if accumulated_dist >= self.max_trajectory_distance:
                target_idx = i + 1
                break
        else:
            target_idx = len(path) - 1

        local_x, local_y = self._transform_point_to_local(
            path[target_idx][0], path[target_idx][1], vehicle_x, vehicle_y, yaw
        )

        if not is_point_safe(
            local_x, local_y, detected_obstacles, margin=SAFETY_RADIUS
        ):
            max_search_offset = 8
            found_safe = False

            for offset in range(1, max_search_offset + 1):
                idx_ahead = target_idx + offset
                if idx_ahead < len(path):
                    lx, ly = self._transform_point_to_local(
                        path[idx_ahead][0],
                        path[idx_ahead][1],
                        vehicle_x,
                        vehicle_y,
                        yaw,
                    )
                    if is_point_safe(lx, ly, detected_obstacles, margin=SAFETY_RADIUS):
                        target_idx = idx_ahead
                        local_x, local_y = lx, ly
                        found_safe = True
                        break

            if not found_safe:
                for offset in range(1, max_search_offset + 1):
                    idx_behind = target_idx - offset
                    if idx_behind >= closest_idx and idx_behind >= 0:
                        lx, ly = self._transform_point_to_local(
                            path[idx_behind][0],
                            path[idx_behind][1],
                            vehicle_x,
                            vehicle_y,
                            yaw,
                        )
                        if is_point_safe(
                            lx, ly, detected_obstacles, margin=SAFETY_RADIUS
                        ):
                            target_idx = idx_behind
                            local_x, local_y = lx, ly
                            found_safe = True
                            break

            if not found_safe:
                self.logger.debug(
                    "Could not slide target point out of obstacle. Checking closest node.",
                    throttle_duration_sec=1.0,
                )
                lx_closest, ly_closest = self._transform_point_to_local(
                    path[closest_idx][0],
                    path[closest_idx][1],
                    vehicle_x,
                    vehicle_y,
                    yaw,
                )
                if is_point_safe(
                    lx_closest, ly_closest, detected_obstacles, margin=SAFETY_RADIUS
                ):
                    local_x, local_y = lx_closest, ly_closest
                    target_idx = closest_idx
                else:
                    self.logger.error("Even the closest point is blocked. Stopping.")
                    return None

        return (local_x, local_y), closest_idx, target_idx

    def generate_corridor_barriers(
        self,
        path: list[tuple[float, float]],
        closest_idx: int,
        target_idx: int,
        vehicle_x: float,
        vehicle_y: float,
        yaw: float,
        barrier_offset: float,
    ) -> tuple[list[LineObstacle], list[LineObstacle]]:
        """Generate continuous left and right barrier line obstacles around a path segment.

        Creates a corridor around the path segment from closest_idx to target_idx,
        with barriers offset by barrier_offset meters on each side. Barriers are
        transformed to the robot's local frame (base_link).

        Args:
            path: The global path as a list of (x, y) coordinates.
            closest_idx: Index of the closest path node to the vehicle.
            target_idx: Index of the target/goal point in the path.
            vehicle_x: Vehicle's x position (global frame).
            vehicle_y: Vehicle's y position (global frame).
            yaw: Vehicle's yaw orientation for coordinate transformation.
            barrier_offset: Distance in meters from path edges to barriers.

        Returns:
            tuple of (left_barriers, right_barriers) where each is a list of
            LineObstacle objects in the robot's local frame.
        """
        if (
            not path
            or closest_idx < 0
            or target_idx >= len(path)
            or closest_idx >= target_idx
        ):
            return [], []

        path_segment = LineString(path[closest_idx : target_idx + 1])

        left_barrier_line = path_segment.parallel_offset(
            barrier_offset, side="left", join_style=2
        )
        right_barrier_line = path_segment.parallel_offset(
            barrier_offset, side="right", join_style=2
        )

        def to_coords(geom) -> list[tuple[float, float]]:
            """Extract coordinate list from Shapely geometry."""
            if geom is None or geom.is_empty:
                return []
            if geom.geom_type == "MultiLineString":
                coords = []
                for line in geom.geoms:
                    coords.extend(list(line.coords))
                return coords
            return list(geom.coords)

        def to_line_obstacles(coords: list[tuple[float, float]]) -> list[LineObstacle]:
            """Convert a polyline to a list of LineObstacle segments."""
            obstacles = []
            for i in range(len(coords) - 1):
                obstacles.append(
                    LineObstacle(
                        x1=coords[i][0],
                        y1=coords[i][1],
                        x2=coords[i + 1][0],
                        y2=coords[i + 1][1],
                    )
                )
            return obstacles

        left_coords = to_coords(left_barrier_line)
        right_coords = to_coords(right_barrier_line)

        vehicle_point = Point(vehicle_x, vehicle_y)
        closest_point = Point(path[closest_idx])
        vehicle_to_closest = LineString([vehicle_point, closest_point])

        if len(left_coords) >= 2:
            first_left_segment = LineString([left_coords[0], left_coords[1]])
            if vehicle_to_closest.crosses(first_left_segment):
                left_coords.pop(0)

        if len(right_coords) >= 2:
            first_right_segment = LineString([right_coords[0], right_coords[1]])
            if vehicle_to_closest.crosses(first_right_segment):
                right_coords.pop(0)

        left_coords = self._transform_coords_to_local(
            left_coords, vehicle_x, vehicle_y, yaw
        )
        right_coords = self._transform_coords_to_local(
            right_coords, vehicle_x, vehicle_y, yaw
        )

        left_barriers = to_line_obstacles(left_coords)
        right_barriers = to_line_obstacles(right_coords)

        return left_barriers, right_barriers
