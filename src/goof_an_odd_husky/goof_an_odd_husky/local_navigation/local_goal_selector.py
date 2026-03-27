import math
import numpy as np

from rclpy.impl.rcutils_logger import RcutilsLogger
from goof_an_odd_husky.local_navigation.safety import is_point_safe
from goof_an_odd_husky_common.obstacles import Obstacle


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

    def select_local_goal(
        self,
        path: list[tuple[float, float]],
        start_index: int,
        vehicle_x: float,
        vehicle_y: float,
        yaw: float,
        detected_obstacles: list[Obstacle],
    ) -> tuple[tuple[float, float], int] | None:
        """Select a safe local goal from the global path.

        Args:
            path: The global path in local coordinates.
            start_index: The index to start searching from (cached closest index).
            vehicle_x: Vehicle's x position.
            vehicle_y: Vehicle's y position.
            yaw: Vehicle's yaw orientation.
            detected_obstacles: List of detected obstacles in local frame.

        Returns:
            tuple[tuple[float, float], int] | None: A tuple containing the safe
            local goal (x, y) and the new closest index, or None if no valid goal could be found.
        """
        if not path:
            return None

        min_dist = float("inf")
        closest_idx = max(start_index - 3, 0)
        search_window_end = min(len(path), start_index + 8)

        for i in range(start_index, search_window_end):
            px, py = path[i]
            dist = math.hypot(px - vehicle_x, py - vehicle_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        target_idx = closest_idx
        for i in range(closest_idx, len(path)):
            px, py = path[i]
            dist_from_robot = math.hypot(px - vehicle_x, py - vehicle_y)
            if dist_from_robot >= self.max_trajectory_distance:
                target_idx = i
                break
        else:
            target_idx = len(path) - 1

        if target_idx < 0 or target_idx >= len(path):
            return None

        c, s = np.cos(-yaw), np.sin(-yaw)

        def get_local_coords(idx: int) -> tuple[float, float]:
            """Transform path coordinate to base_link frame."""
            gx, gy = path[idx]
            dx, dy = gx - vehicle_x, gy - vehicle_y
            return dx * c - dy * s, dx * s + dy * c

        local_x, local_y = get_local_coords(target_idx)

        if not is_point_safe(local_x, local_y, detected_obstacles, margin=2.0):
            max_search_offset = 8
            found_safe = False

            for offset in range(1, max_search_offset + 1):
                idx_ahead = target_idx + offset
                if idx_ahead < len(path):
                    lx, ly = get_local_coords(idx_ahead)
                    if is_point_safe(lx, ly, detected_obstacles, margin=0.4):
                        local_x, local_y = lx, ly
                        found_safe = True
                        break

                idx_behind = target_idx - offset
                if idx_behind >= closest_idx:
                    lx, ly = get_local_coords(idx_behind)
                    if is_point_safe(lx, ly, detected_obstacles, margin=0.4):
                        local_x, local_y = lx, ly
                        found_safe = True
                        break

            if not found_safe:
                self.logger.debug(
                    "Could not slide target point out of obstacle. Reverting to closest node.",
                    throttle_duration_sec=2.0,
                )
                local_x, local_y = get_local_coords(closest_idx)

        return (local_x, local_y), closest_idx
