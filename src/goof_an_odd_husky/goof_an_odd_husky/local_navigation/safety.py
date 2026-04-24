import math
from goof_an_odd_husky_common.obstacles import Obstacle, CircleObstacle, LineObstacle
from goof_an_odd_husky_common.helpers import point_segment_distance, segments_intersect


def is_point_safe(
    local_x: float, local_y: float, obstacles: list[Obstacle], margin: float = 0.4
) -> bool:
    """Checks if a point in the base_link frame is safely away from any obstacle.

    Args:
        local_x: The x coordinate of the point in the base_link frame.
        local_y: The y coordinate of the point in the base_link frame.
        obstacles: A list of detected obstacles.
        margin: The safety margin around the obstacles.

    Returns:
        bool: True if the point is safe, False otherwise.
    """
    if not obstacles:
        return True

    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            if math.hypot(local_x - obs.x, local_y - obs.y) < (obs.radius + margin):
                return False

        elif isinstance(obs, LineObstacle):
            l2 = (obs.x2 - obs.x1) ** 2 + (obs.y2 - obs.y1) ** 2
            if l2 == 0:
                dist = math.hypot(local_x - obs.x1, local_y - obs.y1)
            else:
                t = max(
                    0.0,
                    min(
                        1.0,
                        (
                            (local_x - obs.x1) * (obs.x2 - obs.x1)
                            + (local_y - obs.y1) * (obs.y2 - obs.y1)
                        )
                        / l2,
                    ),
                )
                proj_x = obs.x1 + t * (obs.x2 - obs.x1)
                proj_y = obs.y1 + t * (obs.y2 - obs.y1)
                dist = math.hypot(local_x - proj_x, local_y - proj_y)

            if dist < margin:
                return False

    return True


def is_segment_safe(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    obstacles: list[Obstacle],
    margin: float = 0.4,
) -> bool:
    """Checks if a line segment is free of obstacle collisions and respects clearance margins.

    Tests two conditions per obstacle: exact segment intersection (catches crossing
    segments whose endpoints are far apart) and minimum proximity between the two
    segments (catches near-misses that don't cross).

    Args:
        x1: X coordinate of the segment start point.
        y1: Y coordinate of the segment start point.
        x2: X coordinate of the segment end point.
        y2: Y coordinate of the segment end point.
        obstacles: List of obstacles to check against.
        margin: Minimum required clearance distance from any obstacle boundary.

    Returns:
        True if the segment does not intersect any obstacle and maintains at
        least `margin` distance from all obstacle boundaries, False otherwise.
    """
    min_x, max_x = min(x1, x2) - margin, max(x1, x2) + margin
    min_y, max_y = min(y1, y2) - margin, max(y1, y2) + margin

    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            if (max_x < obs.x - obs.radius or min_x > obs.x + obs.radius or 
                max_y < obs.y - obs.radius or min_y > obs.y + obs.radius):
                continue
            d, _, _, _ = point_segment_distance(obs.x, obs.y, x1, y1, x2, y2)
            if d < obs.radius + margin:
                return False

        elif isinstance(obs, LineObstacle):
            obs_min_x, obs_max_x = min(obs.x1, obs.x2), max(obs.x1, obs.x2)
            obs_min_y, obs_max_y = min(obs.y1, obs.y2), max(obs.y1, obs.y2)
            
            if (max_x < obs_min_x or min_x > obs_max_x or 
                max_y < obs_min_y or min_y > obs_max_y):
                continue
            if segments_intersect(x1, y1, x2, y2, obs.x1, obs.y1, obs.x2, obs.y2):
                return False
            d1, _, _, _ = point_segment_distance(x1, y1, obs.x1, obs.y1, obs.x2, obs.y2)
            d2, _, _, _ = point_segment_distance(x2, y2, obs.x1, obs.y1, obs.x2, obs.y2)
            d3, _, _, _ = point_segment_distance(obs.x1, obs.y1, x1, y1, x2, y2)
            d4, _, _, _ = point_segment_distance(obs.x2, obs.y2, x1, y1, x2, y2)
            if min(d1, d2, d3, d4) < margin:
                return False

    return True

def is_start_segment_safe(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    obstacles: list[Obstacle],
    margin: float = 0.4,
) -> bool:
    """Checks if a line segment originating from the start node is safe.

    If the start node is inside the obstacle margin, the segment is allowed 
    only if it moves away from the obstacle.

    Args:
        start_x: X coordinate of the segment start point.
        start_y: Y coordinate of the segment start point.
        end_x: X coordinate of the segment end point.
        end_y: Y coordinate of the segment end point.
        obstacles: List of obstacles to check against.
        margin: Minimum required clearance distance from any obstacle boundary.

    Returns:
        True if the segment does not intersect any obstacle and maintains at
        least `margin` distance from all obstacle boundaries
        or moves away from the obstacles, False otherwise.
    """
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            d_seg, _, _, _ = point_segment_distance(obs.x, obs.y, start_x, start_y, end_x, end_y)
            
            if d_seg < obs.radius + margin:
                d_start = math.hypot(start_x - obs.x, start_y - obs.y)
                
                if d_start <= obs.radius + 1e-5:
                    return False
                    
                if d_seg < d_start - 1e-5:
                    return False

        elif isinstance(obs, LineObstacle):
            if segments_intersect(start_x, start_y, end_x, end_y, obs.x1, obs.y1, obs.x2, obs.y2):
                return False
                
            d1, _, _, _ = point_segment_distance(start_x, start_y, obs.x1, obs.y1, obs.x2, obs.y2)
            d2, _, _, _ = point_segment_distance(end_x, end_y, obs.x1, obs.y1, obs.x2, obs.y2)
            d3, _, _, _ = point_segment_distance(obs.x1, obs.y1, start_x, start_y, end_x, end_y)
            d4, _, _, _ = point_segment_distance(obs.x2, obs.y2, start_x, start_y, end_x, end_y)
            
            min_dist = min(d1, d2, d3, d4)
            
            if min_dist < margin:
                d_start = d1
                
                if d_start <= 1e-5:
                    return False
                    
                if min_dist < d_start - 1e-5:
                    return False

    return True
