import math
from goof_an_odd_husky_common.obstacles import Obstacle, CircleObstacle, LineObstacle


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
