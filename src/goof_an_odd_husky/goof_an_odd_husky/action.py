import numpy as np

from goof_an_odd_husky_common.helpers import normalize_angle
from goof_an_odd_husky_common.types import Trajectory


def trajectory_to_action(trajectory: Trajectory) -> tuple[float, float]:
    """Converts a trajectory segment into linear and angular velocity commands.

    Args:
        trajectory: An Nx4 array where each row represents [x, y, theta, dt].

    Returns:
        tuple[float, float]: A tuple containing (linear_velocity, angular_velocity).
    """
    if trajectory is None or len(trajectory) < 2:
        return 0.0, 0.0

    x1, y1, th1, dt = trajectory[0]
    x2, y2, th2, _ = trajectory[1]

    if dt < 1e-3:
        return 0.0, 0.0

    dx, dy = x2 - x1, y2 - y1
    direction = 1.0 if (dx * np.cos(th1) + dy * np.sin(th1)) >= 0 else -1.0
    chord = np.hypot(dx, dy)
    arc_angle = normalize_angle(th2 - th1)

    arc_len = (
        chord
        if abs(arc_angle) < 1e-5
        else (chord / (2 * np.sin(arc_angle / 2))) * arc_angle
    )

    return (direction * abs(arc_len) / dt), (arc_angle / dt)
