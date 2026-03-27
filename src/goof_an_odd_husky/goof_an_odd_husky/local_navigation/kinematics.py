import numpy as np
from nav_msgs.msg import Odometry

from goof_an_odd_husky_common.helpers import normalize_angle, quat_to_yaw
from goof_an_odd_husky_common.types import OdomDelta


def get_odom_delta(curr: Odometry, prev: Odometry | None) -> OdomDelta:
    """Calculates the relative positional change (delta) in local coordinates between two odometry messages.

    Args:
        curr: The current Odometry message.
        prev: The previous Odometry message, if available.

    Returns:
        OdomDelta: The shift in the vehicle's relative frame.
    """
    if prev is None:
        return OdomDelta(0.0, 0.0, 0.0)

    curr_pos, prev_pos = curr.pose.pose.position, prev.pose.pose.position
    curr_q, prev_q = curr.pose.pose.orientation, prev.pose.pose.orientation

    curr_yaw = quat_to_yaw(curr_q.x, curr_q.y, curr_q.z, curr_q.w)
    prev_yaw = quat_to_yaw(prev_q.x, prev_q.y, prev_q.z, prev_q.w)

    dx_global = curr_pos.x - prev_pos.x
    dy_global = curr_pos.y - prev_pos.y
    dtheta = normalize_angle(curr_yaw - prev_yaw)

    cos_p, sin_p = np.cos(prev_yaw), np.sin(prev_yaw)
    dx_local = dx_global * cos_p + dy_global * sin_p
    dy_local = -dx_global * sin_p + dy_global * cos_p

    return OdomDelta(dx_local, dy_local, dtheta)
