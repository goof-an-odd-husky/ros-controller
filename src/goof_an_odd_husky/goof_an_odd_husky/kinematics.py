import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from goof_an_odd_husky.helpers import normalize_angle


def get_odom_delta(curr: Odometry, prev: Odometry | None) -> tuple[float, float, float]:
    """
    Calculates the delta in local coordinates between two odometry messages.
    """
    if prev is None:
        return 0.0, 0.0, 0.0

    curr_pos, prev_pos = curr.pose.pose.position, prev.pose.pose.position
    curr_q, prev_q = curr.pose.pose.orientation, prev.pose.pose.orientation

    curr_yaw = Rotation.from_quat([curr_q.x, curr_q.y, curr_q.z, curr_q.w]).as_euler(
        "zyx"
    )[0]

    prev_yaw = Rotation.from_quat([prev_q.x, prev_q.y, prev_q.z, prev_q.w]).as_euler(
        "zyx"
    )[0]

    dx_global, dy_global = curr_pos.x - prev_pos.x, curr_pos.y - prev_pos.y
    dtheta = normalize_angle(curr_yaw - prev_yaw)

    cos_p, sin_p = np.cos(prev_yaw), np.sin(prev_yaw)
    dx_local = dx_global * cos_p + dy_global * sin_p
    dy_local = -dx_global * sin_p + dy_global * cos_p

    return dx_local, dy_local, dtheta
