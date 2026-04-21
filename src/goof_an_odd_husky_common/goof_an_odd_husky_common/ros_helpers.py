from geometry_msgs.msg import PoseStamped
from goof_an_odd_husky_common.helpers import yaw_to_quat


def make_pose_stamped(
    x: float,
    y: float,
    yaw: float,
    stamp=None,
    frame_id: str = "",
) -> PoseStamped:
    """Construct a PoseStamped message from position and heading.

    Args:
        x: X position in meters.
        y: Y position in meters.
        yaw: Heading in radians.
        stamp: Optional ROS timestamp.
        frame_id: Optional coordinate frame identifier.

    Returns:
        PoseStamped: The populated message.
    """
    msg = PoseStamped()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    q = yaw_to_quat(yaw)
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg
