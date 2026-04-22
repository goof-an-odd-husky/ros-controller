from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, PointStamped
from nav_msgs.msg import Path
from rclpy.publisher import Publisher
from std_msgs.msg import String
from goof_an_odd_husky_msgs.msg import ObstacleArray

from goof_an_odd_husky_common.config import TOPICS
from goof_an_odd_husky_common.qos import LATCHED_QOS
from goof_an_odd_husky_common.ros_helpers import make_pose_stamped
from goof_an_odd_husky_common.types import Pose2D, Trajectory
from goof_an_odd_husky_common.obstacles import Obstacle, obstacles_to_msg


class PublisherCommunicator:
    """Handles all ROS 2 publishers and message formatting for the navigation system.

    Attributes:
        node: The parent ROS 2 node.
        velocity_publisher: Publisher for TwistStamped commands.
        pose_publisher: Publisher for the robot's current PoseStamped.
        trajectory_publisher: Publisher for the local planned trajectory path.
        global_path_publisher: Publisher for the global path in the odom frame.
        obstacles_publisher: Publisher for the detected ObstacleArray.
        goal_local_publisher: Publisher for the current local goal point.
        status_publisher: Publisher for the navigation status string.
    """

    node: Node
    velocity_publisher: Publisher
    pose_publisher: Publisher
    trajectory_publisher: Publisher
    global_path_publisher: Publisher
    obstacles_publisher: Publisher
    goal_local_publisher: Publisher
    status_publisher: Publisher

    def __init__(self, node: Node) -> None:
        """Initialize the PublisherCommunicator with its publishers.

        Args:
            node: The ROS 2 node to attach publishers to.
        """
        self.node = node
        self.velocity_publisher = node.create_publisher(TwistStamped, TOPICS["cmd_vel"], 10)
        self.pose_publisher = node.create_publisher(PoseStamped, "/viz/robot_pose", 10)
        self.trajectory_publisher = node.create_publisher(Path, "/viz/trajectory", 10)
        self.global_path_publisher = node.create_publisher(Path, "/viz/global_path", LATCHED_QOS)
        self.obstacles_publisher = node.create_publisher(ObstacleArray, "/viz/obstacles", 10)
        self.goal_local_publisher = node.create_publisher(PointStamped, "/viz/goal_local", 10)
        self.status_publisher = node.create_publisher(String, "/nav/status", LATCHED_QOS)

    def publish_velocity(self, v: float, omega: float) -> None:
        """Publish the velocity command to the robot base.

        Args:
            v: Linear velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        msg = TwistStamped()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(omega)
        self.velocity_publisher.publish(msg)

    def publish_status(self, status: str) -> None:
        """Publish a text status indicating the robot state.

        Args:
            status: The status string.
        """
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)

    def publish_robot_pose(self, pose: Pose2D) -> None:
        """Publish the robot's pose for visualization.

        Args:
            pose: A Pose2D containing x, y, theta.
        """
        self.pose_publisher.publish(
            make_pose_stamped(
                pose.x, pose.y, pose.theta, self.node.get_clock().now().to_msg(), "odom"
            )
        )

    def publish_trajectory(self, trajectory: Trajectory | None) -> None:
        """Publish the current locally planned trajectory path.

        Args:
            trajectory: An Nx4 numpy array representing the trajectory, or None.
        """
        if trajectory is None or len(trajectory) == 0:
            return
        msg = Path()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        for point in trajectory:
            msg.poses.append(
                make_pose_stamped(float(point[0]), float(point[1]), float(point[2]))
            )
        self.trajectory_publisher.publish(msg)

    def publish_global_path(self, path_local: list[tuple[float, float]] | None) -> None:
        """Publish the global path shifted into the local odometry frame.

        Args:
            path_local: A list of (x, y) coordinates representing the path, or None.
        """
        if not path_local:
            return
        msg = Path()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        for x, y in path_local:
            pose = PoseStamped()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.global_path_publisher.publish(msg)

    def publish_obstacles(self, obstacles: list[Obstacle] | None) -> None:
        """Publish detected obstacles for visualization.

        Args:
            obstacles: A list of geometric obstacles, or None.
        """
        if obstacles is not None:
            self.obstacles_publisher.publish(obstacles_to_msg(obstacles))

    def publish_goal_local(self, goal_local: list[float] | None) -> None:
        """Publish the specific local goal currently targeted by the TEB planner.

        Args:
            goal_local: A list containing [x, y, theta], or None.
        """
        if goal_local is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.point.x, msg.point.y = float(goal_local[0]), float(goal_local[1])
        self.goal_local_publisher.publish(msg)
