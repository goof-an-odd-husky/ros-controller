import threading

import numpy as np
from numpy.typing import NDArray

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String, Empty, Header

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from goof_an_odd_husky_msgs.msg import ObstacleArray
from goof_an_odd_husky_common.obstacles import msg_to_obstacles
from goof_an_odd_husky_common.qos import LATCHED_QOS
from goof_an_odd_husky_common.config import USE_GPS
from goof_an_odd_husky_common.helpers import quat_to_yaw
from goof_an_odd_husky_viz.visualizer import TrajectoryVisualizer


class VisualizerNode(Node):
    data_lock: threading.Lock
    pending_robot_pose: list[float] | None
    pending_trajectory: NDArray[np.float64] | None
    pending_global_path: NDArray[np.float64] | None
    pending_obstacles: list | None
    pending_goal_local: list[float] | None
    visualizer: TrajectoryVisualizer
    use_gps: bool

    def __init__(self, use_gps: bool = True) -> None:
        super().__init__("visualizer_node")
        self.use_gps = use_gps
        self.data_lock = threading.Lock()

        self.pending_robot_pose = None
        self.pending_trajectory = None
        self.pending_global_path = None
        self.pending_obstacles = None
        self.pending_goal_local = None

        incoming_cb_group = MutuallyExclusiveCallbackGroup()

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/viz/robot_pose",
            self._on_pose,
            10,
            callback_group=incoming_cb_group,
        )
        self.trajectory_sub = self.create_subscription(
            Path,
            "/viz/trajectory",
            self._on_trajectory,
            10,
            callback_group=incoming_cb_group,
        )
        self.global_path_sub = self.create_subscription(
            Path,
            "/viz/global_path",
            self._on_global_path,
            LATCHED_QOS,
            callback_group=incoming_cb_group,
        )
        self.obstacles_sub = self.create_subscription(
            ObstacleArray,
            "/viz/obstacles",
            self._on_obstacles,
            10,
            callback_group=incoming_cb_group,
        )
        self.goal_local_sub = self.create_subscription(
            PointStamped,
            "/viz/goal_local",
            self._on_goal_local,
            10,
            callback_group=incoming_cb_group,
        )
        self.status_sub = self.create_subscription(
            String,
            "/nav/status",
            self._on_status,
            LATCHED_QOS,
            callback_group=incoming_cb_group,
        )

        self.goal_publisher = self.create_publisher(
            PoseStamped, "/nav/goal", LATCHED_QOS
        )
        self.cancel_publisher = self.create_publisher(Header, "/nav/cancel", LATCHED_QOS)
        self.heartbeat_publisher = self.create_publisher(Empty, "/viz/heartbeat", 10)

        self.heartbeat_timer = self.create_timer(1.0, self._publish_heartbeat)

        self.visualizer = TrajectoryVisualizer(
            x_lim=(-10, 10),
            y_lim=(-4, 16),
            path_render_mode="both",
            interactive_obstacles=False,
            use_gps=use_gps,
            on_goal_set=self._on_goal_set,
            on_cancel=self._on_cancel_pressed,
        )

    def _publish_heartbeat(self) -> None:
        self.heartbeat_publisher.publish(Empty())

    def _on_goal_set(self, x_or_lat: float, y_or_lon: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gps" if self.use_gps else "odom"
        msg.pose.position.x = float(x_or_lat)
        msg.pose.position.y = float(y_or_lon)
        self.goal_publisher.publish(msg)

    def _on_cancel_pressed(self) -> None:
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        self.cancel_publisher.publish(msg)

    def _on_pose(self, msg: PoseStamped) -> None:
        yaw = quat_to_yaw(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        with self.data_lock:
            self.pending_robot_pose = [msg.pose.position.x, msg.pose.position.y, yaw]

    def _on_trajectory(self, msg: Path) -> None:
        traj = np.array(
            [
                [
                    p.pose.position.x,
                    p.pose.position.y,
                    quat_to_yaw(
                        p.pose.orientation.x,
                        p.pose.orientation.y,
                        p.pose.orientation.z,
                        p.pose.orientation.w,
                    ),
                ]
                for p in msg.poses
            ]
        )
        with self.data_lock:
            self.pending_trajectory = traj

    def _on_global_path(self, msg: Path) -> None:
        path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        with self.data_lock:
            self.pending_global_path = path

    def _on_obstacles(self, msg: ObstacleArray) -> None:
        with self.data_lock:
            self.pending_obstacles = msg_to_obstacles(msg)

    def _on_goal_local(self, msg: PointStamped) -> None:
        with self.data_lock:
            self.pending_goal_local = [msg.point.x, msg.point.y, 0.0]

    def _on_status(self, msg: String) -> None:
        self.get_logger().info(f"Controller status: {msg.data}")

    def render_loop(self) -> None:
        if not self.visualizer.is_open:
            return

        with self.data_lock:
            robot_pose = self.pending_robot_pose
            trajectory = self.pending_trajectory
            global_path = self.pending_global_path
            obstacles = self.pending_obstacles
            goal_local = self.pending_goal_local

        if all(
            v is None
            for v in [robot_pose, trajectory, obstacles, goal_local, global_path]
        ):
            return

        self.visualizer.update_world_state(
            robot_pose=robot_pose or [0.0, 0.0, 0.0],
            trajectory=trajectory,
            global_path=global_path,
            obstacles=obstacles,
            start_goal=([0.0, 0.0, 0.0], goal_local)
            if goal_local is not None
            else None,
        )
        self.visualizer.draw()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizerNode(use_gps=USE_GPS)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    qt_timer = QtCore.QTimer()
    qt_timer.timeout.connect(node.render_loop)
    qt_timer.start(100)

    try:
        pg.exec()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
