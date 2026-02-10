import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan, NavSatFix
from geometry_msgs.msg import TwistStamped
import numpy as np
import threading


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()

        self.velocity_publisher = self.create_publisher(
            TwistStamped, "/husky/cmd_vel", 10
        )

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/husky/sensors/lidar2d_0/scan",
            self.lidar_callback,
            10,
            callback_group=self.sensor_callback_group,
        )

        self.gps_subscription = self.create_subscription(
            NavSatFix,
            "/husky/sensors/gps_0/fix", 
            self.gps_callback,
            10,
            callback_group=self.sensor_callback_group,
        )

        self.control_timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.control_callback_group
        )

        self.latest_scan = None
        self.last_scan_time = None
        
        self.latest_gps = None
        
        self.data_lock = threading.Lock()

        self.get_logger().info("Controller initialized with Lidar and GPS")

    def lidar_callback(self, msg):
        with self.data_lock:
            self.latest_scan = msg
            self.last_scan_time = self.get_clock().now()
        self.get_logger().info("Lidar data received", throttle_duration_sec=2.0)

    def gps_callback(self, msg):
        with self.data_lock:
            self.latest_gps = msg
        self.get_logger().info("GPS data received", throttle_duration_sec=2.0)

    def control_loop(self):
        with self.data_lock:
            scan = self.latest_scan
            scan_time = self.last_scan_time
            gps_data = self.latest_gps

        twist = TwistStamped()

        if scan is None:
            self.get_logger().warn("Waiting for sensors...", throttle_duration_sec=2.0)
            return

        scan_age = (self.get_clock().now() - scan_time).nanoseconds / 1e9
        if scan_age > 0.5:
            self.get_logger().error(
                "Lidar data stale, stopping", throttle_duration_sec=1.0
            )
            twist.twist.linear.x = 0.0
            twist.twist.angular.z = 0.0
            self.velocity_publisher.publish(twist)
            return

        if gps_data:
            if gps_data.status.status >= 0:
                self.get_logger().info(
                    f"Current Loc: Lat: {gps_data.latitude:.5f}, Lon: {gps_data.longitude:.5f}",
                    throttle_duration_sec=1.0
                )
            else:
                self.get_logger().warn("GPS received but no fix acquired", throttle_duration_sec=2.0)

        twist.twist.linear.x = 1.0
        twist.twist.angular.z = 0.4 

        self.velocity_publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    controller_node = ControllerNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
