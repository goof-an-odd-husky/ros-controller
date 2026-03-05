import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import time
import math


class LidarVisualizer(Node):
    def __init__(self):
        super().__init__("lidar_visualizer")
        self.subscription = self.create_subscription(
            LaserScan, "/husky/sensors/lidar2d_0/scan", self.lidar_callback, 10
        )
        self.latest_scan = None
        self.last_print_time = 0
        self.print_interval = 0.5

    def lidar_callback(self, msg):
        self.latest_scan = msg

        self.print_grid()

    def print_grid(self):
        if self.latest_scan is None:
            return
        current_time = time.time()
        if current_time - self.last_print_time <= self.print_interval:
            return
        self.last_print_time = current_time

        grid_size = 50
        grid_range = 20.0
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        center = grid_size // 2
        grid[center][center] = "O"

        angle = self.latest_scan.angle_min
        for distance in self.latest_scan.ranges:
            if self.latest_scan.range_min <= distance <= self.latest_scan.range_max:
                y = distance * math.cos(angle)
                x = distance * math.sin(angle)

                grid_x = int(center - x / grid_range * center)
                grid_y = int(center - y / grid_range * center)

                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_y][grid_x] = "#"

            angle += self.latest_scan.angle_increment

        print("\033[2J\033[H")
        for row in grid:
            print("".join(row))
        print(
            f"\nPoints: {len(self.latest_scan.ranges)}, Range: {grid_range}m, Time: {self.last_print_time}"
        )


def main(args=None):
    rclpy.init(args=args)
    lidar_visualizer = LidarVisualizer()
    rclpy.spin(lidar_visualizer)
    lidar_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
