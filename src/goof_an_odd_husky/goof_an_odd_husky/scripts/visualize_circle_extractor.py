import math
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sensor_msgs.msg import LaserScan
from goof_an_odd_husky.local_navigation.obstacle_pipeline import CircleExtractor, LineExtractor, ObstaclePipeline
from goof_an_odd_husky_common.obstacles import Obstacle, CircleObstacle, LineObstacle


class LidarSimulator:
    """Generates 2D radial distance data with occlusion and noise."""
    def __init__(self, angle_min=-math.pi, angle_max=math.pi, num_rays=360, max_range=10.0, noise_std=0.01):
        self.angle_min = angle_min
        self.angle_increment = (angle_max - angle_min) / num_rays
        self.num_rays = num_rays
        self.max_range = max_range
        self.noise_std = noise_std
        self.shapes = []

    def add_circle(self, x, y, r):
        self.shapes.append(('circle', x, y, r))

    def add_segment(self, x1, y1, x2, y2):
        self.shapes.append(('line', x1, y1, x2, y2))

    def add_polyline(self, points):
        for i in range(len(points) - 1):
            self.add_segment(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

    def scan(self) -> LaserScan:
        scan_msg = LaserScan()
        scan_msg.angle_min = self.angle_min
        scan_msg.angle_increment = self.angle_increment
        scan_msg.ranges = []

        for i in range(self.num_rays):
            angle = self.angle_min + i * self.angle_increment
            dx, dy = math.cos(angle), math.sin(angle)
            min_t = self.max_range

            for shape in self.shapes:
                t = float('inf')
                if shape[0] == 'circle':
                    _, cx, cy, r = shape
                    b = -2 * (dx * cx + dy * cy)
                    c = cx**2 + cy**2 - r**2
                    desc = b**2 - 4*c
                    if desc >= 0:
                        t1 = (-b - math.sqrt(desc)) / 2
                        t2 = (-b + math.sqrt(desc)) / 2
                        if t1 > 0: t = t1
                        elif t2 > 0: t = t2
                
                elif shape[0] == 'line':
                    _, ax, ay, bx, by = shape
                    den = dy * (ax - bx) - dx * (ay - by)
                    if abs(den) > 1e-6:
                        t_intersect = (-ax * (ay - by) + ay * (ax - bx)) / den
                        u = (dx * ay - dy * ax) / den
                        if t_intersect > 0 and 0 <= u <= 1:
                            t = t_intersect

                if t < min_t:
                    min_t = t

            if min_t < self.max_range:
                min_t += np.random.normal(0, self.noise_std)
                scan_msg.ranges.append(min_t)
            else:
                scan_msg.ranges.append(float('inf'))

        return scan_msg


def run_visualizer():
    sim = LidarSimulator(num_rays=720, max_range=15.0, noise_std=0.05)
    
    sim.add_circle(x=3, y=0, r=0.5)
    sim.add_circle(x=2, y=3, r=0.8)
    sim.add_segment(x1=-1, y1=4, x2=4, y2=4)
    sim.add_polyline([[5, 1], [6, -2], [3, -4]])
    sim.add_circle(x=-3, y=-1, r=0.2)

    scan_msg = sim.scan()
    
    pipeline = ObstaclePipeline()
    obstacles = pipeline.process(scan_msg)

    points = pipeline._scan_to_cartesian(scan_msg)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('#1e1e2e')
    fig.patch.set_facecolor('#1e1e2e')
    
    ax.plot(0, 0, marker='^', color='white', markersize=10, label='Robot (LiDAR)')
    ax.scatter(points[:, 0], points[:, 1], s=4, c='#00ffcc', alpha=0.6, label='Raw Scan Points')

    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            circle = Circle((obs.x, obs.y), obs.radius, color='#ff0055', fill=False, linewidth=2, linestyle='--')
            ax.add_patch(circle)
            ax.plot(obs.x, obs.y, marker='+', color='#ff0055', markersize=8)
        elif isinstance(obs, LineObstacle):
            ax.plot([obs.x1, obs.x2], [obs.y1, obs.y2], color='#f9e2af', linewidth=3)

    ax.plot([], [], color='#ff0055', linestyle='--', label='Extracted Circle')
    ax.plot([], [], color='#f9e2af', linewidth=3, label='Extracted Line')

    ax.set_xlim(-5, 8)
    ax.set_ylim(-6, 6)
    ax.grid(color='#313244', linestyle=':', linewidth=1)
    ax.legend(loc='upper left', facecolor='#11111b', edgecolor='none', labelcolor='white')
    ax.set_title("LiDAR Geometry Extraction (Xavier Math + Pipeline Routing)", color='white')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visualizer()
