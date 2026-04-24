import math

USE_GPS: bool = False
DEBUG: bool = True

SIM: bool = True
if SIM:
    TOPICS: dict[str, str] = {
        "cmd_vel": "/husky/cmd_vel",
        "scan": "/husky/sensors/lidar2d_0/scan",
        "gps": "/husky/sensors/gps_0/fix",
        "odom": "/husky/odometry/global",
    }
else:
    TOPICS: dict[str, str] = {
        "cmd_vel": "/platform/cmd_vel",
        "scan": "/scan",
        "gps": "/todo",
        "odom": "/platform/odom/filtered",
    }

HEARTBEAT_TIMEOUT_SEC: float = 5.0

# MAX_PATH_EDGE: int = 2
MAX_TRAJECTORY_DISTANCE: int = 7
SAFETY_RADIUS: float = 1.3

MIN_CIRCLE_RADIUS: float = 0.2
MAX_CIRCLE_RADIUS: float = 3.5
MAX_LINE_DISTANCE: float = 0.2
CLUSTER_BREAK_DISTANCE: float = 1.4
GEOMETRY_SPLIT_THRESHOLD: float = 2.5
MIN_SCAN_RANGE: float = 0.33
MEDIAN_FILTER_SIZE: int = 3

# TEB limits
MAX_V: float = 0.5
MAX_OMEGA: float = 0.75
MAX_A: float = 0.1
INITIAL_STEP: float = 1.8
TRAJECTORY_LIMITS: tuple[float, float] = (0.5, 2.0)  # min and max sizes of a segment

# TEB params
TEB_WEIGHTS: dict[str, float] = {
    "velocity": 30.0,
    "angular_velocity": 30.0,
    "acceleration": 30.0,
    "angular_acceleration": 30.0,
    "kinematic": 1000.0,
    "time": 1.0,
    "circle_obstacles": 200.0,
    "line_obstacles": 400.0,
}
SOFTMIN_ALPHA: float = -7.0  # SegmentLineObstaclesCost softmin
MAX_TEB_ITERATIONS: int = 5

BARRIER_OFFSET: float = 2

# E-Stop limits
CRITICAL_STOP_RADIUS: float = 0.5
MAX_PITCH: float = math.radians(15)
MAX_ROLL: float = math.radians(15)
STUCK_TIMEOUT: float = 4.0
STUCK_VEL_TOLERANCE: float = 0.05
MAX_CROSS_TRACK_ERROR: float = 5.5
TRAJECTORY_COLLISION_LOOKAHEAD: int = 5
MAX_ODOM_JUMP_SPEED_LINEAR: float = 3.0
MAX_ODOM_JUMP_SPEED_ANGULAR: float = 3.0
MAX_ESTOP_V: float = 1.5
MAX_ESTOP_OMEGA: float = 2.2

OSM_RELATION_ID: int = 5423208

PAVED_SURFACES: set[str] = {
    "paved",
    "asphalt",
    "concrete",
    "paving_stones",
    "sett",
    "cobblestone",
}
EXCLUDED_HIGHWAY_TYPES: set[str] = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "cycleway",
    "busway",
}
EXCLUDED_ACCESS: set[str] = {"no", "private"}
