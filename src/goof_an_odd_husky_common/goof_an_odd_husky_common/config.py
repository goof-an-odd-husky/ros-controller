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

MAX_PATH_EDGE: int = 2
MAX_TRAJECTORY_DISTANCE: int = 7
SAFETY_RADIUS: float = 1.0

MIN_CIRCLE_RADIUS: float = 0.2
MAX_CIRCLE_RADIUS: float = 3.5
MAX_LINE_DISTANCE: float = 0.2
CLUSTER_BREAK_DISTANCE: float = 1.5
GEOMETRY_SPLIT_THRESHOLD: float = 2.5
MIN_SCAN_RANGE: float = 0.33
MEDIAN_FILTER_SIZE: int = 3

MAX_V: float = 0.5
MAX_A: float = 0.05
INITIAL_STEP: float = 1.8

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
TRAJECTORY_LIMITS: tuple[float, float] = (0.5, 2.0)
SOFTMIN_ALPHA: float = -7.0 # SegmentLineObstaclesCost softmin

BARRIER_OFFSET: float = 2

MAX_SCAN_AGE_SEC = 0.5
MAX_ODOM_AGE_SEC = 0.5
ODOM_IMPLAUSIBILITY_FACTOR = 4.0 # MAX_V times factor = max velocity before emergency
MAX_CONSECUTIVE_REPLANS = 5
STUCK_POSITION_EPSILON_M = 0.05
STUCK_ASTAR_TRIGGER_SEC = 2.0
STUCK_ESTOP_TRIGGER_SEC = 5.0

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
