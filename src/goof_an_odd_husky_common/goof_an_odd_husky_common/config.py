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

BARRIER_OFFSET: float = 2

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
