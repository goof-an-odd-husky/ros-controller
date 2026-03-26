SIM = True

if SIM:
    TOPICS = {
        "cmd_vel": "/husky/cmd_vel",
        "scan": "/husky/sensors/lidar2d_0/scan",
        "gps": "/husky/sensors/gps_0/fix",
        "odom": "/husky/odometry/global",
    }
else:
    TOPICS = {
        "cmd_vel": "/platform/cmd_vel",
        "scan": "/scan",
        "gps": "/todo",
        "odom": "/odometry/filtered",
    }

OSM_RELATION_ID = 5423208

MAX_PATH_EDGE = 5
MAX_TRAJECTORY_DISTANCE = 13
