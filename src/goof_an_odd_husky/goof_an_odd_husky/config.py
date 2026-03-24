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
