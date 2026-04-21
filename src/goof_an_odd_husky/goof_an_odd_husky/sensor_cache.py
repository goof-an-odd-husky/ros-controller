import threading
import rclpy.time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from goof_an_odd_husky_common.types import GpsCoord


class SensorCache:
    """Thread-safe storage for incoming sensor data from ROS 2 subscriptions.

    Attributes:
        _lock: Threading lock for synchronizing sensor callbacks and loop access.
        odom: The most recently received odometry message.
        scan: The most recently received laser scan message.
        scan_time: Timestamp of the last received laser scan.
        gps: The most recently received GPS coordinate.
        first_gps: The first valid GPS anchor coordinate.
        last_heartbeat_time: Timestamp of the last visualizer heartbeat.
    """

    _lock: threading.Lock
    odom: Odometry | None
    scan: LaserScan | None
    scan_time: rclpy.time.Time | None
    gps: GpsCoord | None
    first_gps: GpsCoord | None
    last_heartbeat_time: rclpy.time.Time | None

    def __init__(self) -> None:
        """Initialize the SensorCache with empty values and a lock."""
        self._lock = threading.Lock()
        self.odom = None
        self.scan = None
        self.scan_time = None
        self.gps = None
        self.first_gps = None
        self.last_heartbeat_time = None

    def update_odom(self, msg: Odometry) -> None:
        """Update the latest odometry message.

        Args:
            msg: The incoming Odometry message.
        """
        with self._lock:
            self.odom = msg

    def get_odom(self) -> Odometry | None:
        """Safely retrieve the latest odometry message.

        Returns:
            Odometry | None: The cached message or None.
        """
        with self._lock:
            return self.odom

    def update_scan(self, msg: LaserScan, time: rclpy.time.Time) -> None:
        """Update the latest laser scan and its arrival time.

        Args:
            msg: The incoming LaserScan message.
            time: The ROS clock time at which it was received.
        """
        with self._lock:
            self.scan = msg
            self.scan_time = time

    def update_gps(self, gps_coord: GpsCoord, status_valid: bool) -> None:
        """Update the latest GPS coordinate and capture the first valid anchor.

        Args:
            gps_coord: The converted GpsCoord data.
            status_valid: Boolean indicating if the GPS fix is valid.
        """
        with self._lock:
            self.gps = gps_coord
            if not self.first_gps and status_valid:
                self.first_gps = gps_coord

    def get_first_gps(self) -> GpsCoord | None:
        """Safely retrieve the first valid GPS coordinate anchor.

        Returns:
            GpsCoord | None: The first valid GPS coordinate or None.
        """
        with self._lock:
            return self.first_gps

    def update_heartbeat(self, time: rclpy.time.Time) -> None:
        """Update the timestamp of the last received visualizer heartbeat.

        Args:
            time: The ROS clock time of receipt.
        """
        with self._lock:
            self.last_heartbeat_time = time

    def is_visualizer_alive(self, current_time: rclpy.time.Time, timeout_sec: float) -> bool:
        """Check if the external visualizer is still active.

        Args:
            current_time: The current ROS clock time.
            timeout_sec: The threshold in seconds for being considered stale.

        Returns:
            bool: True if visualizer heartbeat was recently received, False otherwise.
        """
        with self._lock:
            if self.last_heartbeat_time is None:
                return False
            elapsed = (current_time - self.last_heartbeat_time).nanoseconds / 1e9
            return elapsed < timeout_sec

    def get_core_sensors(self) -> tuple[
        Odometry | None, LaserScan | None, rclpy.time.Time | None, GpsCoord | None
    ]:
        """Returns a snapshot of the core navigation data safely.

        Returns:
            tuple: (latest_odom, latest_scan, scan_time, latest_gps)
        """
        with self._lock:
            return self.odom, self.scan, self.scan_time, self.gps
