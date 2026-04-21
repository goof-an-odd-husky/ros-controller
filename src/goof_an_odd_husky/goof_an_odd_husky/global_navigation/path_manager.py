import math
import threading

from rclpy.impl.rcutils_logger import RcutilsLogger
import networkx as nx

from goof_an_odd_husky.global_navigation.graph import (
    load_graph_for_relation,
    filter_walkable_paved,
)
from goof_an_odd_husky.global_navigation.routing import (
    path_between_coordinates,
    stitch_path_coords,
    slice_path,
)
from goof_an_odd_husky_common.helpers import gps_to_vector
from goof_an_odd_husky_common.types import GpsCoord


class GlobalPathManager:
    """Manages the global path, including generation, caching, and GPS-based local frame transformation.

    Attributes:
        use_gps: Boolean dictating if global mapping relies on GPS coordinates.
        logger: A ROS 2 logger instance.
        graph: The loaded, filtered NetworkX MultiDiGraph representing the map.
        lock: A threading lock to protect path state variables.
        needs_global_path: Flag to indicate path must be regenerated.
        global_path_local: Path coordinates converted to the local odometry frame.
        global_path: Path coordinates represented as GpsCoord objects (lat, lon).
        current_path_index: Current progress index along the local path.
        goal: The current global goal coordinate.
        goal_reached: Flag indicating if the vehicle is currently at the goal.
    """

    use_gps: bool
    logger: RcutilsLogger
    graph: nx.MultiDiGraph | None
    lock: threading.Lock

    needs_global_path: bool
    global_path_local: list[tuple[float, float]]
    global_path: list[GpsCoord]
    current_path_index: int
    goal: GpsCoord | None
    goal_reached: bool

    def __init__(self, use_gps: bool, logger: RcutilsLogger, osm_relation_id: int) -> None:
        """Initialize the GlobalPathManager.

        Args:
            use_gps: Whether to load and use OSM graphs for GPS routing.
            logger: A ROS 2 logger instance.
            osm_relation_id: The ID of the OSM relation, in which the navigation happens.
        """
        self.use_gps = use_gps
        self.logger = logger
        self.graph = (
            filter_walkable_paved(load_graph_for_relation(osm_relation_id))
            if use_gps
            else None
        )
        self.lock = threading.Lock()

        self.needs_global_path = False
        self.global_path_local = []
        self.global_path = []
        self.current_path_index = 0
        self.goal = None
        self.goal_reached = False

    def has_goal(self) -> bool:
        """Check if a goal is currently set.

        Returns:
            bool: True if a goal is active.
        """
        with self.lock:
            return self.goal is not None

    def set_goal(self, goal: GpsCoord) -> None:
        """Set a new goal and trigger global path regeneration.

        Args:
            goal: The target GPS coordinate (lat, lon) or map coordinate.
        """
        with self.lock:
            self.goal = goal
            self.goal_reached = False
            self.needs_global_path = True
            self.global_path_local = []
            self.global_path = []
            self.current_path_index = 0

    def cancel_goal(self) -> None:
        """Cancel the current goal and clear the path."""
        with self.lock:
            self.goal = None
            self.goal_reached = False
            self.needs_global_path = False
            self.global_path_local = []
            self.global_path = []
            self.current_path_index = 0

    def update_local_path_from_gps(
        self, gps_anchor: GpsCoord | None, vehicle_x: float, vehicle_y: float
    ) -> None:
        """Update the local coordinates of the global path based on current GPS and Odometry.

        Args:
            gps_anchor: The current GPS position as a GpsCoord, or None.
            vehicle_x: Vehicle's current global X coordinate.
            vehicle_y: Vehicle's current global Y coordinate.
        """
        with self.lock:
            if not self.use_gps or not self.global_path or gps_anchor is None:
                return

            updated_path = []
            for coord in self.global_path:
                dx, dy = gps_to_vector(gps_anchor, coord)
                updated_path.append((vehicle_x + dx, vehicle_y + dy))
            self.global_path_local = updated_path

    def generate_path(
        self, gps_anchor: GpsCoord | None, vehicle_x: float, vehicle_y: float
    ) -> None:
        """Generate a new global path to the goal.

        Args:
            gps_anchor: The current GPS position to anchor path generation, or None.
            vehicle_x: Vehicle's X coordinate (used for non-GPS straight line fallback).
            vehicle_y: Vehicle's Y coordinate (used for non-GPS straight line fallback).
        """
        with self.lock:
            if not self.needs_global_path or self.goal is None:
                return
            goal_coord = self.goal

        try:
            path_coords_local = []
            if self.use_gps and gps_anchor:
                self.logger.info("Generating A* global path from OSM graph...")
                path_nodes, G_ext = path_between_coordinates(
                    self.graph,
                    gps_anchor,
                    goal_coord,
                )
                path_coords = stitch_path_coords(G_ext, path_nodes)
                new_global_path = [coord for coord in slice_path(path_coords)]
            else:
                self.logger.info("Generating straight line global path (No GPS)...")
                new_global_path = []
                dx = goal_coord.lat - vehicle_x
                dy = goal_coord.lon - vehicle_y
                dist = math.hypot(dx, dy)
                slices = max(1, int(dist / 2.0))
                for i in range(slices + 1):
                    t = i / slices
                    path_coords_local.append((vehicle_x + t * dx, vehicle_y + t * dy))

            with self.lock:
                if self.use_gps:
                    self.global_path = new_global_path
                else:
                    self.global_path_local = path_coords_local

                self.current_path_index = 0
                self.needs_global_path = False

        except Exception as e:
            self.logger.error(f"Failed to generate global path: {e}")
            with self.lock:
                self.needs_global_path = True

    def check_goal_reached(
        self, vehicle_x: float, vehicle_y: float, threshold: float = 1.0
    ) -> bool:
        """Check if the vehicle has reached the final point of the global path.

        Args:
            vehicle_x: Vehicle's current global X coordinate.
            vehicle_y: Vehicle's current global Y coordinate.
            threshold: Distance in meters to consider the goal reached.

        Returns:
            bool: True if reached, False otherwise.
        """
        with self.lock:
            if not self.global_path_local:
                return False

            final_x, final_y = self.global_path_local[-1]
            dist_to_final = math.hypot(final_x - vehicle_x, final_y - vehicle_y)

            if dist_to_final < threshold:
                if not self.goal_reached:
                    self.goal_reached = True
                    self.logger.info("Global Goal Reached!")
                    self.goal = None
                    self.needs_global_path = False
                    self.global_path_local = []
                    self.global_path = []
                    self.current_path_index = 0
                return True
            return False

    def get_local_path_snapshot(self) -> list[tuple[float, float]]:
        """Returns a thread-safe copy of the local-frame global path.

        Returns:
            list[tuple[float, float]]: A snapshot of local path coordinates.
        """
        with self.lock:
            return list(self.global_path_local)

    def get_current_index(self) -> int:
        """Returns the current path sequence index safely.

        Returns:
            int: The index.
        """
        with self.lock:
            return self.current_path_index

    def update_current_index(self, index: int) -> None:
        """Updates the current path sequence index safely.

        Args:
            index: The new integer index.
        """
        with self.lock:
            self.current_path_index = index
