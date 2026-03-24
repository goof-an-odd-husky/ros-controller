from goof_an_odd_husky_common.obstacles import (
    Obstacle,
    CircleObstacle,
    LineObstacle,
)
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from numpy.typing import NDArray
from enum import Enum
from typing import Callable


class PathRenderMode(Enum):
    STRAIGHT = "straight"
    ARC = "arc"
    BOTH = "both"


_CIRCLE_THETA = np.linspace(0, 2 * np.pi, 48)
_CIRCLE_COS = np.cos(_CIRCLE_THETA)
_CIRCLE_SIN = np.sin(_CIRCLE_THETA)


def _build_obstacle_polylines(
    obstacles: list[Obstacle],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if not obstacles:
        return np.array([]), np.array([])

    xs, ys = [], []
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            xs.extend((obs.x + obs.radius * _CIRCLE_COS).tolist())
            xs.append(np.nan)
            ys.extend((obs.y + obs.radius * _CIRCLE_SIN).tolist())
            ys.append(np.nan)
        elif isinstance(obs, LineObstacle):
            xs.extend([obs.x1, obs.x2, np.nan])
            ys.extend([obs.y1, obs.y2, np.nan])

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _build_arrow_segments(
    poses: NDArray[np.floating], arrow_length: float = 0.3
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    n = len(poses)
    nan_col = np.full(n, np.nan)
    tip_x = poses[:, 0] + arrow_length * np.cos(poses[:, 2])
    tip_y = poses[:, 1] + arrow_length * np.sin(poses[:, 2])
    xs = np.column_stack([poses[:, 0], tip_x, nan_col]).ravel()
    ys = np.column_stack([poses[:, 1], tip_y, nan_col]).ravel()
    return xs, ys


def _bezier_arc(
    x1: float,
    y1: float,
    theta1: float,
    x2: float,
    y2: float,
    theta2: float,
    num_points: int,
) -> tuple[list[float], list[float]]:
    d = np.hypot(x2 - x1, y2 - y1) / 3.0
    c1x, c1y = x1 + d * np.cos(theta1), y1 + d * np.sin(theta1)
    c2x, c2y = x2 - d * np.cos(theta2), y2 - d * np.sin(theta2)

    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    B = np.hstack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3])
    cpx = np.array([x1, c1x, c2x, x2])
    cpy = np.array([y1, c1y, c2y, y2])
    return (B @ cpx).tolist(), (B @ cpy).tolist()


def _compute_arc_path(
    poses: NDArray[np.floating], points_per_arc: int = 20
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if len(poses) < 2:
        return poses[:, 0], poses[:, 1]
    all_x, all_y = [], []
    for i in range(len(poses) - 1):
        ax, ay = _bezier_arc(*poses[i, :3], *poses[i + 1, :3], points_per_arc)
        all_x.extend(ax)
        all_y.extend(ay)
    return np.array(all_x), np.array(all_y)


def _get_vehicle_poly(
    x: float, y: float, theta: float, scale: float = 0.5
) -> tuple[list[float], list[float]]:
    """Calculates the vertices for an arrow/kite shape to represent the vehicle."""
    pts = np.array(
        [
            [scale, 0.0],
            [-scale / 2, scale / 2],
            [-scale / 4, 0.0],
            [-scale / 2, -scale / 2],
            [scale, 0.0],
        ]
    )
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t_pts = pts @ R.T + np.array([x, y])
    return t_pts[:, 0].tolist(), t_pts[:, 1].tolist()


class TrajectoryVisualizer:
    _is_open: bool
    interactive_obstacles: bool
    obstacles: list[Obstacle]
    obstacle_radius: float
    start_pos: list[float] | None
    goal_pos: list[float] | None
    use_global: bool
    use_gps: bool
    path_render_mode: PathRenderMode
    _app: QtWidgets.QApplication
    _win: QtWidgets.QMainWindow
    _plot: pg.PlotWidget
    _traj_item: pg.PlotDataItem
    _traj_straight_item: pg.PlotDataItem | None
    _start_item: pg.PlotDataItem
    _goal_item: pg.ScatterPlotItem
    _obstacle_item: pg.PlotDataItem
    _arrows_item: pg.PlotDataItem
    _current_obstacles: list[Obstacle]
    on_goal_set: Callable[[float, float], None] | None
    on_cancel: Callable[[], None] | None
    _coord_input: QtWidgets.QLineEdit

    def __init__(
        self,
        x_lim: tuple[float, float] = (0, 10),
        y_lim: tuple[float, float] = (0, 10),
        title: str = "TEB Optimization",
        path_render_mode: PathRenderMode | str = PathRenderMode.STRAIGHT,
        interactive_obstacles: bool = True,
        use_global: bool = False,
        use_gps: bool = True,
        on_goal_set: Callable[[float, float], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._is_open = True
        self.interactive_obstacles = interactive_obstacles
        self.obstacles = []
        self._current_obstacles = []
        self.obstacle_radius = 0.5
        self.start_pos = None
        self.goal_pos = None
        self.use_global = use_global
        self.use_gps = use_gps
        self.on_goal_set = on_goal_set
        self.on_cancel = on_cancel

        self._robot_pose = (0.0, 0.0, 0.0)
        self._raw_trajectory = None
        self._raw_obstacles = None
        self._raw_start_goal = None

        if isinstance(path_render_mode, str):
            path_render_mode = PathRenderMode(path_render_mode)
        self.path_render_mode = path_render_mode

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle(title)
        self._win.resize(900, 700)

        central_widget = QtWidgets.QWidget()
        self._win.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        input_panel = QtWidgets.QWidget()
        input_layout = QtWidgets.QHBoxLayout(input_panel)

        label_text = "Goal (lat, lon):" if self.use_gps else "Goal (x, y):"
        placeholder_text = (
            "e.g., 37.7749, -122.4194" if self.use_gps else "e.g., 5.0, 0.0"
        )

        coord_label = QtWidgets.QLabel(label_text)
        self._coord_input = QtWidgets.QLineEdit()
        self._coord_input.setPlaceholderText(placeholder_text)
        self._coord_input.setMinimumWidth(200)
        self._coord_input.returnPressed.connect(self._on_set_goal_clicked)

        set_goal_btn = QtWidgets.QPushButton("Set Goal")
        set_goal_btn.clicked.connect(self._on_set_goal_clicked)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel_clicked)

        input_layout.addWidget(coord_label)
        input_layout.addWidget(self._coord_input)
        input_layout.addWidget(set_goal_btn)
        input_layout.addStretch()
        input_layout.addWidget(cancel_btn)

        layout.addWidget(input_panel)

        self._plot = pg.PlotWidget()
        layout.addWidget(self._plot)

        self._plot.setXRange(*x_lim, padding=0)
        self._plot.setYRange(*y_lim, padding=0)
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        blue_pen = pg.mkPen("b", width=1.5)
        blue_dashed = pg.mkPen("b", width=1, style=QtCore.Qt.PenStyle.DashLine)

        if path_render_mode == PathRenderMode.STRAIGHT:
            self._traj_item = self._plot.plot(
                [],
                [],
                pen=blue_pen,
                symbol="o",
                symbolSize=4,
                symbolBrush="b",
                symbolPen=None,
            )
        else:
            self._traj_item = self._plot.plot([], [], pen=blue_pen)

        self._traj_straight_item = None
        if path_render_mode == PathRenderMode.BOTH:
            self._traj_straight_item = self._plot.plot(
                [],
                [],
                pen=blue_dashed,
                symbol="o",
                symbolSize=4,
                symbolBrush="b",
                symbolPen=None,
            )

        self._start_item = self._plot.plot([], [], pen=pg.mkPen("r", width=3))
        self._start_item.setZValue(5)

        self._goal_item = pg.ScatterPlotItem(
            symbol="star", size=18, brush=pg.mkBrush("r"), pen=pg.mkPen(None), zValue=5
        )
        self._plot.addItem(self._goal_item)

        self._obstacle_item = self._plot.plot(
            [], [], pen=pg.mkPen("k", width=1.5), fillLevel=None
        )
        self._arrows_item = self._plot.plot(
            [], [], pen=pg.mkPen(color=(0, 0, 200, 140), width=1)
        )

        if interactive_obstacles:
            self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        self._win.keyPressEvent = self._on_key_pressed
        self._win.closeEvent = lambda e: (setattr(self, "_is_open", False), e.accept())
        self._win.show()
        self._app.processEvents()

    def _on_set_goal_clicked(self) -> None:
        text = self._coord_input.text().strip()
        if not text:
            return

        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            if self.use_gps:
                print(
                    "Invalid format. Use: latitude, longitude (e.g., 37.7749, -122.4194)"
                )
            else:
                print("Invalid format. Use: x, y (e.g., 5.0, 0.0)")
            return

        try:
            val1 = float(parts[0])
            val2 = float(parts[1])
            if self.on_goal_set:
                self.on_goal_set(val1, val2)
        except ValueError:
            print("Invalid coordinates. Use numeric values.")

    def _on_cancel_clicked(self) -> None:
        if self.on_cancel:
            self.on_cancel()

    def _on_key_pressed(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_G:
            self.use_global = not self.use_global
            print(f"Visualization mode: {'Global' if self.use_global else 'Robot'}")
            self._process_and_update()

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        if self.use_global:
            return x, y
        return -y, x

    def _from_canvas(self, cx: float, cy: float) -> tuple[float, float]:
        if self.use_global:
            return cx, cy
        return cy, -cx

    @property
    def is_open(self) -> bool:
        return self._is_open

    def update_world_state(
        self,
        robot_pose: tuple[float, float, float] | list[float],
        trajectory: NDArray[np.floating] | None,
        obstacles: list[Obstacle] | None,
        start_goal: tuple[list[float], list[float]] | None,
    ) -> None:
        """Stores the raw local state and triggers a full visual update."""
        self._robot_pose = robot_pose
        self._raw_trajectory = trajectory
        self._raw_obstacles = obstacles
        self._raw_start_goal = start_goal

        self._process_and_update()

    def _process_and_update(self) -> None:
        """Applies coordinate transformations based on the current view mode."""
        rx, ry, rtheta = self._robot_pose
        c, s = np.cos(rtheta), np.sin(rtheta)

        if self._raw_trajectory is not None and len(self._raw_trajectory) > 0:
            if self.use_global:
                gtraj = self._raw_trajectory.copy()
                gtraj[:, 0] = (
                    rx + self._raw_trajectory[:, 0] * c - self._raw_trajectory[:, 1] * s
                )
                gtraj[:, 1] = (
                    ry + self._raw_trajectory[:, 0] * s + self._raw_trajectory[:, 1] * c
                )
                gtraj[:, 2] = self._raw_trajectory[:, 2] + rtheta
                self.update_trajectory(gtraj)
            else:
                self.update_trajectory(self._raw_trajectory)
        else:
            self.update_trajectory(None)

        if self._raw_obstacles is not None:
            if self.use_global:
                graphical_obs = []
                for o in self._raw_obstacles:
                    if isinstance(o, CircleObstacle):
                        graphical_obs.append(
                            CircleObstacle(
                                rx + o.x * c - o.y * s, ry + o.x * s + o.y * c, o.radius
                            )
                        )
                    elif isinstance(o, LineObstacle):
                        graphical_obs.append(
                            LineObstacle(
                                rx + o.x1 * c - o.y1 * s,
                                ry + o.x1 * s + o.y1 * c,
                                rx + o.x2 * c - o.y2 * s,
                                ry + o.x2 * s + o.y2 * c,
                            )
                        )
                self.set_obstacles(graphical_obs)
            else:
                self.set_obstacles(self._raw_obstacles)

        if self._raw_start_goal is not None:
            st, gl = self._raw_start_goal
            if self.use_global:
                gsx, gsy = rx + st[0] * c - st[1] * s, ry + st[0] * s + st[1] * c
                if len(gl) >= 3:
                    ggx, ggy = rx + gl[0] * c - gl[1] * s, ry + gl[0] * s + gl[1] * c
                    self.set_start_goal(
                        [gsx, gsy, st[2] + rtheta], [ggx, ggy, gl[2] + rtheta]
                    )
                else:
                    self.set_start_goal([gsx, gsy, st[2] + rtheta], [])
            else:
                self.set_start_goal(st, gl)

    def set_start_goal(
        self,
        start: NDArray[np.floating] | list[float],
        goal: NDArray[np.floating] | list[float],
    ) -> None:
        sx, sy = self._to_canvas(start[0], start[1])

        if len(start) >= 3:
            canvas_angle_rad = start[2]
            if not self.use_global:
                canvas_angle_rad += np.pi / 2

            vx, vy = _get_vehicle_poly(sx, sy, canvas_angle_rad)
            self._start_item.setData(vx, vy)
        else:
            self._start_item.setData([sx], [sy])

        if len(goal) >= 2:
            gx, gy = self._to_canvas(goal[0], goal[1])
            self._goal_item.setData([gx], [gy])
        else:
            self._goal_item.setData([], [])

    def set_obstacles(self, obstacles: list[Obstacle]) -> None:
        if not self._is_open:
            return
        self._current_obstacles = obstacles
        self._redraw_obstacles()

    def update_trajectory(self, poses: NDArray[np.floating] | None) -> None:
        if poses is None or len(poses) == 0:
            self._traj_item.setData([], [])
            if self._traj_straight_item is not None:
                self._traj_straight_item.setData([], [])
            self._arrows_item.setData([], [])
            return

        transformed = poses.copy()
        if not self.use_global:
            transformed[:, 0], transformed[:, 1] = -poses[:, 1], poses[:, 0]
            transformed[:, 2] = poses[:, 2] + np.pi / 2

        if self.path_render_mode == PathRenderMode.STRAIGHT:
            self._traj_item.setData(transformed[:, 0], transformed[:, 1])
        elif self.path_render_mode == PathRenderMode.ARC:
            ax, ay = _compute_arc_path(transformed)
            self._traj_item.setData(ax, ay)
        else:
            if self._traj_straight_item is not None:
                self._traj_straight_item.setData(transformed[:, 0], transformed[:, 1])
            ax, ay = _compute_arc_path(transformed)
            self._traj_item.setData(ax, ay)

        arrow_xs, arrow_ys = _build_arrow_segments(transformed)
        self._arrows_item.setData(arrow_xs, arrow_ys)

    def get_obstacles(self) -> list[Obstacle]:
        return list(self.obstacles)

    def draw(self, pause_time: float = 0.01) -> None:
        if not self._is_open:
            return
        self._app.processEvents()

    def _redraw_obstacles(self) -> None:
        obs = getattr(self, "_current_obstacles", None)
        if not obs:
            self._obstacle_item.setData([], [])
            return

        canvas_obs = []
        for o in obs:
            if isinstance(o, CircleObstacle):
                cx, cy = self._to_canvas(o.x, o.y)
                canvas_obs.append(CircleObstacle(cx, cy, o.radius))
            elif isinstance(o, LineObstacle):
                cx1, cy1 = self._to_canvas(o.x1, o.y1)
                cx2, cy2 = self._to_canvas(o.x2, o.y2)
                canvas_obs.append(LineObstacle(cx1, cy1, cx2, cy2))

        xs, ys = _build_obstacle_polylines(canvas_obs)
        self._obstacle_item.setData(xs, ys)

    def _on_mouse_clicked(self, event) -> None:
        if not self.interactive_obstacles:
            return

        vb = self._plot.plotItem.vb
        mouse_point = vb.mapSceneToView(event.scenePos())
        x, y = mouse_point.x(), mouse_point.y()

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            robot_x, robot_y = self._from_canvas(x, y)
            self.obstacles.append(
                CircleObstacle(robot_x, robot_y, self.obstacle_radius)
            )
            self._redraw_obstacles()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self._remove_nearest_obstacle(x, y)

    def _remove_nearest_obstacle(self, x: float, y: float) -> None:
        if not self.obstacles:
            return

        robot_x, robot_y = self._from_canvas(x, y)
        dists = []

        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                dists.append(np.hypot(obs.x - robot_x, obs.y - robot_y))
            elif isinstance(obs, LineObstacle):
                px = obs.x2 - obs.x1
                py = obs.y2 - obs.y1
                norm = px * px + py * py
                if norm == 0:
                    dx = obs.x1 - robot_x
                    dy = obs.y1 - robot_y
                else:
                    u = ((robot_x - obs.x1) * px + (robot_y - obs.y1) * py) / float(
                        norm
                    )
                    u = max(0.0, min(1.0, u))
                    dx = obs.x1 + u * px - robot_x
                    dy = obs.y1 + u * py - robot_y
                dists.append(np.hypot(dx, dy))

        if dists:
            idx = int(np.argmin(dists))
            if dists[idx] < 1.0:
                self.obstacles.pop(idx)
                self._redraw_obstacles()
