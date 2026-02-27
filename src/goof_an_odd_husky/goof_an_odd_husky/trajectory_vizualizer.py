import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from numpy.typing import NDArray
from enum import Enum


class PathRenderMode(Enum):
    STRAIGHT = "straight"
    ARC = "arc"
    BOTH = "both"


_CIRCLE_THETA = np.linspace(0, 2 * np.pi, 48)
_CIRCLE_COS = np.cos(_CIRCLE_THETA)
_CIRCLE_SIN = np.sin(_CIRCLE_THETA)


def _build_obstacle_polylines(
    obstacles: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if len(obstacles) == 0:
        return np.array([]), np.array([])
    n = len(obstacles)
    ox = obstacles[:, 0:1]
    oy = obstacles[:, 1:2]
    r = obstacles[:, 2:3]
    circle_x = ox + r * _CIRCLE_COS
    circle_y = oy + r * _CIRCLE_SIN
    nan_col = np.full((n, 1), np.nan)
    xs = np.hstack([circle_x, nan_col]).ravel()
    ys = np.hstack([circle_y, nan_col]).ravel()
    return xs, ys


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


class TrajectoryVisualizer:
    def __init__(
        self,
        x_lim: tuple[float, float] = (0, 10),
        y_lim: tuple[float, float] = (0, 10),
        title: str = "TEB Optimization",
        path_render_mode: PathRenderMode | str = PathRenderMode.STRAIGHT,
        interactive_obstacles: bool = True,
        use_global: bool = False,
    ) -> None:
        self._is_open = True
        self.interactive_obstacles = interactive_obstacles
        self.obstacles: list[list[float]] = []
        self.obstacle_radius = 0.5
        self.start_pos = None
        self.goal_pos = None
        self.use_global = use_global

        if isinstance(path_render_mode, str):
            path_render_mode = PathRenderMode(path_render_mode)
        self.path_render_mode = path_render_mode

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self._win = QtWidgets.QMainWindow()
        self._win.setWindowTitle(title)
        self._win.resize(900, 600)

        self._plot = pg.PlotWidget()
        self._win.setCentralWidget(self._plot)

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

        self._start_item = pg.ScatterPlotItem(
            symbol="s", size=14, brush=pg.mkBrush("g"), pen=pg.mkPen(None), zValue=5
        )
        self._goal_item = pg.ScatterPlotItem(
            symbol="star", size=18, brush=pg.mkBrush("r"), pen=pg.mkPen(None), zValue=5
        )
        self._plot.addItem(self._start_item)
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

    def _on_key_pressed(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_G:
            self.use_global = not self.use_global
            print(f"Visualization mode: {'Global' if self.use_global else 'Robot'}")
            self._redraw_obstacles()

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

    def set_start_goal(
        self,
        start: NDArray[np.floating] | list[float],
        goal: NDArray[np.floating] | list[float],
    ) -> None:
        sx, sy = self._to_canvas(start[0], start[1])
        gx, gy = self._to_canvas(goal[0], goal[1])
        self._start_item.setData([sx], [sy])
        self._goal_item.setData([gx], [gy])

    def set_obstacles(self, obstacles: NDArray[np.floating]) -> None:
        if not self._is_open:
            return
        self._obstacles_array = np.asarray(obstacles, dtype=np.float64)
        self._redraw_obstacles()

    def update_trajectory(self, poses: NDArray[np.floating] | None) -> None:
        if poses is None or len(poses) == 0:
            return

        transformed = poses.copy()
        if self.use_global:
            pass
        else:
            transformed[:, 0], transformed[:, 1] = -poses[:, 1], poses[:, 0]
            transformed[:, 2] = poses[:, 2] + np.pi / 2

        if self.path_render_mode == PathRenderMode.STRAIGHT:
            self._traj_item.setData(transformed[:, 0], transformed[:, 1])
        elif self.path_render_mode == PathRenderMode.ARC:
            ax, ay = _compute_arc_path(transformed)
            self._traj_item.setData(ax, ay)
        else:
            self._traj_straight_item.setData(transformed[:, 0], transformed[:, 1])
            ax, ay = _compute_arc_path(transformed)
            self._traj_item.setData(ax, ay)

        arrow_xs, arrow_ys = _build_arrow_segments(transformed)
        self._arrows_item.setData(arrow_xs, arrow_ys)

    def get_obstacles(self) -> NDArray[np.floating]:
        if not self.obstacles:
            return np.empty((0, 3))
        return np.array(self.obstacles)

    def draw(self, pause_time: float = 0.01) -> None:
        if not self._is_open:
            return
        self._app.processEvents()

    def _redraw_obstacles(self) -> None:
        obs = getattr(self, "_obstacles_array", None)
        if obs is None or len(obs) == 0:
            self._obstacle_item.setData([], [])
            return

        if self.use_global:
            canvas_obs = obs
        else:
            canvas_obs = np.column_stack([-obs[:, 1], obs[:, 0], obs[:, 2]])

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
            self.obstacles.append([robot_x, robot_y, self.obstacle_radius])
            self._redraw_obstacles()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self._remove_nearest_obstacle(x, y)

    def _remove_nearest_obstacle(self, x: float, y: float) -> None:
        if not self.obstacles:
            return
        obs_arr = np.array(self.obstacles)
        dists = np.hypot(obs_arr[:, 0] - x, obs_arr[:, 1] - y)
        idx = int(np.argmin(dists))
        if dists[idx] < 1.0:
            self.obstacles.pop(idx)
            self._redraw_obstacles()
