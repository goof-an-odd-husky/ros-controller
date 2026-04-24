import math
import networkx as nx
from goof_an_odd_husky_common.obstacles import Obstacle, CircleObstacle, LineObstacle
from goof_an_odd_husky.local_navigation.safety import is_point_safe, is_segment_safe, is_start_segment_safe

CIRCLE_BYPASS_SAMPLES = 8


def _obstacle_clearance_nodes(
    obstacles: list[Obstacle], clearance: float
) -> list[tuple[float, float]]:
    """Samples waypoint candidates offset from obstacle boundaries by a clearance distance.

    For circle obstacles, samples `CIRCLE_BYPASS_SAMPLES` points evenly around
    the perimeter at radius + clearance. For line obstacles, generates four offset
    points — one on each side of both endpoints — perpendicular to the segment.

    Args:
        obstacles: List of obstacles to generate bypass nodes for.
        clearance: Radial offset distance from each obstacle boundary.

    Returns:
        List of (x, y) candidate waypoint positions around obstacle boundaries.
        Points are not guaranteed to be collision-free and must be filtered.
    """
    nodes = []
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            r = obs.radius + clearance
            for k in range(CIRCLE_BYPASS_SAMPLES):
                angle = 2 * math.pi * k / CIRCLE_BYPASS_SAMPLES
                nodes.append((obs.x + r * math.cos(angle), obs.y + r * math.sin(angle)))
        elif isinstance(obs, LineObstacle):
            dx, dy = obs.x2 - obs.x1, obs.y2 - obs.y1
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            nx_off = -dy / length * clearance
            ny_off = dx / length * clearance
            for px, py in [(obs.x1, obs.y1), (obs.x2, obs.y2)]:
                nodes.append((px + nx_off, py + ny_off))
                nodes.append((px - nx_off, py - ny_off))
    return nodes


def astar_visibility_path(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    obstacles: list[Obstacle],
    safety_radius: float,
) -> list[tuple[float, float]] | None:
    """Finds a collision-free path using A* search on an obstacle visibility graph.

    Constructs a visibility graph where nodes are the start/goal positions plus
    clearance-offset samples around each obstacle boundary. Two nodes are connected
    by an edge if and only if the straight segment between them is collision-free
    per `is_segment_safe`. A* with Euclidean heuristic then finds the shortest
    such path.

    Args:
        start_xy: (x, y) position of the path start.
        goal_xy: (x, y) position of the path goal.
        obstacles: List of obstacles the path must avoid.
        safety_radius: Minimum clearance from obstacle boundaries for both node
            placement and edge collision checks.

    Returns:
        Ordered list of (x, y) waypoints from start to goal inclusive, or None
        if no collision-free path exists within the visibility graph.
    """
    if not is_point_safe(goal_xy[0], goal_xy[1], obstacles, safety_radius):
        return None

    if not is_point_safe(start_xy[0], start_xy[1], obstacles, margin=0.01):
        return None

    clearance = safety_radius + 0.05

    safe_nodes = [
        v
        for v in _obstacle_clearance_nodes(obstacles, clearance)
        if is_point_safe(v[0], v[1], obstacles, safety_radius)
    ]
    all_nodes = [start_xy] + safe_nodes + [goal_xy]

    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    for i, (x1, y1) in enumerate(all_nodes):
        for j, (x2, y2) in enumerate(all_nodes[i + 1 :], start=i + 1):
            node1 = all_nodes[i]
            node2 = all_nodes[j]
            
            if node1 == start_xy:
                is_safe = is_start_segment_safe(x1, y1, x2, y2, obstacles, safety_radius)
            elif node2 == start_xy:
                is_safe = is_start_segment_safe(x2, y2, x1, y1, obstacles, safety_radius)
            else:
                is_safe = is_segment_safe(x1, y1, x2, y2, obstacles, safety_radius)

            if is_safe:
                G.add_edge(node1, node2, weight=math.hypot(x2 - x1, y2 - y1))

    if G.degree(start_xy) == 0:
        return None

    try:
        return nx.astar_path(
            G,
            start_xy,
            goal_xy,
            heuristic=lambda u, v: math.hypot(v[0] - u[0], v[1] - u[1]),
            weight="weight",
        )
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None
