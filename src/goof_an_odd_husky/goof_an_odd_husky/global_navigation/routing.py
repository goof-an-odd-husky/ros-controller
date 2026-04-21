from typing import Any, Callable

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from goof_an_odd_husky_common.helpers import coords_distance
from goof_an_odd_husky_common.types import GpsCoord


_SYNTHETIC_ORIGIN_ID: int = -1
_SYNTHETIC_DESTINATION_ID: int = -2


def edge_coords(
    G: nx.MultiDiGraph, u: int, v: int, data: dict[str, Any]
) -> list[GpsCoord]:
    """Extract geographical coordinates for a single edge.

    Args:
        G: The map graph.
        u: The starting node ID.
        v: The ending node ID.
        data: Edge data dictionary, which might contain a shapely LineString 'geometry'.

    Returns:
        list[GpsCoord]: A list of GpsCoord forming the edge.
    """
    if "geometry" in data:
        return [GpsCoord(lat, lon) for lon, lat in data["geometry"].coords]
    return [
        GpsCoord(G.nodes[u]["y"], G.nodes[u]["x"]),
        GpsCoord(G.nodes[v]["y"], G.nodes[v]["x"]),
    ]


def stitch_path_coords(G: nx.MultiDiGraph, path: list[int]) -> list[GpsCoord]:
    """Connect edge geometries sequentially across a path of nodes.

    Args:
        G: The graph map.
        path: Ordered list of node IDs.

    Returns:
        list[GpsCoord]: Ordered list of map coordinates.
    """
    stitched: list[GpsCoord] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        segment = edge_coords(G, u, v, G.edges[u, v, 0])
        stitched.extend(segment[1:] if stitched else segment)
    return stitched


def slice_path(path: list[GpsCoord], max_path_edge: float) -> list[GpsCoord]:
    """Interpolate additional points on long straight path segments.

    Ensures no two sequential points are farther apart than max_path_edge meters.

    Args:
        path: A list of GpsCoord points.
        max_path_edge: The max length of each of the edges in the initial sequence.

    Returns:
        list[GpsCoord]: The newly densified coordinate list.
    """
    sliced: list[GpsCoord] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        dist = coords_distance(u, v)
        sliced.append(u)
        if dist <= max_path_edge:
            continue
        slices = int(round(dist / max_path_edge))
        for k in range(1, slices):
            t = k / slices
            sliced.append(
                GpsCoord(
                    u.lat + t * (v.lat - u.lat),
                    u.lon + t * (v.lon - u.lon),
                )
            )
    if path:
        sliced.append(path[-1])
    return sliced


def all_edges_coords(G: nx.MultiDiGraph) -> list[list[GpsCoord]]:
    """Retrieve raw coordinate structures for all edges in the graph.

    Args:
        G: The graph.

    Returns:
        list[list[GpsCoord]]: A list of paths for each edge.
    """
    return [edge_coords(G, u, v, data) for u, v, data in G.edges(data=True)]


def _haversine_heuristic(G: nx.MultiDiGraph) -> Callable[[int, int], float]:
    """Creates a haversine distance heuristic function for A* Search.

    Args:
        G: The map graph.

    Returns:
        Callable: A heuristic function taking nodes (u, v).
    """

    def heuristic(u: int, v: int) -> float:
        return coords_distance(
            GpsCoord(G.nodes[u]["y"], G.nodes[u]["x"]),
            GpsCoord(G.nodes[v]["y"], G.nodes[v]["x"]),
        )

    return heuristic


def _astar(G: nx.MultiDiGraph, origin: int, destination: int) -> list[int]:
    """Executes the A* pathfinding algorithm on the graph.

    Args:
        G: The graph.
        origin: Start node ID.
        destination: End node ID.

    Returns:
        list[int]: The list of node IDs comprising the shortest path.
    """
    return nx.astar_path(
        G,
        origin,
        destination,
        heuristic=_haversine_heuristic(G),
        weight="length",
    )


def _attach_coordinate_node(
    G: nx.MultiDiGraph, synthetic_node_id: int, coord: GpsCoord
) -> None:
    """Attaches an arbitrary GPS point to the graph as a node linked to its nearest neighbors.

    Args:
        G: The mutable graph.
        synthetic_node_id: Unique identifier for the new node.
        coord: The coordinate of the point to attach.
    """
    lat, lon = coord
    nodes = list(G.nodes())
    coords = np.array([(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes])

    tree = cKDTree(coords)

    min_distance, _ = tree.query([lat, lon], k=1)

    radius = 1.5 * min_distance
    indices = tree.query_ball_point([lat, lon], r=radius)

    G.add_node(synthetic_node_id, y=lat, x=lon)

    for idx in indices:
        nearest_node = nodes[idx]
        dist = coords_distance(
            coord,
            GpsCoord(G.nodes[nearest_node]["y"], G.nodes[nearest_node]["x"]),
        )
        G.add_edge(synthetic_node_id, nearest_node, length=dist)
        G.add_edge(nearest_node, synthetic_node_id, length=dist)


def path_between_nodes(
    G: nx.MultiDiGraph,
    origin_node_id: int,
    destination_node_id: int,
) -> list[int]:
    """Generates an A* path strictly between established graph node IDs.

    Args:
        G: The map graph.
        origin_node_id: Start node ID.
        destination_node_id: End node ID.

    Returns:
        list[int]: The computed path of node IDs.
    """
    return _astar(G, origin_node_id, destination_node_id)


def path_between_coordinates(
    G: nx.MultiDiGraph,
    origin: GpsCoord,
    dest: GpsCoord,
) -> tuple[list[int], nx.MultiDiGraph]:
    """Generates an A* path between two arbitrary GPS coordinates by injecting them into a graph copy.

    Args:
        G: The base map graph.
        origin: Start coordinates.
        dest: End coordinates.

    Returns:
        tuple: (The path of node IDs, The extended copy of G containing the synthetic nodes).
    """
    G_extended = G.copy()
    _attach_coordinate_node(G_extended, _SYNTHETIC_ORIGIN_ID, origin)
    _attach_coordinate_node(G_extended, _SYNTHETIC_DESTINATION_ID, dest)
    path = _astar(G_extended, _SYNTHETIC_ORIGIN_ID, _SYNTHETIC_DESTINATION_ID)
    return path, G_extended
