from goof_an_odd_husky.config import MAX_PATH_EDGE
from goof_an_odd_husky.helpers import coords_distance

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

_SYNTHETIC_ORIGIN_ID = -1
_SYNTHETIC_DESTINATION_ID = -2


def edge_coords(
    G: nx.MultiDiGraph, u: int, v: int, data: dict
) -> list[tuple[float, float]]:
    if "geometry" in data:
        return [(lat, lon) for lon, lat in data["geometry"].coords]
    return [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]


def stitch_path_coords(
    G: nx.MultiDiGraph, path: list[int]
) -> list[tuple[float, float]]:
    stitched = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        segment = edge_coords(G, u, v, G.edges[u, v, 0])
        stitched.extend(segment[1:] if stitched else segment)
    return stitched


def slice_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    sliced = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        dist = coords_distance(*u, *v)
        sliced.append(u)
        if dist <= MAX_PATH_EDGE:
            continue
        slices = int(round(dist / MAX_PATH_EDGE))
        for k in range(1, slices):
            t = k / slices
            sliced.append((u[0] + t * (v[0] - u[0]), u[1] + t * (v[1] - u[1])))
    if path:
        sliced.append(path[-1])
    return sliced


def all_edges_coords(G: nx.MultiDiGraph) -> list[list[tuple[float, float]]]:
    return [edge_coords(G, u, v, data) for u, v, data in G.edges(data=True)]


def _haversine_heuristic(G: nx.MultiDiGraph):
    def heuristic(u, v):
        return coords_distance(
            G.nodes[u]["y"],
            G.nodes[u]["x"],
            G.nodes[v]["y"],
            G.nodes[v]["x"],
        )

    return heuristic


def _astar(G: nx.MultiDiGraph, origin: int, destination: int) -> list[int]:
    return nx.astar_path(
        G,
        origin,
        destination,
        heuristic=_haversine_heuristic(G),
        weight="length",
    )


def _attach_coordinate_node(
    G: nx.MultiDiGraph, synthetic_node_id: int, lat: float, lon: float
) -> None:
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
            lat,
            lon,
            G.nodes[nearest_node]["y"],
            G.nodes[nearest_node]["x"],
        )
        G.add_edge(synthetic_node_id, nearest_node, length=dist)
        G.add_edge(nearest_node, synthetic_node_id, length=dist)


def path_between_nodes(
    G: nx.MultiDiGraph,
    origin_node_id: int,
    destination_node_id: int,
) -> list[int]:
    return _astar(G, origin_node_id, destination_node_id)


def path_between_coordinates(
    G: nx.MultiDiGraph,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> tuple[list[int], nx.MultiDiGraph]:
    """Returns the path and an extended copy of G containing the two synthetic
    endpoint nodes. Pass the extended graph as path_graph to build_folium_map."""
    G_extended = G.copy()
    _attach_coordinate_node(G_extended, _SYNTHETIC_ORIGIN_ID, origin_lat, origin_lon)
    _attach_coordinate_node(G_extended, _SYNTHETIC_DESTINATION_ID, dest_lat, dest_lon)
    path = _astar(G_extended, _SYNTHETIC_ORIGIN_ID, _SYNTHETIC_DESTINATION_ID)
    return path, G_extended
