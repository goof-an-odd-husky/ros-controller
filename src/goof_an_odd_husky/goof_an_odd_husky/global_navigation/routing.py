from math import atan2, cos, radians, sin, sqrt

import networkx as nx
import osmnx as ox

_SYNTHETIC_ORIGIN_ID = -1
_SYNTHETIC_DESTINATION_ID = -2


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))


def _haversine_heuristic(G: nx.MultiDiGraph):
    def heuristic(u, v):
        return haversine_distance(
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
    nearest_node = ox.nearest_nodes(G, lon, lat)
    dist = haversine_distance(
        lat,
        lon,
        G.nodes[nearest_node]["y"],
        G.nodes[nearest_node]["x"],
    )
    G.add_node(synthetic_node_id, y=lat, x=lon)
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
