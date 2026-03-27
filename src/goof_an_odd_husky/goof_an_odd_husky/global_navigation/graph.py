from goof_an_odd_husky_common.config import (
    PAVED_SURFACES,
    EXCLUDED_HIGHWAY_TYPES,
    EXCLUDED_ACCESS,
)
import osmnx as ox
import networkx as nx
from typing import Any

from goof_an_odd_husky_common.helpers import coerce_str


ox.settings.useful_tags_way = ox.settings.useful_tags_way + [
    "surface",
    "access",
    "foot",
    "smoothness",
]


def is_walkable_paved(edge_data: dict[str, Any]) -> bool:
    """Determines if a graph edge represents a walkable, paved path.

    Args:
        edge_data: A dictionary containing OSM edge attributes.

    Returns:
        bool: True if the path is walkable and paved, False otherwise.
    """
    surface = coerce_str(edge_data.get("surface", ""), PAVED_SURFACES)
    highway = coerce_str(edge_data.get("highway", ""), EXCLUDED_HIGHWAY_TYPES)
    foot = coerce_str(edge_data.get("foot", ""), EXCLUDED_ACCESS)
    access = coerce_str(edge_data.get("access", ""), EXCLUDED_ACCESS)

    if foot in EXCLUDED_ACCESS or access in EXCLUDED_ACCESS:
        return False
    if highway:
        return False
    return bool(surface)


def load_graph_for_relation(osm_relation_id: int) -> nx.MultiDiGraph:
    """Loads a walkable OpenStreetMap graph for a given relation ID.

    Args:
        osm_relation_id: The OpenStreetMap relation ID representing the area polygon.

    Returns:
        nx.MultiDiGraph: The loaded walking graph.
    """
    park_gdf = ox.geocode_to_gdf(f"R{osm_relation_id}", by_osmid=True)
    park_polygon = park_gdf.geometry.iloc[0]
    return ox.graph_from_polygon(park_polygon, network_type="walk", retain_all=True)


def filter_walkable_paved(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Filters an OSM graph to retain only strictly paved, walkable edges.

    Args:
        G: The raw NetworkX MultiDiGraph.

    Returns:
        nx.MultiDiGraph: A filtered subgraph containing the largest connected paved component.
    """
    non_walkable_edges = [
        (u, v, k)
        for u, v, k, data in G.edges(keys=True, data=True)
        if not is_walkable_paved(data)
    ]
    G.remove_edges_from(non_walkable_edges)
    G.remove_nodes_from(list(nx.isolates(G)))

    components = list(nx.weakly_connected_components(G))
    if not components:
        raise RuntimeError("No edges remain after filtering")

    return G.subgraph(max(components, key=len)).copy()
