import osmnx as ox
import networkx as nx

ox.settings.useful_tags_way = ox.settings.useful_tags_way + [
    "surface",
    "access",
    "foot",
    "smoothness",
]

PAVED_SURFACES = {
    "paved",
    "asphalt",
    "concrete",
    "paving_stones",
    "sett",
    "cobblestone",
    # "compacted",
    # "fine_gravel",
}

EXCLUDED_HIGHWAY_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "cycleway",
    "busway",
}
EXCLUDED_ACCESS = {"no", "private"}


def coerce_str(value, allowed: set | None = None) -> str:
    if isinstance(value, list):
        for v in value:
            if allowed is None or v in allowed:
                return v
        return ""
    return value if (allowed is None or value in allowed) else ""


def is_walkable_paved(edge_data: dict) -> bool:
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
    park_gdf = ox.geocode_to_gdf(f"R{osm_relation_id}", by_osmid=True)
    park_polygon = park_gdf.geometry.iloc[0]
    return ox.graph_from_polygon(park_polygon, network_type="walk", retain_all=True)


def filter_walkable_paved(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
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
