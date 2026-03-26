import folium
import networkx as nx
import osmnx as ox

from goof_an_odd_husky.global_navigation.graph import coerce_str
from goof_an_odd_husky.global_navigation.routing import (
    all_edges_coords,
    stitch_path_coords,
    slice_path,
)

_COPY_TOAST_JS = """
<style>
  #copy-toast {
    display: none;
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: #fff;
    padding: 8px 18px;
    border-radius: 6px;
    font-size: 14px;
    z-index: 9999;
    pointer-events: none;
  }
</style>
<div id="copy-toast">Copied!</div>
<script>
function copyNodeId(nodeId) {
    navigator.clipboard.writeText(String(nodeId)).then(function() {
        var toast = document.getElementById('copy-toast');
        toast.style.display = 'block';
        setTimeout(function() { toast.style.display = 'none'; }, 1500);
    });
}
</script>
"""


def _compute_center(nodes_with_data) -> tuple[float, float]:
    nodes_with_data = list(nodes_with_data)
    center_lat = sum(d["y"] for _, d in nodes_with_data) / len(nodes_with_data)
    center_lon = sum(d["x"] for _, d in nodes_with_data) / len(nodes_with_data)
    return center_lat, center_lon


def _draw_graph_edges(folium_map: folium.Map, G: nx.MultiDiGraph) -> None:
    for (u, v, data), coords in zip(G.edges(data=True), all_edges_coords(G)):
        highway = coerce_str(data.get("highway", "unknown"))
        surface = coerce_str(data.get("surface", "unknown"))
        length = data.get("length", 0)

        folium.PolyLine(
            locations=coords,
            weight=3,
            color="#3388ff",
            opacity=0.8,
            tooltip=f"{highway} | surface: {surface} | {length:.1f}m",
        ).add_to(folium_map)


def _draw_path(
    folium_map: folium.Map, path: list[int], path_graph: nx.MultiDiGraph
) -> None:
    total_length = sum(
        path_graph.edges[path[i], path[i + 1], 0].get("length", 0)
        for i in range(len(path) - 1)
    )

    coords = slice_path(stitch_path_coords(path_graph, path))

    folium.PolyLine(
        locations=coords,
        weight=6,
        color="#ff6600",
        opacity=0.95,
        tooltip=f"A* path | {len(path)} nodes | {total_length:.1f}m total",
    ).add_to(folium_map)

    for coord in coords:
        folium.CircleMarker(
            location=coord,
            radius=3,
            color="#ff6600",
            fill=True,
            fill_color="#ffeedd",
            fill_opacity=1.0,
            weight=2,
        ).add_to(folium_map)


def _draw_nodes(
    folium_map: folium.Map,
    nodes_with_data,
    path_node_set: set[int],
    path_endpoints: tuple[int, int] | None,
) -> None:
    endpoint_set = set(path_endpoints) if path_endpoints else set()

    for node_id, data in nodes_with_data:
        if node_id in endpoint_set:
            color, radius = "#00cc44", 8
        elif node_id in path_node_set:
            color, radius = "#ff4444", 5
        else:
            color, radius = "#1155bb", 3

        folium.CircleMarker(
            location=(data["y"], data["x"]),
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Node {node_id} (click to copy)",
            popup=folium.Popup(
                f"<b>Node ID:</b> {node_id}<br>"
                f'<button onclick="copyNodeId({node_id})" '
                f'style="margin-top:6px;padding:4px 10px;cursor:pointer;">Copy ID</button>',
                max_width=200,
            ),
        ).add_to(folium_map)


def build_folium_map(
    graph: nx.MultiDiGraph | None = None,
    path: list[int] | None = None,
    path_graph: nx.MultiDiGraph | None = None,
) -> folium.Map:
    """Build a folium map for any combination of graph and/or path.

    graph only          → draws all edges and nodes
    graph + path        → draws all edges with the path highlighted
    path + path_graph   → draws only the path (no background edges)

    path_graph must be supplied when path nodes are not all present in graph
    (e.g. synthetic GPS-coordinate nodes from path_between_coordinates).
    When omitted it falls back to graph.
    """
    if graph is None and path is None:
        raise ValueError("At least one of graph or path must be provided")
    if path is not None and graph is None and path_graph is None:
        raise ValueError("path_graph (or graph) is required when rendering a path")

    g = ox.project_graph(graph, to_crs="EPSG:4326") if graph is not None else None
    pg = (
        ox.project_graph(path_graph, to_crs="EPSG:4326")
        if path_graph is not None
        else g
    )

    path_node_set = set(path) if path else set()
    path_endpoints = (path[0], path[-1]) if path else None

    center_nodes = (
        g.nodes(data=True) if g is not None else ((n, pg.nodes[n]) for n in path)
    )
    center_lat, center_lon = _compute_center(center_nodes)

    folium_map = folium.Map(
        location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB Positron"
    )
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OSM",
        max_native_zoom=19,
        max_zoom=21,
    ).add_to(folium_map)
    folium.Element(_COPY_TOAST_JS).add_to(folium_map.get_root().html)

    if g is not None:
        _draw_graph_edges(folium_map, g)

    if path is not None:
        _draw_path(folium_map, path, pg)

    nodes_to_render = (
        g.nodes(data=True)
        if g is not None
        else ((n, pg.nodes[n]) for n in path_node_set)
    )
    _draw_nodes(folium_map, nodes_to_render, path_node_set, path_endpoints)

    return folium_map
