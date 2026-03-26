from goof_an_odd_husky.config import OSM_RELATION_ID
from goof_an_odd_husky.global_navigation.graph import (
    load_graph_for_relation,
    filter_walkable_paved,
)
from goof_an_odd_husky.global_navigation.routing import (
    path_between_nodes,
    path_between_coordinates,
)
from goof_an_odd_husky.visualization.map_visualizer import build_folium_map

ORIGIN_NODE = 1707348491
DESTINATION_NODE = 7600216860

ORIGIN_COORD = (49.81793, 24.02362)
DEST_COORD = (49.82118, 24.02240)


def main():
    print("Fetching and filtering graph...")
    G_raw = load_graph_for_relation(OSM_RELATION_ID)
    G = filter_walkable_paved(G_raw)
    print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    path_nodes = path_between_nodes(G, ORIGIN_NODE, DESTINATION_NODE)
    build_folium_map(graph=G, path=path_nodes).save("map_node_route.html")

    path_coords, G_extended = path_between_coordinates(G, *ORIGIN_COORD, *DEST_COORD)
    build_folium_map(graph=G, path=path_coords, path_graph=G_extended).save(
        "map_coord_route.html"
    )

    build_folium_map(graph=G).save("map_graph_only.html")

    build_folium_map(path=path_nodes, path_graph=G).save("map_path_only.html")

    print("Saved 4 maps.")


if __name__ == "__main__":
    main()
