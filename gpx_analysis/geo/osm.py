import geopandas as gpd
import networkx as nx
import pandas as pd

from .constants import (
    LOCAL_OSM_CRS,
    LOCAL_OSM_EDGES_PATH,
    LOCAL_OSM_EDGES_TILE_DIR,
    LOCAL_OSM_NETWORK_TYPE,
    LOCAL_OSM_NODES_PATH,
    LOCAL_OSM_NODES_TILE_DIR,
    OSM_HIGHWAY_PRIORITY,
    PROJECTED_CRS,
)
from .matching import _select_best_osm_match_per_segment
from .spatial import (
    _build_match_windows,
    _filter_edges_to_segment_corridor,
    _join_candidates_within_distance,
    _route_bbox_polygon,
    _read_tiled_geo_parquet,
    _tile_ids_for_bounds,
)

def _normalize_osm_tag(value: object) -> object:
    """Normalize OSM tag values that can be scalar or list-like."""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return value

def _highway_tags(value: object) -> list[str]:
    """Return normalized OSM highway tags as a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    return [text] if text else []

def _highway_priority(value: object) -> int:
    """Return a stable preference rank for OSM highway values."""
    tags = _highway_tags(value)
    if not tags:
        return 999
    return min(OSM_HIGHWAY_PRIORITY.get(tag, 100) for tag in tags)

def _is_excluded_match_highway(value: object) -> bool:
    """Return True when an edge should be excluded from bike/road matching."""
    return "steps" in _highway_tags(value)

def _road_type_from_osm_highway(value: object) -> str:
    """Collapse detailed OSM highway tags into a simple road/gravel label."""
    tags = set(_highway_tags(value))
    if {"track", "path"} & tags:
        return "gravel"
    return "road"

def _require_local_osm_cache() -> None:
    """Ensure the local GeoParquet OSM cache exists on disk."""
    if not LOCAL_OSM_NODES_PATH.exists() or not LOCAL_OSM_EDGES_PATH.exists():
        raise FileNotFoundError(
            "Local OSM GeoParquet cache not found. Run download_bay_area_osm.py "
            f"to create {LOCAL_OSM_NODES_PATH} and {LOCAL_OSM_EDGES_PATH}."
        )

def _load_local_osm_edges(route_poly: object) -> gpd.GeoDataFrame:
    """Load only the locally cached OSM edges intersecting the route bbox."""
    _require_local_osm_cache()
    route_bbox = route_poly.bounds

    if LOCAL_OSM_EDGES_TILE_DIR.exists():
        tile_ids = _tile_ids_for_bounds(route_bbox)
        edges = _read_tiled_geo_parquet(LOCAL_OSM_EDGES_TILE_DIR, tile_ids)
        if edges.empty:
            edges = gpd.read_parquet(LOCAL_OSM_EDGES_PATH, bbox=route_bbox)
    else:
        edges = gpd.read_parquet(LOCAL_OSM_EDGES_PATH, bbox=route_bbox)

    if edges.empty:
        return gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)

    if {"u", "v", "key"}.issubset(edges.columns):
        edges = edges.drop_duplicates(subset=["u", "v", "key"]).copy()
    return edges

def _load_local_osm_nodes_for_edges(route_poly: object, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load route-scoped nodes referenced by the provided edge subset."""
    if edges.empty:
        return gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)

    route_bbox = route_poly.bounds
    node_ids = pd.Index(edges["u"]).union(pd.Index(edges["v"]))

    if LOCAL_OSM_NODES_TILE_DIR.exists():
        tile_ids = _tile_ids_for_bounds(route_bbox)
        nodes = _read_tiled_geo_parquet(LOCAL_OSM_NODES_TILE_DIR, tile_ids)
        if nodes.empty:
            nodes = gpd.read_parquet(LOCAL_OSM_NODES_PATH, bbox=route_bbox)
    else:
        nodes = gpd.read_parquet(LOCAL_OSM_NODES_PATH, bbox=route_bbox)
    if nodes.empty:
        return gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)

    nodes = nodes[nodes["osmid"].isin(node_ids)].copy()
    return nodes.drop_duplicates(subset=["osmid"]).copy()

def _restore_graph_indexes(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Restore OSMnx-compatible indexes after reading node/edge parquet files."""
    nodes_indexed = nodes.set_index("osmid")
    edges_indexed = edges.set_index(["u", "v", "key"])
    return nodes_indexed, edges_indexed

def _filter_to_largest_component(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Keep only the largest connected component using endpoint ids, not a full OSMnx graph."""
    if nodes.empty or edges.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)
        return empty, empty

    graph = nx.Graph()
    graph.add_edges_from(edges[["u", "v"]].itertuples(index=False, name=None))
    if graph.number_of_nodes() == 0:
        empty = gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)
        return empty, empty

    component_nodes = max(nx.connected_components(graph), key=len)
    component_ids = pd.Index(component_nodes)
    filtered_edges = edges[edges["u"].isin(component_ids) & edges["v"].isin(component_ids)].copy()
    filtered_nodes = nodes[nodes["osmid"].isin(component_ids)].copy()
    return filtered_nodes, filtered_edges

def build_route_graph(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str,
    corridor_m: float,
    retain_all: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Clip the local Bay Area OSM graph to the route corridor and return projected segments, nodes, and edges."""
    if gdf_segments.crs is None:
        raise ValueError("gdf_segments must have a CRS.")

    if gdf_segments.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=gdf_segments.crs)
        return gdf_segments.to_crs(PROJECTED_CRS), empty, empty

    projected_segments = gdf_segments.to_crs(PROJECTED_CRS)
    route_bbox_poly = _route_bbox_polygon(projected_segments, corridor_m)

    # network_type is retained for API compatibility, but all local OSM work
    # uses the prebuilt all_public Bay Area graph.
    _ = network_type

    edges = _load_local_osm_edges(route_bbox_poly)
    edges = _filter_edges_to_segment_corridor(edges, projected_segments, corridor_m)
    if edges.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)
        return projected_segments, empty, empty

    nodes = _load_local_osm_nodes_for_edges(route_bbox_poly, edges)
    if nodes.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)
        return projected_segments, empty, empty

    if not retain_all:
        nodes, edges = _filter_to_largest_component(nodes, edges)

    nodes, edges = _restore_graph_indexes(nodes, edges)

    return projected_segments, nodes, edges

def enrich_segments_with_osm_edges(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str = LOCAL_OSM_NETWORK_TYPE,
    corridor_m: float = 6.0,
    match_max_distance_m: float = 15.0,
    match_preference_tolerance_m: float = 4.0,
    match_window_size: int = 5,
    retain_all: bool = True,
) -> gpd.GeoDataFrame:
    """Return a copy of route segments enriched with nearest OSM edge attributes."""
    edge_attrs = ["highway", "lanes", "maxspeed", "name"]
    result = gdf_segments.copy()

    # Pre-create output columns so the function always returns a predictable schema.
    # Nothing to match if either side has no rows.
    output_cols = [f"osm_{col}" for col in edge_attrs]
    for col in output_cols:
        if col not in result.columns:
            result[col] = pd.NA
    if "road_type" not in result.columns:
        result["road_type"] = "road"
    if result.empty:
        return result

    if gdf_segments.crs is None:
        raise ValueError("gdf_segments must have a CRS.")

    projected_segments = gdf_segments.to_crs(PROJECTED_CRS)
    candidate_corridor_m = max(corridor_m, match_max_distance_m)
    route_bbox_poly = _route_bbox_polygon(projected_segments, candidate_corridor_m)

    # network_type is retained for API compatibility, but all local OSM work
    # uses the prebuilt all_public Bay Area graph.
    _ = (network_type, retain_all)

    edges = _load_local_osm_edges(route_bbox_poly)
    edges = _filter_edges_to_segment_corridor(edges, projected_segments, candidate_corridor_m)
    if edges.empty:
        return result

    # Only keep edge attrs that exist in this graph (OSM coverage varies by area).
    available_edge_attrs = [col for col in edge_attrs if col in edges.columns]
    if not available_edge_attrs:
        return result

    # Keep only required edge columns and normalize list-like OSM tags to simple strings.
    edges_subset = edges[available_edge_attrs + ["geometry"]].copy().reset_index()
    for col in available_edge_attrs:
        edges_subset[col] = edges_subset[col].apply(_normalize_osm_tag)
    if "highway" in edges_subset.columns:
        edges_subset = edges_subset[~edges_subset["highway"].apply(_is_excluded_match_highway)].copy()
    if edges_subset.empty:
        return result
    if "highway" in edges_subset.columns:
        edges_subset["_highway_priority"] = edges_subset["highway"].apply(_highway_priority)
    else:
        edges_subset["_highway_priority"] = 999

    # Prepare both sides in projected CRS so nearest-distance matching is meaningful.
    # _segment_index is a stable key to merge matched attributes back onto original rows.
    left = _build_match_windows(projected_segments, match_window_size)
    right = gpd.GeoDataFrame(edges_subset, geometry="geometry", crs=edges.crs).to_crs(PROJECTED_CRS)

    matched = _join_candidates_within_distance(
        left,
        right,
        max_distance_m=match_max_distance_m,
    )
    if matched.empty:
        return result

    matched = _select_best_osm_match_per_segment(
        matched,
        overlap_buffer_m=corridor_m,
        match_preference_tolerance_m=match_preference_tolerance_m,
    )
    if matched.empty:
        return result

    # Build a compact table of matched attributes to merge back onto segments.
    attrs = pd.DataFrame({"_segment_index": matched["_segment_index"]})
    for col in available_edge_attrs:
        attrs[f"osm_{col}"] = matched[col]

    # Merge by segment key, then fill pre-created columns with matched values when present.
    result = result.reset_index().rename(columns={"index": "_segment_index"}).merge(attrs, on="_segment_index", how="left", suffixes=("", "_matched"))
    for col in available_edge_attrs:
        colname = f"osm_{col}"
        matched_col = f"{colname}_matched"
        result[colname] = result[matched_col].combine_first(result[colname])
        result = result.drop(columns=[matched_col])

    result["road_type"] = result["osm_highway"].apply(_road_type_from_osm_highway)

    # Restore the original index semantics and return as GeoDataFrame.
    result = result.set_index("_segment_index")
    result.index.name = gdf_segments.index.name
    return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf_segments.crs)
