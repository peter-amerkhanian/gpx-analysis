import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

PROJECTED_CRS = 3857

def points_to_segments(gdf: gpd.GeoDataFrame, lon: str = "lon", lat: str = "lat", sort_col: str | None = None) -> gpd.GeoDataFrame:
    """Convert ordered point rows into consecutive line segments in lon/lat space."""
    frame = gdf.sort_values(sort_col) if sort_col else gdf.sort_index()
    x0, y0 = frame[lon].to_numpy()[:-1], frame[lat].to_numpy()[:-1]
    x1, y1 = frame[lon].to_numpy()[1:], frame[lat].to_numpy()[1:]
    segment_geometry = [LineString([(a, b), (c, d)]) for a, b, c, d in zip(x0, y0, x1, y1)]
    segments = gpd.GeoDataFrame(
        {"start_i": frame.index[:-1], "end_i": frame.index[1:]},
        geometry=segment_geometry,
        crs=gdf.crs,
    )
    segments = segments.merge(gdf.drop(columns="geometry"), left_on="end_i", right_on="step", how="left")
    return segments


def _normalize_osm_tag(value: object) -> object:
    """Normalize OSM tag values that can be scalar or list-like."""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return value


def _build_route_graph(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str,
    corridor_m: float,
    retain_all: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build a route-corridor OSM graph and return projected segments, nodes, and edges."""
    if gdf_segments.crs is None:
        raise ValueError("gdf_segments must have a CRS.")

    if gdf_segments.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=gdf_segments.crs)
        return gdf_segments.to_crs(PROJECTED_CRS), empty, empty

    projected_segments = gdf_segments.to_crs(PROJECTED_CRS)
    projected_route_poly = projected_segments.geometry.union_all().buffer(corridor_m)
    route_poly = gpd.GeoSeries([projected_route_poly], crs=PROJECTED_CRS).to_crs(gdf_segments.crs).iloc[0]

    graph = ox.graph_from_polygon(route_poly, network_type=network_type, simplify=False, retain_all=retain_all)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    return projected_segments, nodes, edges


def stop_signs_on_segments(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str = "drive",
    corridor_m: float = 6.0,
    segment_buffer_m: float = 8.0,
    retain_all: bool = True
) -> gpd.GeoDataFrame:
    """Find stop controls near route segments from the OpenStreetMap network."""
    projected_segments, nodes, _ = _build_route_graph(gdf_segments, network_type, corridor_m, retain_all)

    if nodes.empty or "highway" not in nodes.columns:
        return nodes.iloc[0:0].to_crs(gdf_segments.crs)

    allowed = {"stop", "traffic_signals"}
    controls = nodes[nodes["highway"].apply(lambda value: value in allowed or (isinstance(value, list) and any(tag in allowed for tag in value)))].copy()

    if controls.empty:
        return controls.to_crs(gdf_segments.crs)

    projected_route_buffer = projected_segments.geometry.union_all().buffer(segment_buffer_m)
    projected_controls = controls.to_crs(PROJECTED_CRS)
    controls_on_route = projected_controls[projected_controls.geometry.intersects(projected_route_buffer)].copy()
    return controls_on_route.to_crs(gdf_segments.crs)


def enrich_segments_with_osm_edges(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str = "drive",
    corridor_m: float = 6.0,
    match_max_distance_m: float = 15.0,
    retain_all: bool = True,
) -> gpd.GeoDataFrame:
    """Return a copy of route segments enriched with nearest OSM edge attributes."""
    edge_attrs = ["highway", "lanes", "maxspeed", "name"]
    # Build the same route-scoped graph used elsewhere so enrichment stays local to the route.
    projected_segments, _, edges = _build_route_graph(gdf_segments, network_type, corridor_m, retain_all)
    result = gdf_segments.copy()

    # Pre-create output columns so the function always returns a predictable schema.
    # Nothing to match if either side has no rows.
    output_cols = [f"osm_{col}" for col in edge_attrs]
    for col in output_cols:
        if col not in result.columns:
            result[col] = pd.NA
    if result.empty or edges.empty:
        return result

    # Only keep edge attrs that exist in this graph (OSM coverage varies by area).
    available_edge_attrs = [col for col in edge_attrs if col in edges.columns]
    if not available_edge_attrs:
        return result

    # Keep only required edge columns and normalize list-like OSM tags to simple strings.
    edges_subset = edges[available_edge_attrs + ["geometry"]].copy().reset_index()
    for col in available_edge_attrs:
        edges_subset[col] = edges_subset[col].apply(_normalize_osm_tag)

    # Prepare both sides in projected CRS so nearest-distance matching is meaningful.
    # _segment_index is a stable key to merge matched attributes back onto original rows.
    left = projected_segments[["geometry"]].copy().reset_index().rename(columns={"index": "_segment_index"})
    right = gpd.GeoDataFrame(edges_subset, geometry="geometry", crs=edges.crs).to_crs(PROJECTED_CRS)

    # For each segment, find the nearest OSM edge within max distance.
    matched = gpd.sjoin_nearest(
        left,
        right,
        how="left",
        max_distance=match_max_distance_m,
        distance_col="_edge_dist_m",
    )

    # If multiple edge matches appear, keep the closest one per segment.
    matched = matched.sort_values("_edge_dist_m").drop_duplicates(subset=["_segment_index"], keep="first")

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

    # Restore the original index semantics and return as GeoDataFrame.
    result = result.set_index("_segment_index")
    result.index.name = gdf_segments.index.name
    return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf_segments.crs)
