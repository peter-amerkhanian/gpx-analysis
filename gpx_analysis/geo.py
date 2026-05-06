import geopandas as gpd
import networkx as nx
import pandas as pd
import math
from pathlib import Path
from functools import lru_cache
from shapely.geometry import box
from shapely.geometry import LineString
from typing import cast
import fiona

PROJECTED_CRS = 3857
BART_KML_PATH = Path(__file__).resolve().parent.parent / "data" / "bart_stations.kml"
BART_STATION_LAYER = "BART Station"
OSM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "osm"
LOCAL_OSM_NODES_PATH = OSM_DATA_DIR / "sf_bay_area_all_public_nodes.parquet"
LOCAL_OSM_EDGES_PATH = OSM_DATA_DIR / "sf_bay_area_all_public_edges.parquet"
LOCAL_OSM_NETWORK_TYPE = "all_public"
LOCAL_OSM_CRS = 4326
LOCAL_OSM_TILE_SIZE_DEG = 0.05
LOCAL_OSM_NODES_TILE_DIR = OSM_DATA_DIR / "sf_bay_area_all_public_nodes_tiles"
LOCAL_OSM_EDGES_TILE_DIR = OSM_DATA_DIR / "sf_bay_area_all_public_edges_tiles"
OSM_HIGHWAY_PRIORITY = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "unclassified": 5,
    "residential": 6,
    "living_street": 7,
    "road": 8,
    "service": 9,
    "track": 10,
    "cycleway": 11,
    "path": 12,
    "footway": 13,
    "pedestrian": 14,
    "steps": 15,
}


@lru_cache(maxsize=1)
def _load_bart_stations() -> gpd.GeoDataFrame:
    """Load BART station points from the repo KML in projected CRS."""
    fiona.drvsupport.supported_drivers.setdefault("KML", "rw")
    stations = gpd.read_file(BART_KML_PATH, driver="KML", layer=BART_STATION_LAYER)
    return stations.to_crs(PROJECTED_CRS)


def add_bart_station(gdf: gpd.GeoDataFrame, step: int = 0) -> str:
    """Return the nearest BART station name to the selected route step."""
    if gdf.crs is None:
        raise ValueError("gdf must have a CRS.")
    if gdf.empty:
        raise ValueError("gdf must not be empty.")
    if not (-len(gdf) <= step < len(gdf)):
        raise IndexError(f"step {step} is out of bounds for gdf with {len(gdf)} rows.")

    route_geometry = gdf.to_crs(PROJECTED_CRS).geometry.iloc[step]
    stations = _load_bart_stations()
    nearest_station = stations.loc[stations.geometry.distance(route_geometry).idxmin(), "Name"]
    return cast(str, nearest_station)

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


def _project_geometry_to_crs(geometry: object, source_crs: object, target_crs: object) -> object:
    """Project a single shapely geometry between CRS definitions."""
    return gpd.GeoSeries([geometry], crs=source_crs).to_crs(target_crs).iloc[0]


def _expanded_route_bbox(projected_segments: gpd.GeoDataFrame, corridor_m: float) -> tuple[float, float, float, float]:
    """Return the route bounds expanded by the requested corridor in projected CRS."""
    minx, miny, maxx, maxy = projected_segments.total_bounds
    return (minx - corridor_m, miny - corridor_m, maxx + corridor_m, maxy + corridor_m)


def _route_bbox_polygon(projected_segments: gpd.GeoDataFrame, corridor_m: float) -> object:
    """Return an expanded route bbox polygon in local OSM CRS for coarse spatial reads."""
    projected_bbox = box(*_expanded_route_bbox(projected_segments, corridor_m))
    return _project_geometry_to_crs(projected_bbox, PROJECTED_CRS, LOCAL_OSM_CRS)


def _tile_range(min_value: float, max_value: float, tile_size: float) -> range:
    """Return integer tile ids covering a numeric interval."""
    start = math.floor(min_value / tile_size)
    stop = math.floor(max_value / tile_size)
    return range(start, stop + 1)


def _tile_id(ix: int, iy: int) -> str:
    """Return a stable tile id for x/y tile coordinates."""
    return f"x{ix}_y{iy}"


def _tile_ids_for_bounds(bounds: tuple[float, float, float, float], tile_size: float = LOCAL_OSM_TILE_SIZE_DEG) -> list[str]:
    """Return tile ids intersecting the provided lon/lat bounds."""
    minx, miny, maxx, maxy = bounds
    return [
        _tile_id(ix, iy)
        for ix in _tile_range(minx, maxx, tile_size)
        for iy in _tile_range(miny, maxy, tile_size)
    ]


def _read_tiled_geo_parquet(tile_dir: Path, tile_ids: list[str]) -> gpd.GeoDataFrame:
    """Read all existing parquet tiles for the requested ids and concatenate them."""
    paths = [tile_dir / f"{tile_id}.parquet" for tile_id in tile_ids]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)

    frames = [gpd.read_parquet(path) for path in existing]
    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)


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


def _filter_edges_to_segment_corridor(
    edges: gpd.GeoDataFrame,
    projected_segments: gpd.GeoDataFrame,
    corridor_m: float,
) -> gpd.GeoDataFrame:
    """Keep only edges intersecting any buffered route segment, avoiding a global union."""
    if edges.empty or projected_segments.empty:
        return gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)

    buffered_segments = gpd.GeoDataFrame(
        geometry=projected_segments.geometry.buffer(corridor_m),
        crs=PROJECTED_CRS,
    ).reset_index(names="_segment_index")
    projected_edges = edges.to_crs(PROJECTED_CRS).reset_index(names="_edge_index")
    matched = gpd.sjoin(
        projected_edges[["_edge_index", "geometry"]],
        buffered_segments,
        how="inner",
        predicate="intersects",
    )
    if matched.empty:
        return gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)

    edge_ids = matched["_edge_index"].drop_duplicates()
    filtered = projected_edges[projected_edges["_edge_index"].isin(edge_ids)].drop(columns=["_edge_index"])
    return filtered.set_crs(PROJECTED_CRS).to_crs(edges.crs)


def _build_match_windows(
    projected_segments: gpd.GeoDataFrame,
    match_window_size: int,
) -> gpd.GeoDataFrame:
    """Build rolling segment windows so matching uses local route context."""
    if projected_segments.empty:
        return gpd.GeoDataFrame(geometry=[], crs=PROJECTED_CRS)

    if match_window_size <= 1:
        return projected_segments[["geometry"]].copy().reset_index().rename(columns={"index": "_segment_index"})

    window_size = max(1, int(match_window_size))
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2

    segment_frame = projected_segments[["geometry"]].copy().reset_index().rename(columns={"index": "_segment_index"})
    geometries = list(segment_frame.geometry)
    window_geometries: list[LineString] = []

    for center_idx in range(len(geometries)):
        start_idx = max(0, center_idx - radius)
        stop_idx = min(len(geometries), center_idx + radius + 1)
        coords = list(geometries[start_idx].coords)
        for geom in geometries[start_idx + 1:stop_idx]:
            coords.extend(list(geom.coords)[1:])
        window_geometries.append(LineString(coords))

    return gpd.GeoDataFrame(
        {"_segment_index": segment_frame["_segment_index"]},
        geometry=window_geometries,
        crs=PROJECTED_CRS,
    )


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


def stop_signs_on_segments(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str = LOCAL_OSM_NETWORK_TYPE,
    corridor_m: float = 6.0,
    segment_buffer_m: float = 8.0,
    retain_all: bool = True
) -> gpd.GeoDataFrame:
    """Find stop/traffic light controls near route segments from the OpenStreetMap network."""
    projected_segments, nodes, _ = build_route_graph(gdf_segments, network_type, corridor_m, retain_all)

    if nodes.empty or "highway" not in nodes.columns:
        return nodes.iloc[0:0].to_crs(gdf_segments.crs)

    allowed = {"stop", "traffic_signals"}
    controls = nodes[nodes["highway"].apply(lambda value: value in allowed or (isinstance(value, list) and any(tag in allowed for tag in value)))].copy()

    if controls.empty:
        return controls.to_crs(gdf_segments.crs)

    projected_route_buffer = projected_segments.geometry.union_all().buffer(segment_buffer_m)
    projected_controls = controls.to_crs(PROJECTED_CRS)
    controls_on_route = projected_controls[projected_controls.geometry.intersects(projected_route_buffer)].copy()
    controls_on_route = controls_on_route.to_crs(gdf_segments.crs)

    if controls_on_route.empty:
        return controls_on_route

    controls_wgs84 = controls_on_route.to_crs(4326)
    controls_on_route["More Details"] = (
        '<a href="https://www.google.com/maps?q='
        + controls_wgs84.geometry.y.astype(str)
        + ","
        + controls_wgs84.geometry.x.astype(str)
        + '" target="_blank">📍 Open in Google Maps</a>'
    )
    return controls_on_route


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
    route_bbox_poly = _route_bbox_polygon(projected_segments, corridor_m)

    # network_type is retained for API compatibility, but all local OSM work
    # uses the prebuilt all_public Bay Area graph.
    _ = (network_type, retain_all)

    edges = _load_local_osm_edges(route_bbox_poly)
    edges = _filter_edges_to_segment_corridor(edges, projected_segments, corridor_m)
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

    # For each segment, find the nearest OSM edge within max distance.
    matched = gpd.sjoin_nearest(
        left,
        right,
        how="left",
        max_distance=match_max_distance_m,
        distance_col="_edge_dist_m",
        exclusive=False,
    )

    # If multiple candidates are near-tied, prefer road-like highway classes over path-like ones.
    matched["_min_edge_dist_m"] = matched.groupby("_segment_index")["_edge_dist_m"].transform("min")
    matched["_within_pref_tolerance"] = (
        matched["_edge_dist_m"] <= matched["_min_edge_dist_m"] + match_preference_tolerance_m
    )
    matched["_candidate_priority"] = matched["_highway_priority"].where(matched["_within_pref_tolerance"], 999)
    matched = matched.sort_values(
        by=["_segment_index", "_candidate_priority", "_edge_dist_m"],
        kind="stable",
    ).drop_duplicates(subset=["_segment_index"], keep="first")

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
