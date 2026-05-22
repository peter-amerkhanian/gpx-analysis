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
VITAL_SIGNS_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "vital-signs"
LOCAL_OSM_NODES_PATH = OSM_DATA_DIR / "sf_bay_area_all_public_nodes.parquet"
LOCAL_OSM_EDGES_PATH = OSM_DATA_DIR / "sf_bay_area_all_public_edges.parquet"
LOCAL_MTC_STREETS_PATH = VITAL_SIGNS_DATA_DIR / "Streets_and_Roads_20260512.geojson"
LOCAL_MTC_STREETS_PARQUET_PATH = VITAL_SIGNS_DATA_DIR / "Streets_and_Roads_20260512.parquet"
LOCAL_MTC_STREET_ATTRS = [
    "start_location",
    "end_location",
    "road_name",
    "pci_date",
    "pci_info",
]
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

def points_frame(points: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs=f"EPSG:{LOCAL_OSM_CRS}",
    )

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
    if {"track", "path", "footway"} & tags:
        return "gravel"
    return "road"


def _require_local_osm_cache() -> None:
    """Ensure the local GeoParquet OSM cache exists on disk."""
    if not LOCAL_OSM_NODES_PATH.exists() or not LOCAL_OSM_EDGES_PATH.exists():
        raise FileNotFoundError(
            "Local OSM GeoParquet cache not found. Run download_bay_area_osm.py "
            f"to create {LOCAL_OSM_NODES_PATH} and {LOCAL_OSM_EDGES_PATH}."
        )


def _require_local_mtc_streets() -> None:
    """Ensure at least one local MTC streets source file exists on disk."""
    if not LOCAL_MTC_STREETS_PATH.exists() and not LOCAL_MTC_STREETS_PARQUET_PATH.exists():
        raise FileNotFoundError(
            "Local MTC streets source not found. Expected "
            f"{LOCAL_MTC_STREETS_PATH} or {LOCAL_MTC_STREETS_PARQUET_PATH}."
        )


def _ensure_local_mtc_streets_parquet() -> Path:
    """Create a parquet copy of the local MTC streets GeoJSON when needed."""
    _require_local_mtc_streets()
    if LOCAL_MTC_STREETS_PARQUET_PATH.exists():
        return LOCAL_MTC_STREETS_PARQUET_PATH
    if not LOCAL_MTC_STREETS_PATH.exists():
        raise FileNotFoundError(
            "Local MTC streets GeoJSON not found. Expected "
            f"{LOCAL_MTC_STREETS_PATH}."
        )

    streets = gpd.read_file(LOCAL_MTC_STREETS_PATH)
    if streets.crs is None:
        streets = streets.set_crs(LOCAL_OSM_CRS)
    try:
        streets.to_parquet(LOCAL_MTC_STREETS_PARQUET_PATH, write_covering_bbox=True)
    except TypeError:
        streets.to_parquet(LOCAL_MTC_STREETS_PARQUET_PATH)
    return LOCAL_MTC_STREETS_PARQUET_PATH


def _project_geometry_to_crs(geometry: object, source_crs: object, target_crs: object) -> object:
    """Project a single shapely geometry between CRS definitions."""
    return gpd.GeoSeries([geometry], crs=source_crs).to_crs(target_crs).iloc[0]


def _normalize_match_text(value: object) -> str | None:
    """Return a lowercased, trimmed string for fuzzy name comparisons."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def _levenshtein_distance(left: str | None, right: str | None) -> int | None:
    """Return the Levenshtein distance between two normalized strings."""
    left = _normalize_match_text(left)
    right = _normalize_match_text(right)
    if left is None or right is None:
        return None
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _longest_linestring(geometry: object) -> LineString | None:
    """Return the longest linear component for bearing-based comparisons."""
    if geometry is None:
        return None
    geom_type = getattr(geometry, "geom_type", None)
    if geom_type == "LineString":
        return geometry
    if geom_type == "MultiLineString":
        parts = [part for part in geometry.geoms if getattr(part, "length", 0) > 0]
        if not parts:
            return None
        return max(parts, key=lambda part: part.length)
    return None


def _linestring_bearing_degrees(geometry: object) -> float | None:
    """Return the approximate bearing of the geometry from first to last vertex."""
    line = _longest_linestring(geometry)
    if line is None:
        return None
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    start_x, start_y = coords[0]
    end_x, end_y = coords[-1]
    dx = end_x - start_x
    dy = end_y - start_y
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx)) % 180


def _bearing_difference_degrees(left: object, right: object) -> float | None:
    """Return the smallest absolute bearing difference between two linear geometries."""
    left_bearing = _linestring_bearing_degrees(left)
    right_bearing = _linestring_bearing_degrees(right)
    if left_bearing is None or right_bearing is None:
        return None
    diff = abs(left_bearing - right_bearing)
    return min(diff, 180 - diff)


def _overlap_length_m(route_geometry: object, candidate_geometry: object, overlap_buffer_m: float) -> float:
    """Return the candidate length overlapping a buffered route geometry."""
    if route_geometry is None or candidate_geometry is None:
        return 0.0
    return candidate_geometry.intersection(route_geometry.buffer(overlap_buffer_m)).length


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


def _load_local_mtc_streets(route_poly: object) -> gpd.GeoDataFrame:
    """Load only the local MTC streets intersecting the route bbox."""
    parquet_path = _ensure_local_mtc_streets_parquet()
    columns = LOCAL_MTC_STREET_ATTRS + ["geometry"]
    try:
        streets = gpd.read_parquet(parquet_path, columns=columns, bbox=route_poly.bounds)
    except ValueError:
        streets = gpd.read_parquet(parquet_path, columns=columns)
        streets = streets[streets.intersects(route_poly)].copy()
    if streets.empty:
        return gpd.GeoDataFrame(geometry=[], crs=LOCAL_OSM_CRS)

    if streets.crs is None:
        streets = streets.set_crs(LOCAL_OSM_CRS)
    return streets


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


def _join_candidates_within_distance(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    max_distance_m: float,
) -> pd.DataFrame:
    """Return all right-side candidates within max distance of each left geometry."""
    if left.empty or right.empty:
        return pd.DataFrame()

    left_buffered = gpd.GeoDataFrame(
        left.drop(columns="geometry").copy(),
        geometry=left.geometry.buffer(max_distance_m),
        crs=left.crs,
    )
    matched = gpd.sjoin(
        left_buffered,
        right,
        how="inner",
        predicate="intersects",
    )
    if matched.empty:
        return matched

    right_indexed = right.reset_index().rename(columns={"index": "_candidate_index"}).set_index("_candidate_index")
    left_geometry_by_segment = left.set_index("_segment_index").geometry
    matched["_route_geometry"] = matched["_segment_index"].map(left_geometry_by_segment)
    matched["_candidate_geometry"] = matched["index_right"].map(right_indexed.geometry)
    matched["_candidate_dist_m"] = matched.apply(
        lambda row: row["_route_geometry"].distance(row["_candidate_geometry"]),
        axis=1,
    )
    return matched[matched["_candidate_dist_m"] <= max_distance_m].copy()


def _score_mtc_match_candidates(
    matched: pd.DataFrame,
    overlap_buffer_m: float,
    match_preference_tolerance_m: float,
) -> pd.DataFrame:
    """Score route-to-street candidates using distance, name, overlap, and bearing."""
    if matched.empty:
        return matched

    matched = matched.copy()
    matched["_min_candidate_dist_m"] = matched.groupby("_segment_index")["_candidate_dist_m"].transform("min")
    matched["_within_pref_tolerance"] = (
        matched["_candidate_dist_m"] <= matched["_min_candidate_dist_m"] + match_preference_tolerance_m
    )
    matched["_bearing_diff_deg"] = matched.apply(
        lambda row: _bearing_difference_degrees(row["_route_geometry"], row["_candidate_geometry"]),
        axis=1,
    )
    matched["_bearing_diff_deg"] = matched["_bearing_diff_deg"].fillna(999.0)
    matched["_overlap_length_m"] = matched.apply(
        lambda row: _overlap_length_m(row["_route_geometry"], row["_candidate_geometry"], overlap_buffer_m),
        axis=1,
    )
    matched["_osm_name_norm"] = matched.get("osm_name", pd.Series(index=matched.index, dtype="object")).apply(_normalize_match_text)
    matched["_mtc_road_name_norm"] = matched.get("road_name", pd.Series(index=matched.index, dtype="object")).apply(_normalize_match_text)
    matched["_name_distance"] = matched.apply(
        lambda row: _levenshtein_distance(row["_osm_name_norm"], row["_mtc_road_name_norm"]),
        axis=1,
    )
    matched["_has_name_distance"] = matched["_name_distance"].notna()
    matched["_name_distance_rank"] = matched["_name_distance"].fillna(999)
    matched["_tolerance_rank"] = (~matched["_within_pref_tolerance"]).astype(int)
    matched["_name_rank"] = (~matched["_has_name_distance"]).astype(int)
    return matched


def _select_best_mtc_match_per_segment(
    matched: pd.DataFrame,
    overlap_buffer_m: float,
    match_preference_tolerance_m: float,
) -> pd.DataFrame:
    """Return one best-scoring MTC candidate row per route segment."""
    scored = _score_mtc_match_candidates(
        matched,
        overlap_buffer_m=overlap_buffer_m,
        match_preference_tolerance_m=match_preference_tolerance_m,
    )
    if scored.empty:
        return scored

    scored = scored.sort_values(
        by=[
            "_segment_index",
            "_tolerance_rank",
            "_overlap_length_m",
            "_bearing_diff_deg",
            "_name_rank",
            "_name_distance_rank",
            "_candidate_dist_m",
        ],
        ascending=[True, True, False, True, True, True, True],
        kind="stable",
    )
    return scored.drop_duplicates(subset=["_segment_index"], keep="first")


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


def enrich_segments_with_mtc_streets(
    gdf_segments: gpd.GeoDataFrame,
    corridor_m: float = 10.0,
    match_max_distance_m: float = 25.0,
    match_preference_tolerance_m: float = 8.0,
    match_window_size: int = 10,
) -> gpd.GeoDataFrame:
    """Return a copy of route segments enriched with best-scoring MTC street attributes."""
    street_attrs = LOCAL_MTC_STREET_ATTRS
    result = gdf_segments.copy()

    output_cols = [f"mtc_{col}" for col in street_attrs]
    for col in output_cols:
        if col not in result.columns:
            result[col] = pd.NA
    if result.empty:
        return result

    if gdf_segments.crs is None:
        raise ValueError("gdf_segments must have a CRS.")

    projected_segments = gdf_segments.to_crs(PROJECTED_CRS)
    route_bbox_poly = _route_bbox_polygon(projected_segments, corridor_m)
    streets = _load_local_mtc_streets(route_bbox_poly)
    streets = _filter_edges_to_segment_corridor(streets, projected_segments, corridor_m)
    if streets.empty:
        return result

    available_street_attrs = [col for col in street_attrs if col in streets.columns]
    if not available_street_attrs:
        return result

    streets_subset = streets[available_street_attrs + ["geometry"]].copy()
    left = _build_match_windows(projected_segments, match_window_size)
    segment_attrs = result.reset_index().rename(columns={"index": "_segment_index"})
    if "osm_name" in segment_attrs.columns:
        left = left.merge(segment_attrs[["_segment_index", "osm_name"]], on="_segment_index", how="left")
    right = gpd.GeoDataFrame(streets_subset, geometry="geometry", crs=streets.crs).to_crs(PROJECTED_CRS)

    matched = _join_candidates_within_distance(
        left,
        right,
        max_distance_m=match_max_distance_m,
    )
    if matched.empty:
        return result

    matched = _select_best_mtc_match_per_segment(
        matched,
        overlap_buffer_m=corridor_m,
        match_preference_tolerance_m=match_preference_tolerance_m,
    )
    if matched.empty:
        return result

    attrs = pd.DataFrame({"_segment_index": matched["_segment_index"]})
    for col in available_street_attrs:
        attrs[f"mtc_{col}"] = matched[col]

    result = result.reset_index().rename(columns={"index": "_segment_index"}).merge(
        attrs,
        on="_segment_index",
        how="left",
        suffixes=("", "_matched"),
    )
    for col in available_street_attrs:
        colname = f"mtc_{col}"
        matched_col = f"{colname}_matched"
        result[colname] = result[matched_col].combine_first(result[colname])
        result = result.drop(columns=[matched_col])

    result = result.set_index("_segment_index")
    result.index.name = gdf_segments.index.name
    result_gdf = gpd.GeoDataFrame(result, geometry="geometry", crs=gdf_segments.crs)
    if "road_type" in result_gdf.columns:
        result_gdf.loc[(result_gdf["road_type"] == "gravel"), "mtc_pci_info"] = "Gravel"
        result_gdf.loc[(result_gdf["osm_highway"] == "cycleway"), "mtc_pci_info"] = "Cycleway"
        missing_pci = result_gdf["mtc_pci_info"].isna()
        result_gdf["pci_available"] = "PCI Available"
        result_gdf.loc[missing_pci, "pci_available"] = "PCI Unknown"
        result_gdf.loc[result_gdf["mtc_pci_info"].isna(), "mtc_pci_info"] = (
            result_gdf.loc[missing_pci, "osm_highway"].str.title() + " (Unknown)"
        )
    return result_gdf
