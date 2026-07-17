import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from .constants import LOCAL_OSM_CRS, LOCAL_OSM_TILE_SIZE_DEG, PROJECTED_CRS

def _project_geometry_to_crs(geometry: object, source_crs: object, target_crs: object) -> object:
    """Project a single shapely geometry between CRS definitions."""
    return gpd.GeoSeries([geometry], crs=source_crs).to_crs(target_crs).iloc[0]

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
    right_with_key = right.reset_index(drop=True).copy()
    right_with_key["_candidate_index"] = right_with_key.index
    matched = gpd.sjoin(
        left_buffered,
        right_with_key,
        how="inner",
        predicate="intersects",
    )
    if matched.empty:
        return matched

    right_indexed = right_with_key.set_index("_candidate_index")
    left_geometry_by_segment = left.set_index("_segment_index").geometry
    matched["_route_geometry"] = matched["_segment_index"].map(left_geometry_by_segment)
    matched["_candidate_geometry"] = matched["_candidate_index"].map(right_indexed.geometry)
    route_geometry = gpd.GeoSeries(matched["_route_geometry"], crs=left.crs)
    candidate_geometry = gpd.GeoSeries(matched["_candidate_geometry"], crs=right.crs)
    matched["_candidate_dist_m"] = route_geometry.distance(
        candidate_geometry,
        align=False,
    )
    return matched[matched["_candidate_dist_m"] <= max_distance_m].copy()
