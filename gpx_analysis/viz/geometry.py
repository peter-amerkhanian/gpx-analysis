from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
def _marker_point_and_normal(
    segment: object,
    fallback_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a segment start point and a unit normal for label offsets."""
    coords = np.asarray(segment.coords, dtype=float)
    start = coords[0]
    end = coords[-1]
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm == 0:
        return start, np.array([0.0, float(fallback_sign)])
    unit_direction = direction / norm
    normal = np.array([-unit_direction[1], unit_direction[0]])
    return start, normal



def _resolve_number_marker_locations(
    frame: gpd.GeoDataFrame,
    marker_indexes: list[int],
    min_spacing_m: float = 520.0,
    base_offset_m: float = 18.0,
    offset_step_m: float = 28.0,
    max_attempts: int = 6,
) -> list[list[float]]:
    """Place number markers near their segments while avoiding overlap."""
    projected = frame[["geometry"]].to_crs(3857)
    placed_points: list[np.ndarray] = []
    locations: list[list[float]] = []

    for marker_order, marker_index in enumerate(marker_indexes):
        segment = projected.iloc[marker_index].geometry
        base_point, normal = _marker_point_and_normal(
            segment,
            fallback_sign=1 if marker_order % 2 == 0 else -1,
        )

        candidate = base_point
        for attempt in range(max_attempts):
            offset_scale = base_offset_m + (attempt * offset_step_m)
            if marker_order % 2 == 1:
                offset_scale *= -1
            candidate = base_point + (normal * offset_scale)
            if all(np.linalg.norm(candidate - placed) >= min_spacing_m for placed in placed_points):
                break

        placed_points.append(candidate)
        point_wgs84 = (
            gpd.GeoSeries([Point(candidate[0], candidate[1])], crs=3857)
            .to_crs(4326)
            .iloc[0]
        )
        locations.append([point_wgs84.y, point_wgs84.x])

    return locations



def _number_marker_count(frame: gpd.GeoDataFrame) -> int:
    """Return the number of numbered route markers based on route length."""
    distance_m = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0).sum()
    distance_mi = distance_m / 1609.344
    return max(3, 3 + int(distance_mi // 15))



def _number_marker_indexes(frame: gpd.GeoDataFrame) -> list[int]:
    """Spread numbered markers across the route, excluding the start marker."""
    marker_count = _number_marker_count(frame)
    last_index = len(frame) - 1
    if last_index <= 0:
        return [0] * marker_count

    fractions = np.linspace(
        1 / (marker_count + 10),
        marker_count / (marker_count + 1),
        marker_count,
    )
    indexes = [min(max(1, int(round(last_index * fraction))), last_index) for fraction in fractions]
    return indexes



def _route_chevron_dimensions(frame: gpd.GeoDataFrame) -> tuple[float, float]:
    """Scale chevron geometry by overall route length, anchored to a ~30 mile route."""
    distance_m = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0.0).sum()
    distance_mi = distance_m / 1609.344
    scale = float(np.clip(distance_mi / 30.0, 0.55, 1.15))
    return 350.0 * scale, 250.0 * scale



def _chevron_paths_for_segment(
    segment: object,
    chevron_length_m: float,
    chevron_half_width_m: float,
) -> list[list[list[float]]]:
    """Return two WGS84 line paths forming a centered chevron for the segment."""
    coords = np.asarray(segment.coords, dtype=float)
    if len(coords) < 2:
        return []

    start = coords[0]
    end = coords[-1]
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm == 0:
        return []

    unit_direction = direction / norm
    unit_normal = np.array([-unit_direction[1], unit_direction[0]])
    midpoint = (start + end) / 2.0
    tip = midpoint + (unit_direction * (chevron_length_m / 2.0))
    tail_center = midpoint - (unit_direction * (chevron_length_m / 2.0))
    left = tail_center + (unit_normal * chevron_half_width_m)
    right = tail_center - (unit_normal * chevron_half_width_m)

    chevron_lines = gpd.GeoSeries(
        [
            LineString([tuple(left), tuple(tip)]),
            LineString([tuple(right), tuple(tip)]),
        ],
        crs=3857,
    ).to_crs(4326)
    return [[[lat, lon] for lon, lat in line.coords] for line in chevron_lines]



def _chevron_midpoint(segment: object) -> np.ndarray | None:
    """Return the projected midpoint of a segment for chevron overlap checks."""
    coords = np.asarray(segment.coords, dtype=float)
    if len(coords) < 2:
        return None
    start = coords[0]
    end = coords[-1]
    if np.allclose(end, start):
        return None
    return (start + end) / 2.0



def _segment_indexes_with_route_overlap(
    projected: gpd.GeoDataFrame,
    overlap_proximity_m: float = 20.0,
    min_shared_length_m: float = 25.0,
) -> set[int]:
    """Return segment indexes that truly share route geometry with non-adjacent segments."""
    if projected.empty:
        return set()

    buffered = projected[["geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(overlap_proximity_m)
    overlap_indexes: set[int] = set()
    joined = gpd.sjoin(
        projected[["geometry"]],
        buffered[["geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
    )
    joined = joined[
        (joined.index != joined["index_right"])
        & ((joined.index - joined["index_right"]).abs() > 1)
    ]
    if joined.empty:
        return overlap_indexes

    geometries = projected.geometry
    for segment_index, match_index in joined[["index_right"]].itertuples(index=True, name=None):
        left = geometries.loc[segment_index]
        right = geometries.loc[match_index]
        shared_length_m = left.intersection(right).length
        if shared_length_m >= min_shared_length_m:
            overlap_indexes.add(int(segment_index))
            overlap_indexes.add(int(match_index))
    return overlap_indexes



def _frames_share_route_overlap(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    overlap_proximity_m: float = 20.0,
    min_shared_length_m: float = 25.0,
    column: str | None = None,
    ignore_value: str | None = None,
) -> bool:
    """Return True when two route halves share meaningful geometry that merits a pass control."""
    if left.empty or right.empty:
        return False

    right_buffered = right[["geometry"]].copy()
    right_buffered["geometry"] = right_buffered.geometry.buffer(overlap_proximity_m)
    joined = gpd.sjoin(
        left[["geometry"]],
        right_buffered[["geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
    )
    if joined.empty:
        return False

    left_geometries = left.geometry
    right_geometries = right.geometry
    has_meaningful_overlap = False
    for left_index, right_index in joined[["index_right"]].itertuples(index=True, name=None):
        shared_length_m = left_geometries.loc[left_index].intersection(right_geometries.loc[right_index]).length
        if shared_length_m >= min_shared_length_m:
            has_meaningful_overlap = True
            if column is None or ignore_value is None:
                return True

            left_value = left.loc[left_index, column] if column in left.columns else None
            right_value = right.loc[right_index, column] if column in right.columns else None
            if not (left_value == ignore_value and right_value == ignore_value):
                return True

    return False if has_meaningful_overlap else False



def _route_overlap_pass_indexes(
    projected: gpd.GeoDataFrame,
    overlap_proximity_m: float = 20.0,
    min_shared_length_m: float = 25.0,
    column: str | None = None,
    ignore_value: str | None = None,
) -> tuple[set[object], set[object]]:
    """Return earlier/later route-pass indexes for segments that overlap another pass."""
    if projected.empty:
        return set(), set()

    buffered = projected[["geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(overlap_proximity_m)
    joined = gpd.sjoin(
        projected[["geometry"]],
        buffered[["geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
    )
    route_positions = {index: position for position, index in enumerate(projected.index)}
    joined["_left_pos"] = joined.index.map(route_positions)
    joined["_right_pos"] = joined["index_right"].map(route_positions)
    joined = joined[
        (joined.index != joined["index_right"])
        & ((joined["_left_pos"] - joined["_right_pos"]).abs() > 1)
    ]
    if joined.empty:
        return set(), set()

    geometries = projected.geometry
    earlier_pass: set[object] = set()
    later_pass: set[object] = set()
    seen_pairs: set[frozenset[object]] = set()
    for left_index, right_index in joined[["index_right"]].itertuples(index=True, name=None):
        pair = frozenset((left_index, right_index))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        left = geometries.loc[left_index]
        right = geometries.loc[right_index]
        shared_left_length_m = left.intersection(right.buffer(overlap_proximity_m)).length
        shared_right_length_m = right.intersection(left.buffer(overlap_proximity_m)).length
        shared_length_m = min(shared_left_length_m, shared_right_length_m)
        if shared_length_m < min_shared_length_m:
            continue

        if column is not None and ignore_value is not None and column in projected.columns:
            left_value = projected.loc[left_index, column]
            right_value = projected.loc[right_index, column]
            if left_value == ignore_value and right_value == ignore_value:
                continue

        if route_positions[left_index] < route_positions[right_index]:
            earlier_pass.add(left_index)
            later_pass.add(right_index)
        else:
            earlier_pass.add(right_index)
            later_pass.add(left_index)

    return earlier_pass, later_pass



def _overlap_ignore_value(column: str | None) -> str | None:
    """Return the route category that should not trigger pass splitting on shared geometry."""
    if column == "hazard":
        return "flat"
    return None



def _chevron_marker_segments(
    frame: gpd.GeoDataFrame,
    spacing_fraction: int = 9,
    min_segment_length_m: float = 100.0,
) -> list[list[list[list[float]]]]:
    """Return route-spaced chevron paths for sufficiently long, non-overlapping segments."""
    if frame.empty:
        return []

    projected = frame[["geometry"]].to_crs(3857).copy()
    projected["segment_length_m"] = projected.geometry.length
    overlapping_indexes = _segment_indexes_with_route_overlap(projected)
    eligible = projected[projected["segment_length_m"] >= min_segment_length_m].copy()
    if overlapping_indexes:
        eligible = eligible[~eligible.index.isin(overlapping_indexes)]
    if eligible.empty:
        return []

    chevron_length_m, chevron_half_width_m = _route_chevron_dimensions(frame)
    min_chevron_spacing_m = chevron_length_m * 0.9
    target_count = max(1, spacing_fraction - 1)
    route_distances = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0.0)
    cumulative_end_m = route_distances.cumsum()
    total_distance_m = float(cumulative_end_m.iloc[-1]) if not cumulative_end_m.empty else 0.0
    if total_distance_m <= 0:
        target_indexes = list(eligible.index[:target_count])
    else:
        target_positions = np.linspace(
            total_distance_m / spacing_fraction,
            total_distance_m * (spacing_fraction - 1) / spacing_fraction,
            target_count,
        )
        eligible_end_m = cumulative_end_m.loc[eligible.index]
        used_indexes: set[int] = set()
        target_indexes: list[int] = []
        for target_position in target_positions:
            ranked_indexes = (
                (eligible_end_m - target_position)
                .abs()
                .sort_values()
                .index
            )
            for segment_index in ranked_indexes:
                if int(segment_index) not in used_indexes:
                    used_indexes.add(int(segment_index))
                    target_indexes.append(int(segment_index))
                    break

    chevrons: list[list[list[list[float]]]] = []
    accepted_midpoints: list[np.ndarray] = []
    for segment_index in target_indexes:
        segment = projected.loc[segment_index, "geometry"]
        midpoint = _chevron_midpoint(segment)
        if midpoint is None:
            continue
        if any(np.linalg.norm(midpoint - accepted) < min_chevron_spacing_m for accepted in accepted_midpoints):
            continue
        chevron_paths = _chevron_paths_for_segment(
            segment,
            chevron_length_m=chevron_length_m,
            chevron_half_width_m=chevron_half_width_m,
        )
        if chevron_paths:
            accepted_midpoints.append(midpoint)
            chevrons.append(chevron_paths)
    return chevrons



def _split_outbound_return(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split ordered route segments into outbound and return halves by cumulative distance."""
    if frame.empty:
        return frame.copy(), frame.iloc[0:0].copy()

    route_distances = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0.0)
    cumulative_end_m = route_distances.cumsum()
    total_distance_m = float(cumulative_end_m.iloc[-1]) if not cumulative_end_m.empty else 0.0
    split_distance_m = total_distance_m / 2.0
    outbound_mask = cumulative_end_m <= split_distance_m
    if outbound_mask.sum() == 0:
        outbound_mask.iloc[0] = True
    if outbound_mask.sum() == len(frame):
        outbound_mask.iloc[-1] = False
    outbound = frame.loc[outbound_mask].copy()
    returning = frame.loc[~outbound_mask].copy()
    return outbound, returning



def _combine_linestrings(geometries: pd.Series) -> LineString:
    """Combine ordered segment geometries into one continuous route section."""
    coords: list[tuple[float, float]] = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        line_geometries = getattr(geometry, "geoms", [geometry])
        for line in line_geometries:
            if not hasattr(line, "coords"):
                continue
            for coord in line.coords:
                point = (coord[0], coord[1])
                if coords and coords[-1] == point:
                    continue
                coords.append(point)

    if len(coords) < 2:
        return LineString()
    return LineString(coords)

