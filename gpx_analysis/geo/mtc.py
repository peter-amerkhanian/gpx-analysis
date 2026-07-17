from pathlib import Path
from typing import Sequence

import geopandas as gpd
import pandas as pd

from .constants import (
    LOCAL_MTC_STREET_ATTRS,
    LOCAL_MTC_STREETS_PARQUET_PATH,
    LOCAL_MTC_STREETS_PATH,
    LOCAL_OSM_CRS,
    PROJECTED_CRS,
)
from .matching import _select_best_mtc_match_per_segment
from .names import _road_name_key
from .spatial import (
    _build_match_windows,
    _filter_edges_to_segment_corridor,
    _join_candidates_within_distance,
    _route_bbox_polygon,
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

    print(
        "Warning: local MTC streets parquet not found. "
        f"Reading {LOCAL_MTC_STREETS_PATH} and doing a one-time conversion to "
        f"{LOCAL_MTC_STREETS_PARQUET_PATH}."
    )
    streets = gpd.read_file(LOCAL_MTC_STREETS_PATH)
    if streets.crs is None:
        streets = streets.set_crs(LOCAL_OSM_CRS)
    try:
        streets.to_parquet(LOCAL_MTC_STREETS_PARQUET_PATH, write_covering_bbox=True)
    except TypeError:
        streets.to_parquet(LOCAL_MTC_STREETS_PARQUET_PATH)
    return LOCAL_MTC_STREETS_PARQUET_PATH

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

def _unknown_pci_label_from_osm_highway(value: object) -> str:
    """Return a stable unknown PCI label even when OSM matching also failed."""
    if isinstance(value, (list, tuple, set)):
        value = ";".join(str(item) for item in value)
    if value is None or pd.isna(value):
        return "Roadway (Unknown)"
    text = str(value).strip()
    if not text or text == "<NA>":
        return "Roadway (Unknown)"
    return f"{text.title()} (Unknown)"

def _finalize_mtc_unknowns(result: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill unmatched MTC PCI rows so maps and reports never have blank categories."""
    result_gdf = result.copy()
    if "mtc_pci_info" not in result_gdf.columns:
        result_gdf["mtc_pci_info"] = pd.NA
    if "pci_available" not in result_gdf.columns:
        result_gdf["pci_available"] = "PCI Available"

    if "road_type" in result_gdf.columns:
        result_gdf.loc[result_gdf["road_type"] == "gravel", "mtc_pci_info"] = "Gravel"
    if "osm_highway" in result_gdf.columns:
        result_gdf.loc[result_gdf["osm_highway"] == "cycleway", "mtc_pci_info"] = "Cycleway"

    missing_pci = result_gdf["mtc_pci_info"].isna()
    result_gdf.loc[missing_pci, "pci_available"] = "PCI Unknown"
    if "osm_highway" in result_gdf.columns:
        unknown_labels = result_gdf.loc[missing_pci, "osm_highway"].apply(_unknown_pci_label_from_osm_highway)
    else:
        unknown_labels = pd.Series("Roadway (Unknown)", index=result_gdf.index[missing_pci], dtype="object")
    result_gdf.loc[missing_pci, "mtc_pci_info"] = unknown_labels
    return gpd.GeoDataFrame(result_gdf, geometry="geometry", crs=result.crs)

def _fill_mtc_gaps_from_osm_continuity(
    result: gpd.GeoDataFrame,
    street_attrs: Sequence[str] = LOCAL_MTC_STREET_ATTRS,
) -> gpd.GeoDataFrame:
    """Bridge short unmatched MTC runs when OSM and both MTC neighbors agree on the road."""
    result_gdf = result.copy()
    if "osm_name" not in result_gdf.columns or "mtc_road_name" not in result_gdf.columns:
        return gpd.GeoDataFrame(result_gdf, geometry="geometry", crs=result.crs)

    mtc_attr_cols = [f"mtc_{col}" for col in street_attrs if f"mtc_{col}" in result_gdf.columns]
    if not mtc_attr_cols:
        return gpd.GeoDataFrame(result_gdf, geometry="geometry", crs=result.crs)

    missing_mtc_name = result_gdf["mtc_road_name"].isna()
    if not missing_mtc_name.any():
        return gpd.GeoDataFrame(result_gdf, geometry="geometry", crs=result.crs)

    row_labels = list(result_gdf.index)

    pos = 0
    while pos < len(row_labels):
        label = row_labels[pos]
        if not bool(missing_mtc_name.loc[label]):
            pos += 1
            continue

        run_start = pos
        while pos < len(row_labels) and bool(missing_mtc_name.loc[row_labels[pos]]):
            pos += 1
        run_end = pos

        prev_pos = run_start - 1
        next_pos = run_end
        if prev_pos < 0 or next_pos >= len(row_labels):
            continue

        prev_label = row_labels[prev_pos]
        next_label = row_labels[next_pos]
        prev_key = _road_name_key(result_gdf.at[prev_label, "mtc_road_name"])
        next_key = _road_name_key(result_gdf.at[next_label, "mtc_road_name"])
        if prev_key is None or prev_key != next_key:
            continue

        run_labels = row_labels[run_start:run_end]
        run_osm_keys = {
            key
            for key in result_gdf.loc[run_labels, "osm_name"].apply(_road_name_key)
            if key is not None
        }
        if run_osm_keys != {prev_key}:
            continue

        for run_label in run_labels:
            for col in mtc_attr_cols:
                if pd.isna(result_gdf.at[run_label, col]):
                    result_gdf.at[run_label, col] = result_gdf.at[prev_label, col]

    return gpd.GeoDataFrame(result_gdf, geometry="geometry", crs=result.crs)

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
        return _finalize_mtc_unknowns(result)

    if gdf_segments.crs is None:
        raise ValueError("gdf_segments must have a CRS.")

    projected_segments = gdf_segments.to_crs(PROJECTED_CRS)
    candidate_corridor_m = max(corridor_m, match_max_distance_m)
    route_bbox_poly = _route_bbox_polygon(projected_segments, candidate_corridor_m)
    streets = _load_local_mtc_streets(route_bbox_poly)
    streets = _filter_edges_to_segment_corridor(streets, projected_segments, candidate_corridor_m)
    if streets.empty:
        return _finalize_mtc_unknowns(result)

    available_street_attrs = [col for col in street_attrs if col in streets.columns]
    if not available_street_attrs:
        return _finalize_mtc_unknowns(result)

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
        return _finalize_mtc_unknowns(result)

    matched = _select_best_mtc_match_per_segment(
        matched,
        overlap_buffer_m=corridor_m,
        match_preference_tolerance_m=match_preference_tolerance_m,
    )
    if matched.empty:
        return _finalize_mtc_unknowns(result)

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
    result_gdf = _fill_mtc_gaps_from_osm_continuity(result_gdf, available_street_attrs)
    return _finalize_mtc_unknowns(result_gdf)
