from __future__ import annotations

import folium
import geopandas as gpd
import pandas as pd

from .columns import _add_google_maps_details
from .folium_base import add_map_elements
from .formatters import (
    _format_average_grade_label,
    _format_percent,
    _middle_non_empty_value,
    _road_name_from_section_label,
    _safe_float,
)
from .geometry import _combine_linestrings
from .palettes import CHUNK_STATE_COLORS
def _numbered_chunk_section_label(route_part: str, road_name: str, climb_number: int | None) -> str:
    if route_part == "flat or descent":
        return route_part
    label = f"{road_name}: {route_part}"
    if climb_number is None:
        return label
    return f"{climb_number}. {label}"



def _chunk_map_section_label(
    route_part: str,
    road_name: str,
    average_grade: object,
    climb_number: int | None,
) -> str:
    if route_part == "flat or descent":
        return route_part

    grade_label = _format_average_grade_label(average_grade)
    label = f"{road_name} ({grade_label})" if grade_label else road_name
    if climb_number is None:
        return label
    return f"{climb_number}. {label}"



def _chunk_section_map_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return one map feature per contiguous chunk section."""
    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable")
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable")
    elif "end_i" in frame.columns:
        frame = frame.sort_values("end_i", kind="stable")
    else:
        frame = frame.sort_index(kind="stable")

    if "section_id" not in frame.columns:
        frame["section_id"] = frame["chunk_state"].ne(frame["chunk_state"].shift()).cumsum()
    if "section_distance_mi" not in frame.columns:
        frame["section_distance_mi"] = pd.to_numeric(
            frame.get("step_dist_f"),
            errors="coerce",
        ).fillna(0).groupby(frame["section_id"]).transform("sum") / 5280.0
    if "section_climb_gain_ft" not in frame.columns:
        frame["section_climb_gain_ft"] = pd.to_numeric(
            frame.get("step_elevation_f"),
            errors="coerce",
        ).clip(lower=0).fillna(0).groupby(frame["section_id"]).transform("sum")
    if "section_road_name" not in frame.columns:
        section_label_road_names = (
            frame["section_label"].apply(_road_name_from_section_label)
            if "section_label" in frame.columns
            else pd.Series(index=frame.index, dtype="object")
        )
        osm_road_names = frame.get("osm_name", pd.Series(index=frame.index, dtype="object"))
        frame["section_road_name"] = section_label_road_names.combine_first(osm_road_names).fillna("Unknown Road")
    if "section_label" not in frame.columns:
        section_states = frame.groupby("section_id")["chunk_state"].first()
        climb_numbers = {
            section_id: number
            for number, section_id in enumerate(
                section_states[section_states != "flat or descent"].index,
                start=1,
            )
        }
        frame["section_label"] = frame.apply(
            lambda row: _numbered_chunk_section_label(
                row["chunk_state"],
                row["section_road_name"],
                climb_numbers.get(row["section_id"]),
            ),
            axis=1,
        )

    rows: list[dict[str, object]] = []
    climb_number = 0
    for _, group in frame.groupby("section_id", sort=False):
        first = group.iloc[0]
        route_part = first["chunk_state"]
        climb_gain = _safe_float(first["section_climb_gain_ft"])
        distance_mi = _safe_float(first["section_distance_mi"])
        is_climb = route_part != "flat or descent"
        if is_climb:
            climb_number += 1
        road_name = _middle_non_empty_value(group["section_road_name"])
        section_label = _chunk_map_section_label(
            route_part,
            road_name,
            first.get("chunk_avg_grade"),
            climb_number if is_climb else None,
        )
        rows.append({
            "section_id": first["section_id"],
            "chunk_state": route_part,
            "Section": section_label,
            "Distance (mi)": round(distance_mi, 1),
            "Climb (ft)": f"{climb_gain:,.0f}" if is_climb else "",
            "Average Grade": _format_percent(first.get("chunk_avg_grade")) if is_climb else "",
            "Median Grade": _format_percent(first.get("chunk_median_grade")) if is_climb else "",
            "Section Time (min)": first.get("section_time_min", ""),
            "More Details": _middle_non_empty_value(
                group["More Details"] if "More Details" in group.columns else pd.Series(dtype="object"),
                fallback="",
            ),
            "_display_color": CHUNK_STATE_COLORS.get(route_part, "#8a8a8a"),
            "geometry": _combine_linestrings(group.geometry),
        })

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=frame.crs)



def _add_chunk_section_display_columns(
    frame: gpd.GeoDataFrame,
    section_frame: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Attach section-level popup fields to each segment for split pass layers."""
    frame = frame.copy()
    if "section_id" not in frame.columns and "chunk_state" in frame.columns:
        frame["section_id"] = frame["chunk_state"].ne(frame["chunk_state"].shift()).cumsum()

    display_columns = [
        "section_id",
        "Section",
        "Distance (mi)",
        "Climb (ft)",
        "Average Grade",
        "Median Grade",
        "Section Time (min)",
        "More Details",
    ]
    section_display = section_frame[
        [column for column in display_columns if column in section_frame.columns]
    ].copy()
    if "section_id" not in frame.columns or "section_id" not in section_display.columns:
        return frame

    frame_without_display = frame.drop(
        columns=[
            column
            for column in section_display.columns
            if column != "section_id" and column in frame.columns
        ],
        errors="ignore",
    )
    merged = frame_without_display.merge(section_display, on="section_id", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=frame.crs)



def make_chunk_map(
    gdf_segments: gpd.GeoDataFrame,
    popup_cols: list[str] | None = None,
    tooltip_fields: list[str] | None = None,
    tiles: str = "CartoDB Voyager",
    show_gravel_overlay: bool = True,
) -> folium.Map:
    """Build a Folium map with chunk-state colored segments and chunk detail popups/tooltips."""
    if "chunk_state" in gdf_segments.columns:
        frame = gdf_segments.copy()
    else:
        from ..chunks import detect_chunks

        frame = detect_chunks(gdf_segments)
    frame["Segment"] = frame["step"].astype("Int64").astype(str)
    frame = _add_google_maps_details(frame)
    frame["Road Name"] = frame["osm_name"].fillna("Unknown Road")
    frame["Turn"] = frame["step_turn"].round(2).astype(str) + "Â°"
    frame["Grade"] = frame["step_grade"].multiply(100).round(2).astype(str) + "%"
    frame["Chunk Avg Grade"] = frame["chunk_avg_grade"].multiply(100).round(2).astype(str) + "%"
    frame["Chunk Distance (ft)"] = frame["chunk_dist_ft"].round(0).astype("Int64").astype(str)
    frame["Candidate Chunk Distance (ft)"] = frame["candidate_chunk_dist_ft"].round(0).astype("Int64").astype(str)
    if "section_road_name" in frame.columns:
        frame["Section Road Name"] = frame["section_road_name"].fillna(frame["Road Name"])
    if "section_distance_mi" in frame.columns:
        frame["Section Distance (mi)"] = pd.to_numeric(
            frame["section_distance_mi"],
            errors="coerce",
        ).round(1)
    if "section_time_min" in frame.columns:
        frame["Section Time (min)"] = frame["section_time_min"].fillna("")
    frame["_display_color"] = frame["chunk_state"].map(CHUNK_STATE_COLORS).fillna("#8a8a8a")
    section_frame = _chunk_section_map_frame(frame)
    interaction_frame = _add_chunk_section_display_columns(frame, section_frame)

    if tooltip_fields is None:
        tooltip_fields = [
            "Section",
            "Distance (mi)",
            "Climb (ft)",
            "Average Grade",
            "Median Grade",
            "More Details",
        ]
    if popup_cols is None:
        popup_cols = [
            "Section",
            "Distance (mi)",
            "Climb (ft)",
            "Average Grade",
            "Median Grade",
            "Section Time (min)",
            "More Details",
        ]

    m = section_frame.explore(
        column="chunk_state",
        name="Route",
        tooltip=tooltip_fields,
        popup=popup_cols,
        tiles=tiles,
        categorical=True,
        cmap=list(CHUNK_STATE_COLORS.values()),
        categories=list(CHUNK_STATE_COLORS.keys()),
        legend=True,
        style_kwds={"weight": 4},
        escape=False,
    )
    m = add_map_elements(
        m,
        interaction_frame,
        show_route_pass_control=True,
        layer_column="chunk_state",
        popup_cols=popup_cols,
        tooltip_fields=tooltip_fields,
        categories=list(CHUNK_STATE_COLORS.keys()),
        cmap=list(CHUNK_STATE_COLORS.values()),
        style_kwds={"weight": 4},
        touch_target_frame=section_frame,
        show_gravel_overlay=show_gravel_overlay,
    )
    return m

