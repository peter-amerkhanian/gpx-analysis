from __future__ import annotations

import geopandas as gpd
import pandas as pd
import folium

from .viz import add_map_elements, google_maps_link

DESCENT_CHUNK_STATE_COLORS = {
    "other": "#bdbdbd",
    "descent": "#8c0f0f",
}


def detect_descent_chunks(
    df: pd.DataFrame,
    peak_speed_mph: float = 20.0,
    end_speed_mph: float = 5,
    min_chunk_dist_mi: float = 0.2,
) -> pd.DataFrame:
    """Classify sustained coasting-speed descent chunks."""
    min_chunk_dist_ft = min_chunk_dist_mi * 5280
    frame = df.copy()
    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable")
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable")
    elif "end_i" in frame.columns:
        frame = frame.sort_values("end_i", kind="stable")
    else:
        frame = frame.sort_index(kind="stable")

    if "coast_speed_mph" not in frame.columns:
        raise ValueError("detect_descent_chunks requires a coast_speed_mph column.")

    frame["descent_chunk_state"] = "other"
    frame["descent_chunk_label"] = "other"
    frame["descent_chunk_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["descent_chunk_dist_ft"] = pd.NA
    frame["descent_candidate_chunk_dist_ft"] = pd.NA
    frame["descent_chunk_max_speed_mph"] = pd.NA

    next_chunk_id = 1
    active_indices: list[object] = []

    def finalize(indices: list[object]) -> None:
        nonlocal next_chunk_id
        if not indices:
            return

        chunk = frame.loc[indices]
        chunk_distance_ft = float(pd.to_numeric(chunk["step_dist_f"], errors="coerce").fillna(0).sum())
        max_speed_mph = float(pd.to_numeric(chunk["coast_speed_mph"], errors="coerce").max())
        frame.loc[indices, "descent_candidate_chunk_dist_ft"] = chunk_distance_ft
        if chunk_distance_ft < min_chunk_dist_ft or max_speed_mph < peak_speed_mph:
            return

        frame.loc[indices, "descent_chunk_state"] = "descent"
        frame.loc[indices, "descent_chunk_label"] = "descent"
        frame.loc[indices, "descent_chunk_id"] = next_chunk_id
        frame.loc[indices, "descent_chunk_dist_ft"] = chunk_distance_ft
        frame.loc[indices, "descent_chunk_max_speed_mph"] = max_speed_mph
        next_chunk_id += 1

    speeds = pd.to_numeric(frame["coast_speed_mph"], errors="coerce")
    for idx, speed in speeds.items():
        is_above_end_speed = pd.notna(speed) and float(speed) > end_speed_mph
        if is_above_end_speed:
            active_indices.append(idx)
            continue

        finalize(active_indices)
        active_indices = []

    finalize(active_indices)
    return frame


def _combine_linestrings(geometries: pd.Series) -> object:
    coords: list[tuple[float, float]] = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        for line in getattr(geometry, "geoms", [geometry]):
            if not hasattr(line, "coords"):
                continue
            for coord in line.coords:
                point = (coord[0], coord[1])
                if coords and coords[-1] == point:
                    continue
                coords.append(point)
    if len(coords) < 2:
        return None
    from shapely.geometry import LineString

    return LineString(coords)


def _format_mph(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.1f} mph"


def _descent_section_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    section_frame = frame.copy()
    if "descent_section_id" not in section_frame.columns:
        section_frame["descent_section_id"] = (
            section_frame["descent_chunk_state"].ne(section_frame["descent_chunk_state"].shift()).cumsum()
        )

    rows: list[dict[str, object]] = []
    descent_number = 0
    for _, group in section_frame.groupby("descent_section_id", sort=False):
        first = group.iloc[0]
        state = first["descent_chunk_state"]
        is_descent = state == "descent"
        if is_descent:
            descent_number += 1

        distance_mi = float(pd.to_numeric(group["step_dist_f"], errors="coerce").fillna(0).sum() / 5280)
        max_speed_mph = float(pd.to_numeric(group["coast_speed_mph"], errors="coerce").max())
        road_name = "Unknown Road"
        if "osm_name" in group.columns:
            names = [str(value).strip() for value in group["osm_name"] if pd.notna(value) and str(value).strip()]
            if names:
                road_name = names[len(names) // 2]

        section = f"{descent_number}. {road_name}" if is_descent else "other"
        rows.append(
            {
                "descent_section_id": first["descent_section_id"],
                "descent_chunk_state": state,
                "Section": section,
                "Distance (mi)": round(distance_mi, 2),
                "Max Coast Speed": _format_mph(max_speed_mph) if is_descent else "",
                "_display_color": DESCENT_CHUNK_STATE_COLORS.get(state, "#8a8a8a"),
                "geometry": _combine_linestrings(group.geometry),
            }
        )

    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=frame.crs)
    return result[~result.geometry.isna()].copy()


def _add_descent_section_display_columns(
    frame: gpd.GeoDataFrame,
    section_frame: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    display_columns = [
        "descent_section_id",
        "Section",
        "Distance (mi)",
        "Max Coast Speed",
    ]
    section_display = section_frame[[column for column in display_columns if column in section_frame.columns]].copy()
    if "descent_section_id" not in frame.columns:
        frame = frame.copy()
        frame["descent_section_id"] = frame["descent_chunk_state"].ne(frame["descent_chunk_state"].shift()).cumsum()

    merged = frame.drop(
        columns=[column for column in section_display.columns if column != "descent_section_id" and column in frame.columns],
        errors="ignore",
    ).merge(section_display, on="descent_section_id", how="left")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=frame.crs)


def make_descent_chunk_map(
    gdf_segments: gpd.GeoDataFrame,
    popup_cols: list[str] | None = None,
    tooltip_fields: list[str] | None = None,
    tiles: str = "CartoDB Voyager",
    show_gravel_overlay: bool = False,
) -> folium.Map:
    """Build a Folium map showing descent chunks from coasting speed."""
    if "descent_chunk_state" in gdf_segments.columns:
        frame = gdf_segments.copy()
    else:
        frame = detect_descent_chunks(gdf_segments)

    frame["Segment"] = frame["step"].astype("Int64").astype(str) if "step" in frame.columns else frame.index.astype(str)
    if {"lat", "lon"}.issubset(frame.columns):
        frame["More Details"] = google_maps_link(frame["lat"], frame["lon"])
    if "osm_name" in frame.columns:
        frame["Road Name"] = frame["osm_name"].fillna("Unknown Road")
    frame["_display_color"] = frame["descent_chunk_state"].map(DESCENT_CHUNK_STATE_COLORS).fillna("#8a8a8a")

    section_frame = _descent_section_frame(frame)
    interaction_frame = _add_descent_section_display_columns(frame, section_frame)

    if tooltip_fields is None:
        tooltip_fields = ["Section", "Distance (mi)", "Max Coast Speed"]
    if popup_cols is None:
        popup_cols = ["Section", "Distance (mi)", "Max Coast Speed"]

    m = section_frame.explore(
        column="descent_chunk_state",
        tooltip=tooltip_fields,
        popup=popup_cols,
        tiles=tiles,
        categorical=True,
        cmap=list(DESCENT_CHUNK_STATE_COLORS.values()),
        categories=list(DESCENT_CHUNK_STATE_COLORS.keys()),
        legend=True,
        style_kwds={"weight": 4},
        escape=False,
    )
    add_map_elements(
        m,
        interaction_frame,
        show_route_pass_control=True,
        layer_column="descent_chunk_state",
        popup_cols=popup_cols,
        tooltip_fields=tooltip_fields,
        categories=list(DESCENT_CHUNK_STATE_COLORS.keys()),
        cmap=list(DESCENT_CHUNK_STATE_COLORS.values()),
        style_kwds={"weight": 4},
        touch_target_frame=section_frame,
        show_gravel_overlay=show_gravel_overlay,
    )
    return m
