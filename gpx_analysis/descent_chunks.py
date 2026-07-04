from __future__ import annotations

import geopandas as gpd
import pandas as pd
import folium

from .viz import add_map_elements, google_maps_link

DESCENT_CHUNK_STATE_COLORS = {
    "other": "#bdbdbd",
    "light descent": "#ffa959",
    "descent": "#e80000",
    "steep descent": "#950500",
    "dangerous descent": "#4f0080",
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
    frame["descent_chunk_avg_speed_mph"] = pd.NA
    frame["descent_chunk_avg_grade"] = pd.NA

    next_chunk_id = 1
    active_indices: list[object] = []

    def finalize(indices: list[object]) -> None:
        nonlocal next_chunk_id
        if not indices:
            return

        chunk = frame.loc[indices]
        chunk_distance_ft = float(pd.to_numeric(chunk["step_dist_f"], errors="coerce").fillna(0).sum())
        speeds = pd.to_numeric(chunk["coast_speed_mph"], errors="coerce")
        max_speed_mph = float(speeds.max())
        frame.loc[indices, "descent_candidate_chunk_dist_ft"] = chunk_distance_ft
        if chunk_distance_ft < min_chunk_dist_ft or max_speed_mph < peak_speed_mph:
            return

        distances = pd.to_numeric(chunk["step_dist_f"], errors="coerce").fillna(0)
        speed_weights = distances.where(speeds.notna(), 0)
        avg_speed_mph = (
            float((speeds.fillna(0) * speed_weights).sum() / speed_weights.sum())
            if float(speed_weights.sum()) > 0
            else max_speed_mph
        )
        grades = pd.to_numeric(chunk.get("step_grade"), errors="coerce") if "step_grade" in chunk.columns else pd.Series(pd.NA, index=chunk.index)
        grade_weights = distances.where(grades.notna(), 0)
        avg_grade = (
            float((grades.fillna(0) * grade_weights).sum() / grade_weights.sum())
            if float(grade_weights.sum()) > 0
            else pd.NA
        )
        state = _descent_state_from_max_speed(max_speed_mph)

        frame.loc[indices, "descent_chunk_state"] = state
        frame.loc[indices, "descent_chunk_label"] = state
        frame.loc[indices, "descent_chunk_id"] = next_chunk_id
        frame.loc[indices, "descent_chunk_dist_ft"] = chunk_distance_ft
        frame.loc[indices, "descent_chunk_max_speed_mph"] = max_speed_mph
        frame.loc[indices, "descent_chunk_avg_speed_mph"] = avg_speed_mph
        frame.loc[indices, "descent_chunk_avg_grade"] = avg_grade
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


def _descent_state_from_max_speed(max_speed_mph: float) -> str:
    if max_speed_mph >= 51.0:
        return "dangerous descent"
    if max_speed_mph >= 41.0:
        return "steep descent"
    if max_speed_mph >= 31.0:
        return "descent"
    return "light descent"


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


def _format_percent(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric) * 100:.1f}%"


def _weighted_average(values: pd.Series, weights: pd.Series) -> float | object:
    numeric = pd.to_numeric(values, errors="coerce")
    clean_weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    valid_weights = clean_weights.where(numeric.notna(), 0)
    total_weight = float(valid_weights.sum())
    if total_weight <= 0:
        return pd.NA
    return float((numeric.fillna(0) * valid_weights).sum() / total_weight)


def _descent_road_quality_label(group: pd.DataFrame, distance_column: str = "step_dist_f") -> str:
    distance = pd.to_numeric(group[distance_column], errors="coerce").fillna(0)
    total_distance = float(distance.sum())
    if total_distance <= 0:
        return ""

    gravel = pd.Series(False, index=group.index)
    if "road_type" in group.columns:
        gravel = gravel | group["road_type"].fillna("").astype(str).str.lower().eq("gravel")
    if "mtc_pci_info" in group.columns:
        gravel = gravel | group["mtc_pci_info"].fillna("").astype(str).eq("Gravel")
    if bool(gravel.all()):
        return "Gravel"

    if "mtc_pci_info" not in group.columns:
        return ""
    good_quality = group["mtc_pci_info"].isin(["Excellent", "Very Good", "Good", "Fair"])
    good_distance = float(distance.where(good_quality, 0).sum())
    return f"{good_distance / total_distance * 100:.0f}%"


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
        is_descent = state != "other"
        if is_descent:
            descent_number += 1

        distance_mi = float(pd.to_numeric(group["step_dist_f"], errors="coerce").fillna(0).sum() / 5280)
        max_speed_mph = float(pd.to_numeric(group["coast_speed_mph"], errors="coerce").max())
        avg_speed_mph = first.get("descent_chunk_avg_speed_mph")
        avg_grade = first.get("descent_chunk_avg_grade")
        road_name = "Unknown Road"
        if "osm_name" in group.columns:
            names = [str(value).strip() for value in group["osm_name"] if pd.notna(value) and str(value).strip()]
            if names:
                road_name = names[len(names) // 2]

        section = f"{descent_number}. {road_name}: {state}" if is_descent else "other"
        rows.append(
            {
                "descent_section_id": first["descent_section_id"],
                "descent_chunk_state": state,
                "Section": section,
                "Distance (mi)": round(distance_mi, 2),
                "Max Coast Speed": _format_mph(max_speed_mph) if is_descent else "",
                "Average Coast Speed": _format_mph(avg_speed_mph) if is_descent else "",
                "Average Grade": _format_percent(avg_grade) if is_descent else "",
                "_display_color": DESCENT_CHUNK_STATE_COLORS.get(state, "#8a8a8a"),
                "geometry": _combine_linestrings(group.geometry),
            }
        )

    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=frame.crs)
    return result[~result.geometry.isna()].copy()


def summarize_descent_chunk_sections(
    df: pd.DataFrame,
    distance_column: str = "step_dist_f",
) -> pd.DataFrame:
    """Summarize descent chunks with speed, grade, and road-quality details."""
    if "descent_chunk_state" in df.columns:
        frame = df.copy()
    else:
        frame = detect_descent_chunks(df)

    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable")
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable")
    elif "end_i" in frame.columns:
        frame = frame.sort_values("end_i", kind="stable")
    else:
        frame = frame.sort_index(kind="stable")

    if "descent_section_id" not in frame.columns:
        frame["descent_section_id"] = frame["descent_chunk_state"].ne(frame["descent_chunk_state"].shift()).cumsum()

    rows: list[dict[str, object]] = []
    descent_number = 0
    for _, group in frame.groupby("descent_section_id", sort=False):
        first = group.iloc[0]
        state = first["descent_chunk_state"]
        if state == "other":
            continue

        descent_number += 1
        distance = pd.to_numeric(group[distance_column], errors="coerce").fillna(0)
        distance_mi = float(distance.sum() / 5280)
        if distance_mi <= 0:
            continue

        road_name = "Unknown Road"
        if "osm_name" in group.columns:
            names = [str(value).strip() for value in group["osm_name"] if pd.notna(value) and str(value).strip()]
            if names:
                road_name = names[len(names) // 2]
        avg_grade = _weighted_average(group.get("step_grade", pd.Series(pd.NA, index=group.index)), distance)
        max_speed_mph = float(pd.to_numeric(group["coast_speed_mph"], errors="coerce").max())

        rows.append(
            {
                "Section": f"{descent_number}. {road_name}: {state}",
                "Average Grade": _format_percent(avg_grade),
                "Max Coast Speed": _format_mph(max_speed_mph),
                "Good+ Pavement": _descent_road_quality_label(group, distance_column=distance_column),
                "Distance (mi)": round(distance_mi, 1),
                "_avg_grade_raw": avg_grade,
                "_max_speed_raw": max_speed_mph,
            }
        )

    columns = ["Section", "Average Grade", "Max Coast Speed", "Good+ Pavement", "Distance (mi)"]
    if not rows:
        return pd.DataFrame(columns=columns)

    sections = pd.DataFrame(rows)
    total_distance_ft = float(pd.to_numeric(frame.loc[frame["descent_chunk_state"] != "other", distance_column], errors="coerce").fillna(0).sum())
    descent_frame = frame[frame["descent_chunk_state"] != "other"].copy()
    total_row = pd.DataFrame(
        [
            {
                "Section": "TOTAL",
                "Average Grade": _format_percent(
                    _weighted_average(
                        descent_frame.get("step_grade", pd.Series(pd.NA, index=descent_frame.index)),
                        pd.to_numeric(descent_frame[distance_column], errors="coerce").fillna(0),
                    )
                ),
                "Max Coast Speed": _format_mph(pd.to_numeric(descent_frame["coast_speed_mph"], errors="coerce").max()),
                "Good+ Pavement": _descent_road_quality_label(descent_frame, distance_column=distance_column),
                "Distance (mi)": round(total_distance_ft / 5280, 1),
            }
        ]
    )
    result = pd.concat([total_row, sections[columns]], ignore_index=True)
    return result[columns]


def _add_descent_section_display_columns(
    frame: gpd.GeoDataFrame,
    section_frame: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    display_columns = [
        "descent_section_id",
        "Section",
        "Distance (mi)",
        "Max Coast Speed",
        "Average Coast Speed",
        "Average Grade",
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
    show_gravel_overlay: bool = True,
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
        tooltip_fields = ["Section", "Distance (mi)", "Max Coast Speed", "Average Coast Speed", "Average Grade"]
    if popup_cols is None:
        popup_cols = ["Section", "Distance (mi)", "Max Coast Speed", "Average Coast Speed", "Average Grade"]

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
