from __future__ import annotations

from typing import Mapping

import folium
import geopandas as gpd
import pandas as pd

from .columns import _add_google_maps_details, _select_present_columns, prepare_segment_display_columns
from .folium_base import _present_interaction_fields, add_map_elements
from .palettes import (
    DEFAULT_HAZARD_PROFILE,
    HazardProfileName,
    resolve_hazard_profile,
    resolve_simplified_road_quality_profile,
    simplify_road_quality_category,
)
def make_hazard_map(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
    popup_cols: list[str] | None = ["Road Name", "Ride Type", "Turn", "Grade", "Hazard Grade", "More Details"],
    tooltip_fields: list[str] | None = ['Segment', 'Road Name', 'Ride Type', "More Details"],
    tiles: str = "CartoDB Voyager",
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
    show_gravel_overlay: bool = False,
) -> folium.Map:
    """Build a Folium map with hazard-colored segments and route popups/tooltips."""
    frame = prepare_segment_display_columns(
        gdf_segments,
        hazard_colors=hazard_colors,
        hazard_profile=hazard_profile,
    )
    frame = _select_present_columns(
        frame,
        [
            "geometry",
            "step_dist_m",
            "Segment",
            "Road Name",
            "Ride Type",
            "Turn",
            "Grade",
            "Hazard Grade",
            "More Details",
            "hazard",
            "road_type",
            "mtc_pci_info",
            "_display_color",
        ],
    )
    _, colors, _ = resolve_hazard_profile(
        hazard_profile=hazard_profile,
        hazard_colors=hazard_colors,
    )
    m = frame.explore(
    column='hazard',
    name="Route",
    tooltip=tooltip_fields,
    popup=popup_cols,
    tiles=tiles,
    categorical=True,
    cmap=list(colors.values()),
    categories=list(colors.keys()),
    legend=True,
    style_kwds={"weight": 4},
    escape=False
    )
    m = add_map_elements(
        m,
        frame,
        show_route_pass_control=True,
        layer_column="hazard",
        tooltip_fields=tooltip_fields,
        popup_cols=popup_cols,
        categories=list(colors.keys()),
        cmap=list(colors.values()),
        style_kwds={"weight": 4},
        escape=False,
        show_gravel_overlay=show_gravel_overlay,
    )
    return m



def make_route_overview_map(
    gdf_segments: gpd.GeoDataFrame,
    tiles: str = "openstreetmap",
    show_gravel_overlay: bool = True,
) -> folium.Map:
    """Build a simple route overview map with direction arrows."""
    frame = gdf_segments.copy()
    if "osm_name" in frame.columns:
        frame["Road Name"] = frame["osm_name"].fillna("Unknown Road")
    if "elevation_f" in frame.columns:
        frame["Elevation (ft)"] = pd.to_numeric(
            frame["elevation_f"],
            errors="coerce",
        ).round(0).astype("Int64").astype(str) + " ft"
    frame = _add_google_maps_details(frame)
    frame = _select_present_columns(
        frame,
        [
            "geometry",
            "step_dist_m",
            "Road Name",
            "Elevation (ft)",
            "step",
            "More Details",
            "road_type",
            "mtc_pci_info",
        ],
    )
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=gdf_segments.crs)
    interaction_fields = ["Road Name", "Elevation (ft)", "step", "More Details"]
    m = frame.explore(
        name="Route",
        tooltip=_present_interaction_fields(frame, interaction_fields),
        popup=_present_interaction_fields(frame, interaction_fields),
        tiles=tiles,
        color="#1da0eb",
        style_kwds={
            "weight": 4,
            "opacity": 0.92,
            "line_cap": "round",
            "line_join": "round",
        },
        escape=False,
    )

    add_map_elements(
        m,
        frame,
        show_numbers=False,
        popup_cols=interaction_fields,
        tooltip_fields=interaction_fields,
        show_gravel_overlay=show_gravel_overlay,
    )
    return m



def _smooth_grade_by_distance(
    frame: gpd.GeoDataFrame,
    grade_column: str,
    distance_column: str,
    smoothing_window_m: float,
) -> pd.Series:
    """Return a centered, distance-weighted grade average."""
    grade = pd.to_numeric(frame[grade_column], errors="coerce")
    distance = pd.to_numeric(frame[distance_column], errors="coerce").fillna(0)
    if smoothing_window_m <= 0 or frame.empty:
        return grade

    weighted_grade = grade.fillna(0).mul(distance.where(grade.notna(), 0))
    valid_distance = distance.where(grade.notna(), 0)
    positions = distance.cumsum()
    half_window_m = smoothing_window_m / 2.0
    smoothed: list[float] = []

    for center in positions:
        in_window = positions.sub(center).abs().le(half_window_m)
        window_distance = float(valid_distance.loc[in_window].sum())
        if window_distance > 0:
            smoothed.append(float(weighted_grade.loc[in_window].sum() / window_distance))
        else:
            smoothed.append(float("nan"))
    return pd.Series(smoothed, index=frame.index, dtype="float64")



def make_grade_map(
    gdf_segments: gpd.GeoDataFrame,
    grade_column: str = "avg_step_grade",
    smoothing_window_m: float = 180.0,
    popup_cols: list[str] | None = None,
    tooltip_fields: list[str] | None = None,
    tiles: str = "CartoDB Positron",
    cmap: str = "RdYlGn",
    vmin: float = -0.1,
    vmax: float = 0.1,
    show_gravel_overlay: bool = True,
) -> folium.Map:
    """Build a continuous-color map of smoothed route grade."""
    frame = gdf_segments.copy()
    if grade_column not in frame.columns:
        if grade_column == "avg_step_grade" and "step_grade" in frame.columns:
            grade_column = "step_grade"
        else:
            raise ValueError(f"make_grade_map requires a {grade_column!r} column.")
    if "step_dist_m" not in frame.columns:
        raise ValueError("make_grade_map requires a 'step_dist_m' column.")

    frame["smooth_grade"] = _smooth_grade_by_distance(
        frame,
        grade_column=grade_column,
        distance_column="step_dist_m",
        smoothing_window_m=smoothing_window_m,
    )
    frame["Grade"] = pd.to_numeric(frame.get("step_grade"), errors="coerce").multiply(100).round(2).astype(str) + "%"
    frame["Smoothed Grade"] = frame["smooth_grade"].multiply(100).round(2).astype(str) + "%"
    frame["Segment"] = frame["step"].astype("Int64").astype(str) if "step" in frame.columns else frame.index.astype(str)
    if "osm_name" in frame.columns:
        frame["Road Name"] = frame["osm_name"].fillna("Unknown Road")
    frame = _add_google_maps_details(frame)

    if tooltip_fields is None:
        tooltip_fields = ["Segment", "Road Name", "Smoothed Grade", "More Details"]
    if popup_cols is None:
        popup_cols = ["Road Name", "Smoothed Grade", "Grade", "More Details"]

    frame = _select_present_columns(
        frame,
        [
            "geometry",
            "step_dist_m",
            "Segment",
            "Road Name",
            "Grade",
            "Smoothed Grade",
            "More Details",
            "smooth_grade",
            "road_type",
            "mtc_pci_info",
        ],
    )
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=gdf_segments.crs)
    m = frame.explore(
        column="smooth_grade",
        name="Route",
        tooltip=_present_interaction_fields(frame, tooltip_fields),
        popup=_present_interaction_fields(frame, popup_cols),
        tiles=tiles,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        style_kwds={"weight": 4},
        escape=False,
    )
    add_map_elements(
        m,
        frame,
        show_route_pass_control=True,
        layer_column="smooth_grade",
        popup_cols=popup_cols,
        tooltip_fields=tooltip_fields,
        cmap=cmap,
        style_kwds={"weight": 4},
        categorical=False,
        vmin=vmin,
        vmax=vmax,
        show_gravel_overlay=show_gravel_overlay,
    )
    return m



def make_road_quality_map(
    gdf_segments: gpd.GeoDataFrame,
    popup_cols: list[str] | None = ["mtc_road_name", 'Road Quality', 'mtc_pci_info', 'mtc_pci_date',"Ride Type", "Turn", "Grade", "More Details"],
    tooltip_fields: list[str] | None = ['Segment', 'Road Quality', "More Details"],
    tiles: str = "Cartodb Positron",
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
    show_gravel_overlay: bool = False,
) -> folium.Map:
    """Build a Folium map with hazard-colored segments and route popups/tooltips."""
    frame = prepare_segment_display_columns(
        gdf_segments,
        hazard_profile=hazard_profile,
    )
    if "mtc_pci_info" not in frame.columns:
        frame["mtc_pci_info"] = "Roadway (Unknown)"
    else:
        frame["mtc_pci_info"] = frame["mtc_pci_info"].fillna("Roadway (Unknown)")
    frame["road_quality_simple"] = frame["mtc_pci_info"].apply(simplify_road_quality_category)
    frame["Road Quality"] = frame["road_quality_simple"]
    colors = resolve_simplified_road_quality_profile(frame)
    frame["_display_color"] = frame["road_quality_simple"].map(colors).fillna("#8a8a8a")
    frame = _select_present_columns(
        frame,
        [
            "geometry",
            "step_dist_m",
            "Segment",
            "Road Quality",
            "mtc_road_name",
            "mtc_pci_info",
            "mtc_pci_date",
            "road_type",
            "Ride Type",
            "Turn",
            "Grade",
            "More Details",
            "road_quality_simple",
            "_display_color",
        ],
    )
    m = frame.explore(
    column='road_quality_simple',
    name="Route",
    tooltip=tooltip_fields,
    popup=popup_cols,
    tiles=tiles,
    categorical=True,
    cmap=list(colors.values()),
    categories=list(colors.keys()),
    legend=True,
    style_kwds={"weight": 4},
    escape=False
    )
    m = add_map_elements(
        m,
        frame,
        popup_cols=popup_cols,
        tooltip_fields=tooltip_fields,
        show_gravel_overlay=show_gravel_overlay,
    )
    return m
