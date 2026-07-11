from __future__ import annotations

from typing import Mapping

import geopandas as gpd
import pandas as pd

from .formatters import make_google_maps_link
from .palettes import DEFAULT_HAZARD_PROFILE, HazardProfileName, resolve_hazard_profile
def apply_hazard_profile(
    frame: pd.DataFrame,
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> pd.DataFrame:
    result = frame.copy()
    remap, _, labels = resolve_hazard_profile(hazard_profile=hazard_profile)
    result["hazard_raw"] = result["hazard"]
    result["hazard"] = result["hazard"].map(remap).fillna(result["hazard"])
    result["hazard_label"] = result["hazard"].map(labels).fillna(
        result["hazard"].str.replace("_", " ", regex=False).str.title()
    )
    return result



def _add_google_maps_details(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a Google Maps link column using lat/lon columns or segment geometry."""
    if {"lat", "lon"}.issubset(frame.columns):
        frame["More Details"] = make_google_maps_link(frame["lat"], frame["lon"])
        return frame

    if "geometry" not in frame.columns:
        return frame

    coords = frame.geometry.apply(
        lambda geometry: (
            geometry.coords[0][1],
            geometry.coords[0][0],
        )
        if geometry is not None and not geometry.is_empty and hasattr(geometry, "coords")
        else (pd.NA, pd.NA)
    )
    lat = coords.apply(lambda value: value[0])
    lon = coords.apply(lambda value: value[1])
    valid = lat.notna() & lon.notna()
    frame["More Details"] = ""
    frame.loc[valid, "More Details"] = make_google_maps_link(lat.loc[valid], lon.loc[valid])
    return frame



def prepare_segment_display_columns(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> gpd.GeoDataFrame:
    """Return a copy with presentation columns used by folium visualizations."""
    frame = apply_hazard_profile(gdf_segments, hazard_profile=hazard_profile)
    _, colors, _ = resolve_hazard_profile(
        hazard_profile=hazard_profile,
        hazard_colors=hazard_colors,
    )
    frame["Segment"] = frame["step"].astype("Int64").astype(str)
    frame = _add_google_maps_details(frame)
    frame["Turn"] = (
        frame["step_turn"].round(2).astype(str) + "Â°"
    )
    frame["Grade"] = (
        frame["step_grade"].multiply(100).round(2).astype(str) + "%"
    )
    hazard_grade_source = frame["hazard_grade"] if "hazard_grade" in frame.columns else frame["step_grade"]
    frame["Hazard Grade"] = (
        pd.to_numeric(hazard_grade_source, errors="coerce").multiply(100).round(2).astype(str) + "%"
    )
    frame["Ride Type"] = frame["hazard_label"]
    if "osm_name" in frame.columns:
        frame["Road Name"] = frame["osm_name"].fillna("Unknown Road")
    if "elevation_f" in frame.columns:
        frame["Elevation (ft)"] = pd.to_numeric(
            frame["elevation_f"],
            errors="coerce",
        ).round(0).astype("Int64").astype(str) + " ft"
    frame["_display_color"] = frame["hazard"].map(colors).fillna("#8a8a8a")
    return frame


def prepare_osm_columns(gdf_segments_enriched: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = gdf_segments_enriched.copy()
    frame["Road Name"] = (
    frame["osm_name"].fillna("Unknown Road")
    )
    frame["Road Type"] = (
    frame["osm_highway"].str.title().fillna('Unknown type') + " " +
    frame["osm_lanes"].fillna('unknown') +
    " lane road"
    )
    frame["Speed Limit"] = (
    frame["osm_maxspeed"].fillna("Unknown")
    )
    return frame



def _select_present_columns(
    frame: gpd.GeoDataFrame,
    columns: list[str],
) -> gpd.GeoDataFrame:
    """Return a frame with just the requested columns that are present."""
    keep = [column for column in columns if column in frame.columns]
    return frame.loc[:, keep].copy()

