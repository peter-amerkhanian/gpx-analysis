from __future__ import annotations

from typing import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import folium


DEFAULT_HAZARD_COLORS = {
    "flat": "#31D492",
    "light_descent": "#F9C74F",
    "steep_descent": "#F9844A",
    "ultra_steep_descent": "#F94144",
    "turn_on_descent": "#577590",
    "turn_on_steep_descent": "#277DA1",
    "climb": "#2D9966",
    "steep_climb": "#1B5E20",
}


def google_maps_url(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Build Google Maps query URLs for latitude/longitude series pairs."""
    return "https://www.google.com/maps?q=" + lat.astype(str) + "," + lon.astype(str)


def prepare_segment_display_columns(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
) -> gpd.GeoDataFrame:
    """Return a copy with presentation columns used by folium visualizations."""
    frame = gdf_segments.copy()
    colors = dict(DEFAULT_HAZARD_COLORS)
    if hazard_colors:
        colors.update(hazard_colors)
    frame["More Details"] = (
    '<a href="' +
    google_maps_url(frame['lat'], frame['lon']) +
    '" target="_blank">📍 Open in Google Maps</a>'
    )
    frame["Turn"] = (
        frame["step_turn"].round(2).astype(str) + "°"
    )
    frame["Grade"] = (
        frame["step_grade"].multiply(100).round(2).astype(str) + "%"
    )
    frame["Ride Type"] = (
        frame["hazard"].str.title().str.replace("_", " ", regex=False)
    )
    return frame

def prepare_osm_columns(gdf_segments_enriched: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = gdf_segments_enriched.copy()
    frame["Road Name"] = (
    frame["osm_name"].fillna("Unknown Road")
    )
    frame["Road Type"] = (
    frame["osm_lanes"].fillna('Unknown') +
    " lane, " +
    frame["osm_highway"].fillna('unknown') + " road"
    )
    frame["Speed Limit"] = (
    frame["osm_maxspeed"]
    )
    return frame


def make_route_map(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
    popup_cols: str = [],
    tooltip_fields: list[str] | None = ['Ride Type'],
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Build a Folium map with hazard-colored segments and route popups/tooltips."""
    frame = prepare_segment_display_columns(gdf_segments, hazard_colors=hazard_colors)
    m = frame.explore(
    column='hazard',
    tooltip=tooltip_fields,
    popup=popup_cols,
    tiles=tiles,
    categorical=True,
    cmap=list(hazard_colors.values()),
    categories=list(hazard_colors.keys()),
    legend=True,
    style_kwds={"weight": 6},
    escape=False
    )
    return m
