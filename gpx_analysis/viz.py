from __future__ import annotations

from typing import Literal, Mapping
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
import pandas as pd
import folium


DETAILED_HAZARD_COLORS = {
    "steep_climb": "#012C22",
    "climb": "#2D9966",
    "flat": "#31D492",
    "light_descent": "#fee08b",
    "steep_descent": "#f46d43",
    "ultra_steep_descent": "#9F0712",
    "turn_on_descent": "#4F39F6",
    "turn_on_steep_descent": "#8A0194",
}

SIMPLIFIED_HAZARD_COLORS = {
    "steep_climb": "#012C22",
    "climb": "#29865B",
    "mellow": "#31D492",
    "descent": "#f46d43",
    "danger_zone": "#9F0712",
}

DEFAULT_HAZARD_COLORS = DETAILED_HAZARD_COLORS
DEFAULT_HAZARD_PROFILE = "simplified"

HAZARD_PROFILE_LABELS = {
    "detailed": {
        "steep_climb": "Steep Climb",
        "climb": "Climb",
        "flat": "Flat",
        "light_descent": "Light Descent",
        "steep_descent": "Steep Descent",
        "ultra_steep_descent": "Ultra Steep Descent",
        "turn_on_descent": "Turn On Descent",
        "turn_on_steep_descent": "Turn On Steep Descent",
    },
    "simplified": {
        "steep_climb": "Steep Climb",
        "climb": "Climb",
        "mellow": "Mellow",
        "descent": "Descent",
        "danger_zone": "Danger Zone",
    },
}

HAZARD_PROFILE_REMAPS = {
    "detailed": {
        "steep_climb": "steep_climb",
        "climb": "climb",
        "flat": "flat",
        "light_descent": "light_descent",
        "steep_descent": "steep_descent",
        "ultra_steep_descent": "ultra_steep_descent",
        "turn_on_descent": "turn_on_descent",
        "turn_on_steep_descent": "turn_on_steep_descent",
    },
    "simplified": {
        "steep_climb": "steep_climb",
        "climb": "climb",
        "flat": "mellow",
        "light_descent": "mellow",
        "steep_descent": "descent",
        "turn_on_descent": "descent",
        "ultra_steep_descent": "danger_zone",
        "turn_on_steep_descent": "danger_zone",
    },
}

HAZARD_PROFILE_COLORS = {
    "detailed": DETAILED_HAZARD_COLORS,
    "simplified": SIMPLIFIED_HAZARD_COLORS,
}

HazardProfileName = Literal["detailed", "simplified"]


def resolve_hazard_profile(
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
    hazard_colors: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    remap = dict(HAZARD_PROFILE_REMAPS[hazard_profile])
    colors = dict(HAZARD_PROFILE_COLORS[hazard_profile])
    if hazard_colors:
        colors.update(hazard_colors)
    labels = dict(HAZARD_PROFILE_LABELS[hazard_profile])
    return remap, colors, labels


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


def google_maps_url(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Build Google Maps query URLs for latitude/longitude series pairs."""
    return "https://www.google.com/maps?q=" + lat.astype(str) + "," + lon.astype(str)


def prepare_segment_display_columns(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> gpd.GeoDataFrame:
    """Return a copy with presentation columns used by folium visualizations."""
    frame = apply_hazard_profile(gdf_segments, hazard_profile=hazard_profile)
    frame["Segment"] = frame["step"].astype("Int64").astype(str)
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
    frame["Ride Type"] = frame["hazard_label"]
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


def make_route_map(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
    popup_cols: list[str] | None = None,
    tooltip_fields: list[str] | None = ['Segment', 'Ride Type'],
    tiles: str = "CartoDB Voyager",
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> folium.Map:
    """Build a Folium map with hazard-colored segments and route popups/tooltips."""
    frame = prepare_segment_display_columns(
        gdf_segments,
        hazard_colors=hazard_colors,
        hazard_profile=hazard_profile,
    )
    _, colors, _ = resolve_hazard_profile(
        hazard_profile=hazard_profile,
        hazard_colors=hazard_colors,
    )
    m = frame.explore(
    column='hazard',
    tooltip=tooltip_fields,
    popup=popup_cols or [],
    tiles=tiles,
    categorical=True,
    cmap=list(colors.values()),
    categories=list(colors.keys()),
    legend=True,
    style_kwds={"weight": 6},
    escape=False
    )
    third = int(len(frame) / 8)
    start = frame.iloc[0].geometry.coords[0]

    folium.Marker(
        location=[start[1], start[0]],  # folium uses [lat, lon]
        icon=folium.Icon(color="green", icon="arrow-right", prefix="fa"),
    ).add_to(m)
    number_style = "font-size:51px; font-weight:700; color:#C96A1B; opacity:0.65;"
    folium.Marker(
        location=[frame.iloc[third].geometry.coords[0][1], frame.iloc[third].geometry.coords[0][0]],  # folium uses [lat, lon]
        icon=folium.DivIcon(
            html=(
                f'<div style="{number_style}">'
                '1'
                '</div>'
            )
        ),
    ).add_to(m)
    folium.Marker(
        location=[frame.iloc[third*4].geometry.coords[0][1], frame.iloc[third*4].geometry.coords[0][0]],  # folium uses [lat, lon]
        icon=folium.DivIcon(
            html=(
                f'<div style="{number_style}">'
                '2'
                '</div>'
            )
        ),
    ).add_to(m)
    folium.Marker(
        location=[frame.iloc[third*7].geometry.coords[0][1], frame.iloc[third*7].geometry.coords[0][0]],  # folium uses [lat, lon]
        icon=folium.DivIcon(
            html=(
                f'<div style="{number_style}">'
                '3'
                '</div>'
            )
        ),
    ).add_to(m)
    return m
