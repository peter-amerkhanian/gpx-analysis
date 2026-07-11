from __future__ import annotations

from typing import Literal, Mapping

import geopandas as gpd
import pandas as pd
ROAD_QUALITY_COLORS = {
    'Excellent': '#1a9850',
    'Very Good': '#91cf60',
    'Good': '#d9ef8b',
    'Fair': '#ffffbf',
    'At Risk': '#fee08b',
    'Poor': '#fc8d59',
    'Failed': '#d73027',
    'Gravel': "#712f00",
    'Cycleway': "#0078da",
}

SIMPLIFIED_ROAD_QUALITY_COLORS = {
    "Great": "#1a9850",
    "Good": "#d9ef8b",
    "Ok": "#fee08b",
    "Poor": "#d73027",
    "Roadway (Unknown)": "#8a8a8a",
    "Gravel": "#712f00",
    "Cycleway": "#0078da",
    "Cycleway (Unknown)": "#0078da",
}

DETAILED_HAZARD_COLORS = {
    "steep_climb": "#012C22",
    "climb": "#2D9966",
    "flat": "#79DFB7",
    "light_descent": "#f99860",
    "steep_descent": "#fc5b2a",
    "ultra_steep_descent": "#9F0712",
    "turn_on_descent": "#fc5b2a",
    "turn_on_steep_descent": "#9F0712",
}

SIMPLIFIED_HAZARD_COLORS = {
    "steep_climb": "#012C22",
    "climb": "#2D9966",
    "flat": "#79DFB7",
    "light_descent": "#f99860",
    "descent": "#fc5b2a",
    "steep_descent": "#9F0712",
}

CHUNK_STATE_COLORS = {
    "flat or descent": "#bdbdbd",
    "climb (easy)": "#9bd770",
    "climb (medium)": "#0C9000",
    "climb (hard)": "#052C01",
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
        "flat": "Flat",
        "light_descent": "Light Descent",
        "descent": "Descent",
        "steep_descent": "Steep Descent",
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
        "flat": "flat",
        "light_descent": "light_descent",
        "steep_descent": "descent",
        "turn_on_descent": "descent",
        "ultra_steep_descent": "steep_descent",
        "turn_on_steep_descent": "steep_descent",
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

def resolve_road_quality_profile(gdf_segments: gpd.GeoDataFrame):
    colors = ROAD_QUALITY_COLORS.copy()
    for val in gdf_segments['mtc_pci_info'].unique():
        if val not in ROAD_QUALITY_COLORS:
            colors[val] = "#8a8a8a"
    for pci, _ in ROAD_QUALITY_COLORS.items():
        if pci not in gdf_segments['mtc_pci_info'].unique():
            del colors[pci]
    return colors


def simplify_road_quality_category(value: object) -> str | object:
    """Collapse detailed PCI labels into simpler map categories."""
    if value is None or pd.isna(value):
        return value

    text = str(value)
    if text in {"Gravel", "Cycleway", "Cycleway (Unknown)"}:
        return text
    if text in {"Excellent", "Very Good"}:
        return "Great"
    if text in {"Good", "Fair"}:
        return "Good"
    if text == "At Risk":
        return "Ok"
    if text in {"Poor", "Failed"}:
        return "Poor"
    if text.endswith(" (Unknown)"):
        return "Roadway (Unknown)"
    return text


def resolve_simplified_road_quality_profile(gdf_segments: gpd.GeoDataFrame) -> dict[str, str]:
    """Return colors for simplified road-quality map categories present in the frame."""
    colors = SIMPLIFIED_ROAD_QUALITY_COLORS.copy()
    present = set(gdf_segments["road_quality_simple"].dropna().astype(str).unique())
    return {label: color for label, color in colors.items() if label in present}

