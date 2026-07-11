from __future__ import annotations

import pandas as pd
def make_google_maps_link(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Build Google Maps query URLs for latitude/longitude series pairs."""
    gmaps_url = (
        "https://www.google.com/maps?q=" + lat.astype(str) + "," + lon.astype(str)
    )
    gmaps_link = (
        '<a href="' + gmaps_url + '" target="_blank">📍Open in Google Maps📍</a>'
    )
    return gmaps_link



def _middle_non_empty_value(values: pd.Series, fallback: str = "Unknown Road") -> str:
    filtered = [
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    ]
    if not filtered:
        return fallback
    return filtered[len(filtered) // 2]



def _road_name_from_section_label(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text == "flat or descent":
        return None

    if ". " in text:
        prefix, remainder = text.split(". ", 1)
        if prefix.isdigit():
            text = remainder

    if ": " in text:
        road_name, _ = text.split(": ", 1)
        road_name = road_name.strip()
        return road_name or None

    if " (" in text:
        road_name, _ = text.split(" (", 1)
        road_name = road_name.strip()
        return road_name or None

    return text



def _format_average_grade_label(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric) * 100:.0f}% avg"



def _safe_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)



def _format_percent(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric) * 100:.2f}%"

