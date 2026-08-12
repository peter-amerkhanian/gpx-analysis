from __future__ import annotations

import geopandas as gpd
import pandas as pd

from ..physics import compute_elevation_totals

GRAVEL_TITLE_THRESHOLD_PERCENT = 10.0
CYCLEWAY_TITLE_THRESHOLD_PERCENT = 20.0
GRAVEL_HIGHLIGHT_COLOR = "chocolate"
CYCLEWAY_HIGHLIGHT_COLOR = "forestgreen"


def compute_route_summary(points: pd.DataFrame, segments: gpd.GeoDataFrame) -> dict[str, object]:
    total_distance_m = float(points["step_dist_m"].sum())
    elevation_totals = compute_elevation_totals(points)

    max_row = points.loc[points["elevation_m"].fillna(float("-inf")).idxmax()] if points["elevation_m"].notna().any() else None

    return {
        "point_count": int(len(points)),
        "segment_count": int(len(segments)),
        "distance_m": round(total_distance_m, 2),
        "distance_mi": round(total_distance_m / 1609.344, 2),
        "elevation_gain_m": round(elevation_totals["elevation_gain_m"], 0),
        "elevation_gain_ft": round(elevation_totals["elevation_gain_ft"], 0),
        "elevation_loss_m": round(elevation_totals["elevation_loss_m"], 0),
        "elevation_loss_ft": round(elevation_totals["elevation_loss_ft"], 0),
        "max_elevation_m": None if max_row is None else round(float(max_row["elevation_m"]), 0),
        "max_elevation_ft": None if max_row is None else round(float(max_row["elevation_f"]), 0),
        "start": {
            "lat": round(float(points.iloc[0]["lat"]), 6),
            "lon": round(float(points.iloc[0]["lon"]), 6),
        },
        "end": {
            "lat": round(float(points.iloc[-1]["lat"]), 6),
            "lon": round(float(points.iloc[-1]["lon"]), 6),
        },
    }


def format_duration_hhmm(minutes: float) -> str:
    """Return a duration like 1:30, preserving a leading zero hour for short rides."""
    total_minutes = max(0, int(round(minutes)))
    hours, remaining_minutes = divmod(total_minutes, 60)
    return f"{hours}:{remaining_minutes:02d}"


def total_estimated_time_minutes(chunk_sections_summary: pd.DataFrame) -> float:
    section_column = (
        "Section (avg grade)"
        if "Section (avg grade)" in chunk_sections_summary.columns
        else "Section"
    )
    total_rows = chunk_sections_summary[chunk_sections_summary[section_column].eq("TOTAL")]
    if total_rows.empty:
        return 0.0
    total_time = str(total_rows.iloc[0]["Time (Min)"]).strip()
    if not total_time:
        return 0.0
    return float(total_time.split()[0])


def route_display_title(
    base_title: str,
    gravel_percent: float,
    cycleway_percent: float,
) -> str:
    """Return the route title with gravel/cycleway suffixes when those route types are prominent."""
    notes: list[str] = []
    if gravel_percent > GRAVEL_TITLE_THRESHOLD_PERCENT:
        notes.append(f"{round(gravel_percent)}% Gravel")
    if cycleway_percent > CYCLEWAY_TITLE_THRESHOLD_PERCENT:
        notes.append(f"{round(cycleway_percent)}% Cycleway")
    if notes:
        return f"{base_title} ({', '.join(notes)})"
    return base_title


def route_display_title_html(
    base_title: str,
    gravel_percent: float,
    cycleway_percent: float,
) -> str:
    """Return HTML title markup with colored gravel/cycleway suffixes when applicable."""
    notes: list[str] = []
    if gravel_percent > GRAVEL_TITLE_THRESHOLD_PERCENT:
        notes.append(
            f'<span style="color: {GRAVEL_HIGHLIGHT_COLOR};">({round(gravel_percent)}% Gravel)</span>'
        )
    if cycleway_percent > CYCLEWAY_TITLE_THRESHOLD_PERCENT:
        notes.append(
            f'<span style="color: {CYCLEWAY_HIGHLIGHT_COLOR};">({round(cycleway_percent)}% Cycleway)</span>'
        )
    if notes:
        return f'{base_title} <span>{", ".join(notes)}</span>'
    return base_title
