from typing import Literal

import pandas as pd

from .chunks import detect_chunks
from .hazards import detect_hazards
from .physics import compute_adjusted_elevation_deltas
from .viz import DEFAULT_HAZARD_PROFILE, apply_hazard_profile


DEFAULT_SEGMENT_SPEED_RANGES_MPH = {
    "flat or descent": (9.0, 14.0),
    "climb (easy)": (4.0, 7.0),
    "climb (medium)": (3.0, 6.0),
    "climb (hard)": (2.0, 5.0),
}

DEFAULT_HAZARD_SPEED_RANGES_MPH = {
    "descent": (12.0, 17.0),
    "mellow": (9.0, 14.0),
    "climb": (4.0, 7.0),
    "steep_climb": (2.0, 5.0),
}


def _adjusted_climb_gain_ft(frame: pd.DataFrame) -> pd.Series:
    """Return per-row climb gain in feet using the shared elevation-total method."""
    if "elevation_m" in frame.columns or "step_elevation_m" in frame.columns:
        return compute_adjusted_elevation_deltas(frame).clip(lower=0) * 3.28084
    if "elevation_f" in frame.columns or "step_elevation_f" in frame.columns:
        return compute_adjusted_elevation_deltas(
            frame,
            elevation_column="elevation_f",
            elevation_delta_column="step_elevation_f",
            distance_column="step_dist_f",
            smoothing_window_m=230.0 * 3.28084,
            reversal_threshold_m=4.0 * 3.28084,
        ).clip(lower=0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _middle_non_empty_value(values: pd.Series, fallback: str = "Unknown Road") -> str:
    """Return the middle non-empty string value from an ordered series."""
    filtered = [
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    ]
    if not filtered:
        return fallback
    return filtered[len(filtered) // 2]


def _chunk_section_label(route_part: str, road_name: str, climb_number: int | None = None) -> str:
    """Return a single display label for a chunk section."""
    if route_part == "flat or descent":
        return route_part
    label = f"{road_name}: {route_part}"
    if climb_number is None:
        return label
    return f"{climb_number}. {label}"


def _format_average_grade(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{numeric * 100:.0f}% avg"


def _chunk_table_section_label(
    route_part: str,
    road_name: str,
    average_grade: object,
    climb_number: int | None = None,
) -> str:
    """Return the chunk table section label with average grade instead of class."""
    if route_part == "flat or descent":
        return route_part

    grade_label = _format_average_grade(average_grade)
    label = f"{road_name} ({grade_label})" if grade_label else road_name
    if climb_number is None:
        return label
    return f"{climb_number}. {label}"


def _resolve_timing_hazard_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return segment hazards aligned to the input frame for timing estimates."""
    if "hazard" in df.columns:
        return apply_hazard_profile(df, hazard_profile=DEFAULT_HAZARD_PROFILE)
    return apply_hazard_profile(
        detect_hazards(df),
        hazard_profile=DEFAULT_HAZARD_PROFILE,
    )


def _resolve_chunk_section_frame(
    df: pd.DataFrame,
    distance_column: str = "step_dist_f",
    timing_basis: Literal["hazard", "chunk"] = "hazard",
    speed_ranges_mph: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Return ordered chunk sections with shared distance and timing metadata."""
    timing_frame = _resolve_timing_hazard_frame(df) if timing_basis == "hazard" else None
    if "chunk_state" in df.columns:
        frame = df.copy()
    else:
        frame = detect_chunks(df)

    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable")
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable")
    elif "end_i" in frame.columns:
        frame = frame.sort_values("end_i", kind="stable")
    else:
        frame = frame.sort_index(kind="stable")

    frame["section_id"] = frame["chunk_state"].ne(frame["chunk_state"].shift()).cumsum()
    frame["climb_gain_ft"] = _adjusted_climb_gain_ft(frame)
    if "chunk_avg_grade" not in frame.columns:
        frame["chunk_avg_grade"] = pd.NA
    summary = (
        frame.groupby(["section_id", "chunk_state"], as_index=False)
        .agg(
            distance_ft=(distance_column, "sum"),
            road_name=("osm_name", _middle_non_empty_value),
            climb_gain_ft=("climb_gain_ft", "sum"),
            chunk_avg_grade=("chunk_avg_grade", "first"),
        )
        .rename(columns={"chunk_state": "route_part"})
    )
    summary["distance_mi"] = summary["distance_ft"] / 5280.0
    is_climb = summary["route_part"] != "flat or descent"
    summary["climb_number"] = pd.NA
    summary.loc[is_climb, "climb_number"] = range(1, int(is_climb.sum()) + 1)
    summary["section"] = summary.apply(
        lambda row: _chunk_section_label(
            row["route_part"],
            row["road_name"],
            int(row["climb_number"]) if pd.notna(row["climb_number"]) else None,
        ),
        axis=1,
    )

    if timing_basis == "hazard":
        speed_lookup = DEFAULT_HAZARD_SPEED_RANGES_MPH.copy()
        default_speed_range = DEFAULT_HAZARD_SPEED_RANGES_MPH["mellow"]
        if speed_ranges_mph:
            speed_lookup.update(speed_ranges_mph)

        if timing_frame is None:
            raise ValueError("Hazard timing frame was not resolved")

        frame["timing_hazard"] = timing_frame.loc[frame.index, "hazard"]
        frame["timing_low_mph"] = frame["timing_hazard"].map(
            lambda hazard: speed_lookup.get(hazard, default_speed_range)[0]
        )
        frame["timing_fast_mph"] = frame["timing_hazard"].map(
            lambda hazard: speed_lookup.get(hazard, default_speed_range)[1]
        )
        frame["timing_avg_mph"] = (frame["timing_low_mph"] + frame["timing_fast_mph"]) / 2.0
        distance_mi = pd.to_numeric(frame[distance_column], errors="coerce").fillna(0) / 5280.0
        frame["segment_time_low_min"] = distance_mi / frame["timing_low_mph"] * 60.0
        frame["segment_time_fast_min"] = distance_mi / frame["timing_fast_mph"] * 60.0
        frame["segment_time_avg_min"] = distance_mi / frame["timing_avg_mph"] * 60.0

        time_summary = (
            frame.groupby("section_id", as_index=False)
            .agg(
                time_low_min=("segment_time_low_min", "sum"),
                time_fast_min=("segment_time_fast_min", "sum"),
                time_avg_min=("segment_time_avg_min", "sum"),
            )
        )
        summary = summary.merge(time_summary, on="section_id", how="left")
        summary["low_mph"] = summary["distance_mi"] / (summary["time_low_min"] / 60.0)
        summary["fast_mph"] = summary["distance_mi"] / (summary["time_fast_min"] / 60.0)
        summary["avg_mph"] = summary["distance_mi"] / (summary["time_avg_min"] / 60.0)
        summary["time_std_min"] = summary["time_low_min"] - summary["time_avg_min"]
    else:
        speed_lookup = DEFAULT_SEGMENT_SPEED_RANGES_MPH.copy()
        if speed_ranges_mph:
            speed_lookup.update(speed_ranges_mph)

        summary["low_mph"] = summary["route_part"].map(
            lambda part: speed_lookup.get(part, DEFAULT_SEGMENT_SPEED_RANGES_MPH["flat or descent"])[0]
        )
        summary["fast_mph"] = summary["route_part"].map(
            lambda part: speed_lookup.get(part, DEFAULT_SEGMENT_SPEED_RANGES_MPH["flat or descent"])[1]
        )
        summary["avg_mph"] = (summary["low_mph"] + summary["fast_mph"]) / 2.0
        summary["time_avg_min"] = summary["distance_mi"] / summary["avg_mph"] * 60.0
        summary["time_std_min"] = summary["distance_mi"] / summary["low_mph"] * 60.0 - summary["time_avg_min"]

    summary["time (min)"] = summary.apply(
        lambda row: (
            f"{row['time_avg_min']:.0f}"
            if row["route_part"] == "flat or descent"
            else f"{row['time_avg_min']:.0f} \N{PLUS-MINUS SIGN} {row['time_std_min']:.0f}"
        ),
        axis=1,
    )
    return summary


def attach_chunk_section_details(
    df: pd.DataFrame,
    distance_column: str = "step_dist_f",
    timing_basis: Literal["hazard", "chunk"] = "hazard",
    speed_ranges_mph: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Attach shared chunk-section metadata to each segment row."""
    summary = _resolve_chunk_section_frame(
        df,
        distance_column=distance_column,
        timing_basis=timing_basis,
        speed_ranges_mph=speed_ranges_mph,
    )
    annotated = summary[[
        "section_id",
        "route_part",
        "section",
        "climb_gain_ft",
        "road_name",
        "distance_mi",
        "time (min)",
    ]].rename(
        columns={
            "section": "section_label",
            "climb_gain_ft": "section_climb_gain_ft",
            "road_name": "section_road_name",
            "distance_mi": "section_distance_mi",
            "time (min)": "section_time_min",
        }
    )

    if "chunk_state" in df.columns:
        frame = df.copy()
    else:
        original_frame = df.copy()
        frame = detect_chunks(df)
        for column in ["hazard", "hazard_raw", "hazard_label", "Ride Type", "_display_color"]:
            if column in original_frame.columns:
                frame[column] = original_frame.loc[frame.index, column]

    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable")
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable")
    elif "end_i" in frame.columns:
        frame = frame.sort_values("end_i", kind="stable")
    else:
        frame = frame.sort_index(kind="stable")

    frame["section_id"] = frame["chunk_state"].ne(frame["chunk_state"].shift()).cumsum()
    return frame.merge(annotated, on=["section_id"], how="left")


def aggregate_by_hazard(
    df: pd.DataFrame,
    column: str = 'step_dist_m',
    hazard_profile: str = DEFAULT_HAZARD_PROFILE,
) -> pd.DataFrame:
    """Summarize a numeric column by hazard label with percentages and total."""
    frame = apply_hazard_profile(df, hazard_profile=hazard_profile)
    summary = (
        frame.groupby(["hazard", "hazard_label"], dropna=False, as_index=False)[column]
        .sum()
        .sort_values(by=column, ascending=False)
        .rename(columns={column: column})
    )

    total = summary[column].sum()
    summary["percent"] = summary[column] / total * 100

    grand_total = pd.DataFrame([{
        "hazard": "TOTAL",
        "hazard_label": "TOTAL",
        column: total,
        "percent": 100.0
    }])

    summary_with_total = pd.concat([summary, grand_total], ignore_index=True).round()
    return summary_with_total


def aggregate_by_road_quality(
    df: pd.DataFrame,
    column: str = "step_dist_f",
) -> pd.DataFrame:
    """Summarize route distance by PCI availability and MTC PCI category in miles and percent."""
    frame = df.copy()
    mtc_info = frame.get("mtc_pci_info", pd.Series(index=frame.index, dtype="object"))
    frame["mtc_pci_info_grouped"] = mtc_info.where(
        ~mtc_info.fillna("").astype(str).str.endswith(" (Unknown)"),
        "Unknown",
    )
    summary = (
        frame.groupby(["pci_available", "mtc_pci_info_grouped"], dropna=False)[column]
        .sum()
        .div(5280)
        .sort_values(ascending=False)
        .reset_index(name="Miles")
        .assign(Percent=lambda frame: frame["Miles"] / frame["Miles"].sum() * 100)
        .sort_values(["pci_available", "Percent"], ascending=[True, False])
        .round(1)
        .rename(columns={"mtc_pci_info_grouped": "mtc_pci_info"})
        .set_index(["mtc_pci_info"])
        .drop(columns=["pci_available"])
    )
    return summary


def road_quality_score(
    df: pd.DataFrame,
    column: str = "step_dist_f",
) -> float:
    """Return the share of ride miles on good PCI-rated roads, excluding gravel and cycleway-unknown miles."""
    pci_report = aggregate_by_road_quality(df, column=column)
    good_roads = [
        "Excellent",
        "Very Good",
        "Good",
        "Fair"
    ]
    miles = pci_report["Miles"].to_dict()
    gravel = miles.get("Gravel", 0.0) or 0.0
    cycleway = miles.get("Cycleway (Unknown)", 0.0) or 0.0
    denominator = float(pci_report["Miles"].sum() - gravel - cycleway)
    if denominator <= 0:
        return 0.0

    numerator = float(
        pci_report.loc[pci_report.index.intersection(good_roads), "Miles"].sum()
    )
    return numerator / denominator


def summarize_chunk_sections(
    df: pd.DataFrame,
    distance_column: str = "step_dist_f",
    timing_basis: Literal["hazard", "chunk"] = "hazard",
    speed_ranges_mph: dict[str, tuple[float, float]] | None = None,
    include_rest_periods: bool = True,
) -> pd.DataFrame:
    """Summarize chunk sections with merged labels and climb gain for climb chunks."""
    summary = _resolve_chunk_section_frame(
        df,
        distance_column=distance_column,
        timing_basis=timing_basis,
        speed_ranges_mph=speed_ranges_mph,
    )
    summary = summary[summary["distance_mi"] > 0].copy()
    if not include_rest_periods:
        summary = summary[summary["route_part"] != "flat or descent"].copy()
    sections = summary[[
        "section",
        "route_part",
        "road_name",
        "chunk_avg_grade",
        "climb_number",
        "climb_gain_ft",
        "distance_mi",
        "time (min)",
    ]].copy()
    sections["section"] = sections.apply(
        lambda row: _chunk_table_section_label(
            row["route_part"],
            row["road_name"],
            row["chunk_avg_grade"],
            int(row["climb_number"]) if pd.notna(row["climb_number"]) else None,
        ),
        axis=1,
    )
    sections["climb_gain_ft"] = sections.apply(
        lambda row: row["climb_gain_ft"] if row["route_part"] != "flat or descent" else pd.NA,
        axis=1,
    )
    sections = sections[[
        "section",
        "climb_gain_ft",
        "distance_mi",
        "time (min)",
    ]].round({"climb_gain_ft": 0, "distance_mi": 1}).rename(
        columns={
            "section": "Section (avg grade)",
            "climb_gain_ft": "Climb (ft)",
            "distance_mi": "Distance (mi)",
            "time (min)": "Time (Min)",
        }
    )
    sections["Climb (ft)"] = sections["Climb (ft)"].apply(
        lambda value: f"{value:,.0f}" if pd.notna(value) else pd.NA
    )

    total_row = pd.DataFrame([{
        "Section (avg grade)": "TOTAL",
        "Climb (ft)": f"{summary.loc[summary['route_part'] != 'flat or descent', 'climb_gain_ft'].sum():,.0f}",
        "Distance (mi)": sections["Distance (mi)"].sum(),
        "Time (Min)": f"{summary['time_avg_min'].sum():.0f} \N{PLUS-MINUS SIGN} {summary['time_std_min'].sum():.0f}",
    }])
    df = pd.concat([total_row, sections], ignore_index=True)
    df = df[df['Distance (mi)'] > 0]
    df = df.round({"Distance (mi)": 1})
    df["Climb (ft)"] = df["Climb (ft)"].fillna("")
    return df
