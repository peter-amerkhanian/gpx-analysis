import pandas as pd

from .analytics import detect_chunks
from .viz import DEFAULT_HAZARD_PROFILE, apply_hazard_profile


DEFAULT_SEGMENT_SPEED_RANGES_MPH = {
    "flat or descent": (9.0, 14.0),
    "climb (easy)": (4.0, 7.0),
    "climb (medium)": (3.0, 6.0),
    "climb (hard)": (2.0, 5.0),
}


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


def _resolve_chunk_section_frame(
    df: pd.DataFrame,
    distance_column: str = "step_dist_f",
    speed_ranges_mph: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Return ordered chunk sections with shared distance and timing metadata."""
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

    speed_lookup = DEFAULT_SEGMENT_SPEED_RANGES_MPH.copy()
    if speed_ranges_mph:
        speed_lookup.update(speed_ranges_mph)

    frame["section_id"] = frame["chunk_state"].ne(frame["chunk_state"].shift()).cumsum()
    summary = (
        frame.groupby(["section_id", "chunk_state"], as_index=False)
        .agg(
            distance_ft=(distance_column, "sum"),
            road_name=("osm_name", _middle_non_empty_value),
        )
        .rename(columns={"chunk_state": "route_part"})
    )
    summary["distance_mi"] = summary["distance_ft"] / 5280.0
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
        lambda row: ""
        if row["route_part"] == "flat or descent"
        else f"{row['time_avg_min']:.0f} \N{PLUS-MINUS SIGN} {row['time_std_min']:.0f}",
        axis=1,
    )
    return summary


def attach_chunk_section_details(
    df: pd.DataFrame,
    distance_column: str = "step_dist_f",
    speed_ranges_mph: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Attach shared chunk-section metadata to each segment row."""
    summary = _resolve_chunk_section_frame(
        df,
        distance_column=distance_column,
        speed_ranges_mph=speed_ranges_mph,
    )
    annotated = summary[[
        "section_id",
        "route_part",
        "road_name",
        "distance_mi",
        "time (min)",
    ]].rename(
        columns={
            "road_name": "section_road_name",
            "distance_mi": "section_distance_mi",
            "time (min)": "section_time_min",
        }
    )

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
    speed_ranges_mph: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Summarize climb sections with distance, road name, and speed/time estimates."""
    summary = _resolve_chunk_section_frame(
        df,
        distance_column=distance_column,
        speed_ranges_mph=speed_ranges_mph,
    )
    climbs_only = summary[summary["route_part"] != "flat or descent"].copy()
    climbs_only = climbs_only[[
        "route_part",
        "road_name",
        "distance_mi",
        "time (min)",
    ]].round({"distance_mi": 1}).rename(
        columns={
            "route_part": "Workout",
            "road_name": "Road",
            "distance_mi": "Distance (mi)",
            "time (min)": "Time (Min)",
        }
    )

    total_row = pd.DataFrame([{
        "Workout": "TOTAL",
        "Road": "TOTAL",
        "Distance (mi)": climbs_only["Distance (mi)"].sum(),
        "Time (Min)": "",
    }]).round({"Distance (mi)": 1})

    return pd.concat([total_row, climbs_only], ignore_index=True)
