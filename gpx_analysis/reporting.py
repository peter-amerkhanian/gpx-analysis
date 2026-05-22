import pandas as pd

from .analytics import detect_chunks
from .viz import DEFAULT_HAZARD_PROFILE, apply_hazard_profile


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
    summary = (
        df.groupby(["pci_available", "mtc_pci_info"], dropna=False)[column]
        .sum()
        .div(5280)
        .sort_values(ascending=False)
        .reset_index(name="Miles")
        .assign(Percent=lambda frame: frame["Miles"] / frame["Miles"].sum() * 100)
        .sort_values(["pci_available", "Percent"], ascending=[True, False])
        .round(1)
        .set_index(["pci_available", "mtc_pci_info"])
    )
    return summary


def road_quality_score(
    df: pd.DataFrame,
    column: str = "step_dist_f",
) -> float:
    """Return the share of ride miles on good PCI-rated roads, excluding gravel and cycleway-unknown miles."""
    pci_report = aggregate_by_road_quality(df, column=column)
    good_roads = [
        ("PCI Available", "Excellent"),
        ("PCI Available", "Very Good"),
        ("PCI Available", "Good"),
        ("PCI Available", "Fair"),
    ]
    miles = pci_report["Miles"].to_dict()
    gravel = miles.get(("PCI Available", "Gravel"), 0.0) or 0.0
    cycleway = miles.get(("PCI Unknown", "Cycleway (Unknown)"), 0.0) or 0.0
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
) -> pd.DataFrame:
    """Summarize climb sections with distance, road name, time estimates, and following recovery time."""
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

    speed_lookup = {
        "flat or descent": (10.0, 15.0),
        "climb (easy)": (5.0, 10.0),
        "climb (medium)": (5.0, 10.0),
        "climb (hard)": (2.0, 5.0),
    }

    frame["section_id"] = frame["chunk_state"].ne(frame["chunk_state"].shift()).cumsum()
    summary = (
        frame.groupby(["section_id", "chunk_state"], as_index=False)
        .agg(
            distance_ft=(distance_column, "sum"),
            road_name=("osm_name", lambda values: next((value for value in values if pd.notna(value) and str(value).strip()), "Unknown Road")),
        )
        .rename(columns={"chunk_state": "route_part"})
    )
    summary["distance_mi"] = summary["distance_ft"] / 5280.0
    summary["low_mph"] = summary["route_part"].map(lambda part: speed_lookup.get(part, (10.0, 15.0))[0])
    summary["fast_mph"] = summary["route_part"].map(lambda part: speed_lookup.get(part, (10.0, 15.0))[1])
    summary["time_low_min"] = summary["distance_mi"] / summary["low_mph"] * 60.0
    summary["time_fast_min"] = summary["distance_mi"] / summary["fast_mph"] * 60.0
    summary["rest_time_after_min"] = 0.0

    for idx in range(len(summary) - 1):
        if summary.loc[idx, "route_part"] == "flat or descent":
            continue
        if summary.loc[idx + 1, "route_part"] != "flat or descent":
            continue
        rest_distance_mi = float(summary.loc[idx + 1, "distance_mi"])
        summary.loc[idx, "rest_time_after_min"] = rest_distance_mi / 10.0 * 60.0

    climbs_only = summary[summary["route_part"] != "flat or descent"].copy()

    climbs_only = climbs_only[[
        "route_part",
        "road_name",
        "distance_mi",
        "time_low_min",
        "time_fast_min",
        "rest_time_after_min",
    ]].round({
        "distance_mi": 1,
        "time_low_min": 0,
        "time_fast_min": 0,
        "rest_time_after_min": 0,
    })

    total_row = pd.DataFrame([{
        "route_part": "TOTAL",
        "road_name": "TOTAL",
        "distance_mi": climbs_only["distance_mi"].sum(),
        "time_low_min": climbs_only["time_low_min"].sum(),
        "time_fast_min": climbs_only["time_fast_min"].sum(),
        "rest_time_after_min": climbs_only["rest_time_after_min"].sum(),
    }]).round({
        "distance_mi": 1,
        "time_low_min": 0,
        "time_fast_min": 0,
        "rest_time_after_min": 0,
    })

    return pd.concat([total_row, climbs_only], ignore_index=True)
