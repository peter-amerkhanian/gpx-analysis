import pandas as pd

from .viz import DEFAULT_HAZARD_PROFILE, apply_hazard_profile


def aggregate_by_hazard(
    df: pd.DataFrame,
    column: str = 'distance_m',
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
