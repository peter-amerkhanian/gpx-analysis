import pandas as pd

def aggregate_by_hazard(df: pd.DataFrame, column='distance_m') -> pd.DataFrame:
    """Summarize a numeric column by hazard label with percentages and total."""
    summary = (
        df.groupby("hazard", dropna=False, as_index=False)[column]
        .sum()
        .sort_values(by=column, ascending=False)
        .rename(columns={column: column})
    )

    total = summary[column].sum()
    summary["percent"] = summary[column] / total * 100

    grand_total = pd.DataFrame([{
        "hazard": "TOTAL",
        column: total,
        "percent": 100.0
    }])

    summary_with_total = pd.concat([summary, grand_total], ignore_index=True).round()
    return summary_with_total
