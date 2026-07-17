from __future__ import annotations

import io

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROFILE_HIGHLIGHT_THRESHOLD_PERCENT = 10.0
PROFILE_FIXED_YLIM_MAX_ELEVATION_FT = 250.0
PROFILE_FIXED_YLIM_FT = (0.0, 500.0)
GRAVEL_HIGHLIGHT_COLOR = "chocolate"


def _route_elevation_ylim(elevation: pd.Series) -> tuple[float, float] | None:
    max_elevation_ft = pd.to_numeric(elevation, errors="coerce").max()
    if pd.notna(max_elevation_ft) and max_elevation_ft < PROFILE_FIXED_YLIM_MAX_ELEVATION_FT:
        return PROFILE_FIXED_YLIM_FT
    return None


def route_elevation_svg(segments: gpd.GeoDataFrame, debug=False) -> str:
    elevation = pd.to_numeric(segments.get("elevation_f"), errors="coerce")
    if elevation is None or elevation.notna().sum() < 2:
        return ""

    frame = segments.copy()
    color_map = {
        "gravel": GRAVEL_HIGHLIGHT_COLOR,
        "road": "tab:blue",
    }
    if "road_type" in frame.columns:
        frame["profile_surface"] = "road"
        total_segment_distance_m = float(
            pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0).sum()
        )
        gravel_percent = 0.0
        if total_segment_distance_m > 0:
            gravel_distance_m = float(
                pd.to_numeric(
                    frame.loc[frame["road_type"].eq("gravel"), "step_dist_m"],
                    errors="coerce",
                ).fillna(0).sum()
            )
            gravel_percent = gravel_distance_m / total_segment_distance_m * 100.0
        if gravel_percent > PROFILE_HIGHLIGHT_THRESHOLD_PERCENT:
            frame.loc[frame["road_type"].eq("gravel"), "profile_surface"] = "gravel"
        color_col = "profile_surface"
    elif "track" in frame.columns:
        color_col = "track"
    else:
        color_col = None
    frame["roll_elevation"] = elevation.interpolate(limit_direction="both").rolling(10, min_periods=1).mean()

    x = np.arange(len(frame))
    baseline = np.full(len(frame), float(np.nanmin(elevation)))

    fig, ax = plt.subplots(figsize=(4, 1))
    if color_col is None:
        ax.plot(x, frame["roll_elevation"], linewidth=2.5, alpha=0.9, color="tab:blue")
    else:
        frame["_profile_group"] = frame[color_col].ne(frame[color_col].shift()).cumsum()
        for _, subset in frame.groupby("_profile_group", sort=False):
            if len(subset) == 0:
                continue
            start = int(subset.index[0])
            stop = int(subset.index[-1])
            subset_x = x[start:stop + 1]
            ax.plot(
                subset_x,
                subset["roll_elevation"],
                linewidth=2.5,
                alpha=0.9,
                color=color_map.get(subset[color_col].iloc[-1], "tab:blue"),
            )

    ax.plot(
        x,
        baseline,
        linewidth=2,
        color="#8d99ae",
        linestyle=":",
        alpha=0.7,
    )
    profile_ylim = _route_elevation_ylim(elevation)
    if profile_ylim is not None:
        ax.set_ylim(profile_ylim)
    ax.set_axis_off()
    if debug:
        return ax

    svg_buffer = io.StringIO()
    fig.savefig(
        svg_buffer,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    return svg_buffer.getvalue()
