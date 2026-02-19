import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances


def bearing_deg(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlon = lon2 - lon1

    y_coord = np.sin(dlon) * np.cos(lat2)
    x_coord = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.rad2deg(np.arctan2(y_coord, x_coord))
    return (bearing + 360) % 360


def compute_step_metrics(df: pd.DataFrame, min_step_dist_m: float = 2.0) -> pd.DataFrame:
    frame = df.copy()
    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable").reset_index(drop=True)
    else:
        frame = frame.sort_index().reset_index(drop=True)

    frame["elevation_m"] = pd.to_numeric(frame.get("elevation_m"), errors="coerce")
    if "elevation_f" in frame.columns:
        frame["elevation_f"] = pd.to_numeric(frame["elevation_f"], errors="coerce")
    rad_coords = frame[["lat", "lon"]].apply(np.deg2rad)
    step_meters = haversine_distances(rad_coords.iloc[:-1], rad_coords.iloc[1:]).diagonal() * 6371000

    lat = frame["lat"].to_numpy()
    lon = frame["lon"].to_numpy()
    bearing = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])

    frame["step_bearing"] = np.array([*bearing, np.nan])
    frame["step_bearing"] = frame["step_bearing"].replace({0: np.nan})
    #  minimum circular difference, always in [0, 180]
    delta_bearing = frame["step_bearing"].diff()
    frame["step_turn"] = ((delta_bearing + 180) % 360 - 180).abs()
    frame["step_dist_m"] = np.array([0, *step_meters])
    frame["step_dist_f"] = frame["step_dist_m"] * 3.28084
    frame["step_elevation_m"] = frame["elevation_m"].diff().fillna(0)
    if "elevation_f" in frame.columns:
        frame["step_elevation_f"] = frame["elevation_f"].diff().fillna(0)
    else:
        frame["step_elevation_f"] = frame["step_elevation_m"] * 3.28084

    valid_dist = frame["step_dist_m"] >= min_step_dist_m
    frame["step_grade"] = np.where(valid_dist, frame["step_elevation_m"] / frame["step_dist_m"], np.nan)
    return frame


def detect_hazards(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    frame = df.copy()
    frame["avg_step_grade"] = frame["step_grade"].rolling(rolling_window).mean()
    frame["avg_bearing_change"] = frame["step_turn"].rolling(rolling_window).mean()
    thresholds = {
        "light_descent": -0.049,
        "steep_descent": -0.099,
        "ultra_steep_descent": -0.199,
        "turn": 19,
        "climb": 0.049,
        "steep_climb": 0.099,
    }
    on_descent = (frame["step_grade"].round(2) < thresholds["light_descent"])
    on_steep_descent = (frame["step_grade"].round(2) < thresholds["steep_descent"])
    on_ultra_steep_descent = (frame["step_grade"].round(2) < thresholds["ultra_steep_descent"])
    on_turn = (frame["step_turn"].round() > thresholds["turn"])
    on_climb = (frame["step_grade"].round(2) > thresholds["climb"])
    on_steep_climb = (frame["step_grade"].round(2) > thresholds["steep_climb"])

    coming_off_descent = (frame["step_grade"].shift(-1).round(2) < np.mean([thresholds["light_descent"], thresholds["steep_descent"]]))
    coming_off_steep_descent = (frame["step_grade"].shift(-1).round(2) < np.mean([thresholds["steep_descent"], thresholds["ultra_steep_descent"]]))
    coming_off_ultra_steep_descent = (frame["step_grade"].shift(-1).round(2) < (thresholds["ultra_steep_descent"] * 1.1))
    hazards = {
        "light_descent": on_descent | coming_off_descent,
        "steep_descent": on_steep_descent | coming_off_steep_descent,
        "ultra_steep_descent": on_ultra_steep_descent | coming_off_ultra_steep_descent,
        "turn_on_descent": (
            (on_turn & coming_off_descent)
            | (on_turn & on_descent)
            | ((frame["avg_bearing_change"].round() > thresholds["turn"]) & (frame["avg_step_grade"].round(2) < thresholds["light_descent"]))
        ),
        "turn_on_steep_descent": (
            (on_turn & coming_off_steep_descent)
            | (on_turn & on_steep_descent)
            | ((frame["avg_bearing_change"].round() > thresholds["turn"]) & (frame["avg_step_grade"].round(2) < thresholds["steep_descent"]))
        ),
        "climb": on_climb,
        "steep_climb": on_steep_climb
    }

    frame["hazard"] = "flat"
    for hazard_name, hazard_condition in hazards.items():
        frame.loc[hazard_condition, "hazard"] = hazard_name

    return frame


def analyze_steps(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    return detect_hazards(compute_step_metrics(df), rolling_window=rolling_window)
