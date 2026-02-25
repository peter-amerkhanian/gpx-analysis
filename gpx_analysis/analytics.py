import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances


def compute_bearing(rad_n: np.ndarray, rad_n_plus_1: np.ndarray) -> np.ndarray:
    """Compute compass bearing (degrees) from paired start/end radian coordinates."""
    lat1 = rad_n[:, 0]
    lon1 = rad_n[:, 1]
    lat2 = rad_n_plus_1[:, 0]
    lon2 = rad_n_plus_1[:, 1]
    dlon = lon2 - lon1
    y_coord = np.sin(dlon) * np.cos(lat2)
    x_coord = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.rad2deg(np.arctan2(y_coord, x_coord))
    normalized_bearing = (bearing + 360) % 360
    normalized_bearing[normalized_bearing == 0] = np.nan
    return np.array([np.nan, *normalized_bearing])

def compute_turn(rad_n: np.ndarray, rad_n_plus_1: np.ndarray) -> np.ndarray:
    """Compute absolute turn angle (degrees) from paired start/end radian coordinates."""
    step_bearing = compute_bearing(rad_n, rad_n_plus_1)
    delta_bearing = np.diff(step_bearing, prepend=[np.nan])
    return np.abs((delta_bearing + 180) % 360 - 180)

def compute_distance(rad_n: np.ndarray, rad_n_plus_1: np.ndarray) -> np.ndarray:
    """Compute haversine distance in meters from paired start/end radian coordinates."""
    distance = haversine_distances(rad_n, rad_n_plus_1).diagonal() * 6371000
    return np.array([0, *distance])

def compute_step_metrics(df: pd.DataFrame, min_step_dist_m: float = 2.0) -> pd.DataFrame:
    """Add per-step distance, bearing, turn, elevation delta, and grade metrics."""
    frame = df.copy()
    # Sort the data
    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable").reset_index(drop=True)
    else:
        frame = frame.sort_index().reset_index(drop=True)
    # Compute elevations
    frame["elevation_m"] = pd.to_numeric(frame.get("elevation_m"), errors="coerce")
    frame["elevation_f"] = pd.to_numeric(frame.get("elevation_f"), errors="coerce")
    frame["step_elevation_m"] = frame["elevation_m"].diff().fillna(0)
    frame["step_elevation_f"] = frame["elevation_f"].diff().fillna(0)
    # Create radian inputs for distance and bearing calculations
    rad_coords = frame[["lat", "lon"]].apply(np.deg2rad).to_numpy()
    rad_n = rad_coords[:-1]
    rad_n_plus_1 = rad_coords[1:]
    # Compute distances
    frame["step_dist_m"] = compute_distance(rad_n, rad_n_plus_1)
    frame["step_dist_f"] = frame["step_dist_m"] * 3.28084
    # Compute grade changes
    valid_dist = frame["step_dist_m"] >= min_step_dist_m
    frame["step_grade"] = np.where(valid_dist, frame["step_elevation_m"] / frame["step_dist_m"], np.nan)
    # Compute turns
    frame["step_turn"] = compute_turn(rad_n, rad_n_plus_1)
    return frame


def detect_hazards(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """Classify each step into hazard categories from grade and turning signals."""
    frame = df.copy()
    frame["avg_step_grade"] = frame["step_grade"].rolling(rolling_window).mean()
    frame["avg_bearing_change"] = frame["step_turn"].rolling(rolling_window).mean()
    thresholds = {
        "light_descent": -0.049,
        "steep_descent": -0.099,
        "ultra_steep_descent": -0.199,
        "turn": 19.9,
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
    """Compute step metrics and run hazard detection in one call."""
    return detect_hazards(compute_step_metrics(df), rolling_window=rolling_window)