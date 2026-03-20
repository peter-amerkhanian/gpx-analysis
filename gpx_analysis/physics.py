import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances


def compute_bearing(rad_n: np.ndarray, rad_n_plus_1: np.ndarray) -> np.ndarray:
    """Compute compass bearing (degrees) from paired start/end radian coordinates (lat then lon)   ."""
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

def compute_speed(grade, distances, params:dict, v_0=0):
    pass

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
