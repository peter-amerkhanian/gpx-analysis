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


def compute_step_metrics(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    rad_coords = frame[["lat", "lon"]].apply(np.deg2rad)
    step_meters = haversine_distances(rad_coords.iloc[:-1], rad_coords.iloc[1:]).diagonal() * 6371000

    lat = frame["lat"].to_numpy()
    lon = frame["lon"].to_numpy()
    bearing = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])

    frame["step_bearing"] = np.array([*bearing, np.nan])
    frame["step_bearing"] = frame["step_bearing"].replace({0: np.nan})
    frame["step_turn"] = frame["step_bearing"].diff().abs()
    frame["step_dist_m"] = np.array([0, *step_meters])
    frame["step_dist_f"] = frame["step_dist_m"] * 3.28084
    frame["step_elevation_m"] = frame["elevation_m"].diff().fillna(0)
    frame["step_elevation_f"] = frame["elevation_f"].diff().fillna(0)
    frame["step_grade"] = frame["step_elevation_m"] / frame["step_dist_m"].replace(0, np.nan)
    return frame


def detect_hazards(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    frame = df.copy()
    frame["avg_step_grade"] = frame["step_grade"].rolling(rolling_window).mean()
    frame["avg_bearing_change"] = frame["step_turn"].rolling(rolling_window).mean()
    thresholds = {
        "light_descent": -0.03,
        "steep_descent": -0.1,
        "ultra_steep_descent": -0.2,
        "turn": 24
    }
    on_descent = (frame["step_grade"].round(2) < thresholds["light_descent"])
    on_steep_descent = (frame["step_grade"].round(2) < thresholds["steep_descent"])
    on_ultra_steep_descent = (frame["step_grade"].round(2) < thresholds["ultra_steep_descent"])
    on_turn = (frame["step_turn"].round() > thresholds["turn"])
    
    coming_off_descent = (frame["step_grade"].shift(-1).round(2) < np.mean([thresholds["light_descent"], thresholds["steep_descent"]]))
    coming_off_steep_descent = (frame["step_grade"].shift(-1).round(2) < np.mean([thresholds["steep_descent"], thresholds["ultra_steep_descent"]]))

    hazards = {
        "light_descent": on_descent | (frame["avg_step_grade"].round(2) < -0.08),
        "steep_descent": on_steep_descent | (frame["avg_step_grade"].round(2) < -0.15),
        "ultra_steep_descent": on_ultra_steep_descent | (frame["avg_step_grade"].round(2) < -0.2),
        "turn_on_descent": (
            (on_turn & coming_off_descent)
            | (on_turn & on_descent)
            | ((frame["avg_bearing_change"].round() > 24) & (frame["avg_step_grade"].round(2) < -0.05))
        ),
        "turn_on_steep_descent": (
            (on_turn & coming_off_steep_descent)
            | (on_turn & on_steep_descent)
            | ((frame["avg_bearing_change"].round() > 24) & (frame["avg_step_grade"].round(2) < -0.1))
        ),
    }

    frame["hazard"] = "none"
    for hazard_name, hazard_condition in hazards.items():
        frame.loc[hazard_condition, "hazard"] = hazard_name

    return frame


def analyze_steps(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    return detect_hazards(compute_step_metrics(df), rolling_window=rolling_window)
