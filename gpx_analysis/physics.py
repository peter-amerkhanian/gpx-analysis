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


def compute_elevation_totals(
    frame: pd.DataFrame,
    elevation_delta_column: str = "step_elevation_m",
) -> dict[str, float]:
    """Return total climbing and descending from per-step elevation deltas."""
    deltas = pd.to_numeric(frame.get(elevation_delta_column), errors="coerce").fillna(0)
    climbing_m = float(deltas.clip(lower=0).sum())
    descending_m = float(deltas.clip(upper=0).abs().sum())
    return {
        "elevation_gain_m": climbing_m,
        "elevation_gain_ft": climbing_m * 3.28084,
        "elevation_loss_m": descending_m,
        "elevation_loss_ft": descending_m * 3.28084,
    }


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

def _smooth_grade_by_distance(
    grade: np.ndarray,
    distances_m: np.ndarray,
    window_m: float,
) -> np.ndarray:
    """Return a centered, distance-weighted grade average."""
    if window_m <= 0:
        return grade.copy()

    clean_grade = np.nan_to_num(np.asarray(grade, dtype=float), nan=0.0)
    clean_dist = np.nan_to_num(np.asarray(distances_m, dtype=float), nan=0.0)
    positions = np.cumsum(clean_dist)
    half_window_m = window_m / 2.0
    smoothed = np.empty_like(clean_grade)

    for i, center in enumerate(positions):
        left = np.searchsorted(positions, center - half_window_m, side="left")
        right = np.searchsorted(positions, center + half_window_m, side="right")
        weights = clean_dist[left:right]
        values = clean_grade[left:right]
        total_weight = weights.sum()
        smoothed[i] = np.average(values, weights=weights) if total_weight > 0 else clean_grade[i]

    return smoothed

def compute_coast_speed(df, v0=0.0,
                        m_lb=190,        # kg (rider + bike)
                        CdA=0.50,      # m^2
                        Crr=0.008,     # -
                        rho=1.225,     # kg/m^3
                        g=9.81,
                        grade_smoothing_window_m=30.0):
    """
    Coasting speed compounded across segments.
    Assumes: downhill segments have NEGATIVE grade (e.g., -0.10 for -10%).
    Requires columns:
      - step_grade (decimal)
      - step_dist_m (meters)

    Tunable parameters:
      - grade_smoothing_window_m reduces acceleration from short grade spikes
    """
    m = m_lb * 0.45359237 # kg
    L = df["step_dist_m"].fillna(0).to_numpy(dtype=float)
    raw_grade = df["step_grade"].fillna(0).to_numpy(dtype=float)
    grade = _smooth_grade_by_distance(raw_grade, L, grade_smoothing_window_m)
    speeds = np.empty(len(df), dtype=float)
    v = float(v0)
    k_drag = rho * CdA / m  # constant in the v^2 drag term
    for i in range(len(df)):
        # Convert "negative = downhill" into a positive downhill component
        downhill_component = -grade[i]  # e.g. grade=-0.10 => downhill_component=+0.10
        # Update v^2 using a simple energy/force approximation over distance
        v2 = (
            v**2
            + 2.0 * g * (downhill_component - Crr) * L[i]
            - k_drag * v*v * L[i]
        )
        v = np.sqrt(v2) if v2 > 0 else 0.0
        speeds[i] = v
    df = df.copy()
    df["coast_speed_mps"] = speeds
    df["coast_speed_mph"] = speeds * 2.2369362920544
    return df
