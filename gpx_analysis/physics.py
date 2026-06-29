import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

DEFAULT_ELEVATION_SMOOTHING_WINDOW_M = 230.0
DEFAULT_ELEVATION_REVERSAL_THRESHOLD_M = 4.0


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
    elevation_column: str = "elevation_m",
    distance_column: str = "step_dist_m",
    smoothing_window_m: float = DEFAULT_ELEVATION_SMOOTHING_WINDOW_M,
    reversal_threshold_m: float = DEFAULT_ELEVATION_REVERSAL_THRESHOLD_M,
) -> dict[str, float]:
    """Return total climbing and descending from a smoothed elevation profile.

    Dense GPX points often contain small elevation oscillations that add large
    false totals if every positive point-to-point delta is counted. Smooth the
    profile over distance, then count only sustained reversals.
    """
    elevation = _elevation_profile_from_frame(
        frame,
        elevation_column=elevation_column,
        elevation_delta_column=elevation_delta_column,
    )
    if elevation.empty:
        climbing_m = 0.0
        descending_m = 0.0
    else:
        if distance_column in frame.columns:
            distances = pd.to_numeric(frame[distance_column], errors="coerce").fillna(0)
        else:
            distances = pd.Series(np.ones(len(elevation)), dtype=float)
        smoothed = _smooth_elevation_by_distance(elevation, distances, smoothing_window_m)
        adjusted_deltas = _sustained_elevation_step_deltas(
            smoothed,
            threshold_m=reversal_threshold_m,
        )
        climbing_m = float(np.clip(adjusted_deltas, 0, None).sum())
        descending_m = float(np.clip(adjusted_deltas, None, 0).sum() * -1)
    return {
        "elevation_gain_m": climbing_m,
        "elevation_gain_ft": climbing_m * 3.28084,
        "elevation_loss_m": descending_m,
        "elevation_loss_ft": descending_m * 3.28084,
    }


def compute_adjusted_elevation_deltas(
    frame: pd.DataFrame,
    elevation_column: str = "elevation_m",
    elevation_delta_column: str = "step_elevation_m",
    distance_column: str = "step_dist_m",
    smoothing_window_m: float = DEFAULT_ELEVATION_SMOOTHING_WINDOW_M,
    reversal_threshold_m: float = DEFAULT_ELEVATION_REVERSAL_THRESHOLD_M,
) -> pd.Series:
    """Return per-row elevation deltas from the smoothed sustained-change profile."""
    elevation = _elevation_profile_from_frame(
        frame,
        elevation_column=elevation_column,
        elevation_delta_column=elevation_delta_column,
    )
    result = pd.Series(0.0, index=frame.index, dtype=float)
    if elevation.empty:
        return result

    if distance_column in frame.columns:
        distances = pd.to_numeric(frame[distance_column], errors="coerce").fillna(0)
    else:
        distances = pd.Series(np.ones(len(elevation)), dtype=float)
    smoothed = _smooth_elevation_by_distance(elevation, distances, smoothing_window_m)
    adjusted = _sustained_elevation_step_deltas(
        smoothed,
        threshold_m=reversal_threshold_m,
    )
    result.iloc[: len(adjusted)] = adjusted
    return result


def _elevation_profile_from_frame(
    frame: pd.DataFrame,
    elevation_column: str,
    elevation_delta_column: str,
) -> pd.Series:
    if elevation_column in frame.columns:
        elevation = pd.to_numeric(frame[elevation_column], errors="coerce")
    elif elevation_delta_column in frame.columns:
        deltas = pd.to_numeric(frame[elevation_delta_column], errors="coerce").fillna(0)
        elevation = deltas.cumsum()
    else:
        return pd.Series(dtype=float)
    return elevation.interpolate(limit_direction="both").dropna().reset_index(drop=True)


def _smooth_elevation_by_distance(
    elevation_m: pd.Series,
    distances_m: pd.Series,
    window_m: float,
) -> np.ndarray:
    elevation = elevation_m.to_numpy(dtype=float)
    if window_m <= 0 or len(elevation) < 2:
        return elevation.copy()

    distances = np.nan_to_num(distances_m.to_numpy(dtype=float), nan=0.0)
    if len(distances) != len(elevation):
        distances = np.resize(distances, len(elevation))

    positive_distances = distances[distances > 0]
    fallback_weight = float(np.median(positive_distances)) if len(positive_distances) else 1.0
    positions = np.cumsum(distances)
    half_window_m = window_m / 2.0
    smoothed = np.empty_like(elevation)

    for i, center in enumerate(positions):
        left = np.searchsorted(positions, center - half_window_m, side="left")
        right = np.searchsorted(positions, center + half_window_m, side="right")
        weights = distances[left:right].copy()
        values = elevation[left:right]
        if len(values) == 0:
            smoothed[i] = elevation[i]
            continue
        weights[weights <= 0] = fallback_weight
        smoothed[i] = np.average(values, weights=weights)

    return smoothed


def _sum_sustained_elevation_changes(
    elevation_m: np.ndarray,
    threshold_m: float,
) -> tuple[float, float]:
    adjusted_deltas = _sustained_elevation_step_deltas(elevation_m, threshold_m)
    return float(np.clip(adjusted_deltas, 0, None).sum()), float(
        np.clip(adjusted_deltas, None, 0).sum() * -1
    )


def _sustained_elevation_step_deltas(
    elevation_m: np.ndarray,
    threshold_m: float,
) -> np.ndarray:
    elevation = np.asarray(elevation_m, dtype=float)
    adjusted = np.zeros(len(elevation), dtype=float)
    if len(elevation) < 2:
        return adjusted
    if threshold_m <= 0:
        return np.diff(elevation, prepend=elevation[0])

    anchor = float(elevation[0])
    extremum = anchor
    pending_low = anchor
    pending_high = anchor
    pending_low_idx = 0
    pending_high_idx = 0
    anchor_idx = 0
    extremum_idx = 0
    direction = 0

    def add_run_delta(start_idx: int, end_idx: int) -> None:
        if start_idx == end_idx:
            return
        start, end = sorted((start_idx, end_idx))
        amount = elevation[end_idx] - elevation[start_idx]
        raw_deltas = np.diff(elevation[start : end + 1])
        weights = np.clip(raw_deltas, 0, None) if amount > 0 else np.clip(-raw_deltas, 0, None)
        weight_total = weights.sum()
        if weight_total > 0:
            values = weights / weight_total * abs(amount)
            adjusted[start + 1 : end + 1] += values if amount > 0 else -values
        else:
            adjusted[end_idx] += amount

    for i, value in enumerate(elevation[1:], start=1):
        z = float(value)
        if not np.isfinite(z):
            continue

        if direction == 0:
            if z < pending_low:
                pending_low = z
                pending_low_idx = i
            if z > pending_high:
                pending_high = z
                pending_high_idx = i
            if pending_high - pending_low < threshold_m:
                continue
            if z - pending_low >= pending_high - z:
                direction = 1
                anchor = pending_low
                extremum = pending_high
                anchor_idx = pending_low_idx
                extremum_idx = pending_high_idx
            else:
                direction = -1
                anchor = pending_high
                extremum = pending_low
                anchor_idx = pending_high_idx
                extremum_idx = pending_low_idx
            continue

        if direction > 0:
            if z >= extremum:
                extremum = z
                extremum_idx = i
            elif extremum - z >= threshold_m:
                add_run_delta(anchor_idx, extremum_idx)
                anchor = extremum
                extremum = z
                anchor_idx = extremum_idx
                extremum_idx = i
                direction = -1

        if direction < 0:
            if z <= extremum:
                extremum = z
                extremum_idx = i
            elif z - extremum >= threshold_m:
                add_run_delta(anchor_idx, extremum_idx)
                anchor = extremum
                extremum = z
                anchor_idx = extremum_idx
                extremum_idx = i
                direction = 1

    if direction != 0:
        add_run_delta(anchor_idx, extremum_idx)

    return adjusted


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
