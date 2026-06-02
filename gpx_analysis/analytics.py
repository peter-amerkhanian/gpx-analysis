import pandas as pd
import numpy as np
from .physics import compute_step_metrics

DEFAULT_CHUNK_CLIMB_THRESHOLD = 0.039
DEFAULT_CHUNK_HARD_CLIMB_THRESHOLD = 0.07
DEFAULT_CHUNK_AVG_CLIMB_THRESHOLD = 0.04
DEFAULT_CHUNK_FLAT_RECOVERY_FT = 700.0
DEFAULT_CHUNK_MIN_DIST_FT = 1000.0
DEFAULT_CHUNK_MIN_AVG_GRADE = 0.02

DEFAULT_HAZARD_SHORT_SEGMENT_M = 100


def _weighted_grade_average(frame: pd.DataFrame) -> float:
    """Return the distance-weighted average grade for a chunk."""
    valid = frame["step_grade"].notna() & frame["step_dist_m"].notna()
    if not valid.any():
        return np.nan

    distances = frame.loc[valid, "step_dist_m"]
    total_distance = distances.sum()
    if total_distance <= 0:
        return np.nan
    return float((frame.loc[valid, "step_grade"] * distances).sum() / total_distance)


def _weighted_grade_median(frame: pd.DataFrame) -> float:
    """Return the distance-weighted median grade for a chunk."""
    valid = frame["step_grade"].notna() & frame["step_dist_m"].notna()
    if not valid.any():
        return np.nan

    ordered = (
        pd.DataFrame({
            "grade": frame.loc[valid, "step_grade"],
            "distance": frame.loc[valid, "step_dist_m"],
        })
        .sort_values("grade", kind="stable")
        .reset_index(drop=True)
    )
    total_distance = float(ordered["distance"].sum())
    if total_distance <= 0:
        return np.nan

    cutoff = total_distance / 2.0
    cumulative = ordered["distance"].cumsum()
    median_index = cumulative.ge(cutoff).idxmax()
    return float(ordered.loc[median_index, "grade"])


def _smooth_short_segment_grades(
    frame: pd.DataFrame,
    rolling_window: int,
    short_segment_threshold_m: float,
) -> pd.Series:
    """Smooth only short segments with a centered distance-weighted rolling grade."""
    grade = pd.to_numeric(frame["step_grade"], errors="coerce")
    distance = pd.to_numeric(frame["step_dist_m"], errors="coerce")
    effective_grade = grade.copy()

    if rolling_window <= 1:
        return effective_grade

    valid_distance = distance.where(grade.notna())
    weighted_grade = grade.mul(valid_distance)
    rolling_distance = valid_distance.rolling(rolling_window, center=True, min_periods=1).sum()
    rolling_grade = weighted_grade.rolling(rolling_window, center=True, min_periods=1).sum()
    smoothed_grade = rolling_grade.div(rolling_distance.where(rolling_distance > 0))

    short_segment_mask = distance.le(short_segment_threshold_m) & grade.notna()
    effective_grade.loc[short_segment_mask] = smoothed_grade.loc[short_segment_mask]
    return effective_grade


def _merge_short_flat_recoveries(
    base_state: pd.Series,
    step_dist_f: pd.Series,
    flat_recovery_ft: float,
) -> pd.Series:
    """Bridge short flat runs when they sit between same-direction efforts."""
    frame = pd.DataFrame({
        "base_state": base_state.fillna("flat"),
        "step_dist_f": step_dist_f.fillna(0),
    })
    frame["run_id"] = frame["base_state"].ne(frame["base_state"].shift()).cumsum()
    runs = (
        frame.groupby("run_id", as_index=False)
        .agg(
            base_state=("base_state", "first"),
            distance_ft=("step_dist_f", "sum"),
        )
    )

    merged_state = frame["base_state"].copy()
    for idx in range(1, len(runs) - 1):
        current = runs.iloc[idx]
        previous = runs.iloc[idx - 1]
        following = runs.iloc[idx + 1]
        if current["base_state"] != "flat":
            continue
        if current["distance_ft"] > flat_recovery_ft:
            continue
        if previous["base_state"] == following["base_state"] and previous["base_state"] != "flat":
            merged_state.loc[frame["run_id"] == current["run_id"]] = previous["base_state"]

    return merged_state


def detect_hazards(
    df: pd.DataFrame,
    rolling_window: int = 5,
    short_segment_threshold_m: float = DEFAULT_HAZARD_SHORT_SEGMENT_M,
) -> pd.DataFrame:
    """Classify each step from one effective grade signal plus same-step turns."""
    frame = df.copy()
    frame["hazard_grade"] = _smooth_short_segment_grades(
        frame,
        rolling_window=rolling_window,
        short_segment_threshold_m=short_segment_threshold_m,
    )
    frame["avg_step_grade"] = frame["hazard_grade"]
    frame["avg_bearing_change"] = frame["step_turn"]
    thresholds = {
        "light_descent": -0.04,
        "steep_descent": -0.079,
        "ultra_steep_descent": -0.199,
        "turn": 19.9,
        "climb": 0.04,
        "steep_climb": 0.079,
    }
    round_n = 5
    hazard_grade = frame["hazard_grade"]
    grade_rounded = hazard_grade.round(round_n)
    on_turn = frame["step_turn"].round() > thresholds["turn"]

    frame["hazard"] = "flat"
    frame.loc[grade_rounded <= thresholds["light_descent"], "hazard"] = "light_descent"
    frame.loc[grade_rounded <= thresholds["steep_descent"], "hazard"] = "steep_descent"
    frame.loc[grade_rounded <= thresholds["ultra_steep_descent"], "hazard"] = "ultra_steep_descent"
    frame.loc[grade_rounded >= thresholds["climb"], "hazard"] = "climb"
    frame.loc[grade_rounded >= thresholds["steep_climb"], "hazard"] = "steep_climb"
    frame.loc[on_turn & (grade_rounded <= thresholds["light_descent"]), "hazard"] = "turn_on_descent"
    frame.loc[on_turn & (grade_rounded <= thresholds["steep_descent"]), "hazard"] = "turn_on_steep_descent"

    return frame


def detect_chunks(
    df: pd.DataFrame,
    climb_threshold: float = DEFAULT_CHUNK_CLIMB_THRESHOLD,
    hard_climb_threshold: float = DEFAULT_CHUNK_HARD_CLIMB_THRESHOLD,
    avg_climb_threshold: float = DEFAULT_CHUNK_AVG_CLIMB_THRESHOLD,
    flat_recovery_ft: float = DEFAULT_CHUNK_FLAT_RECOVERY_FT,
    min_chunk_dist_ft: float = DEFAULT_CHUNK_MIN_DIST_FT,
    min_chunk_avg_grade: float = DEFAULT_CHUNK_MIN_AVG_GRADE,
) -> pd.DataFrame:
    """Classify route chunks into sustained climb workouts or flat."""
    frame = df.copy()
    if "time" in frame.columns and frame["time"].notna().any():
        frame = frame.sort_values("time", kind="stable")
    elif "step" in frame.columns:
        frame = frame.sort_values("step", kind="stable")
    elif "end_i" in frame.columns:
        frame = frame.sort_values("end_i", kind="stable")
    else:
        frame = frame.sort_index(kind="stable")
    frame["hazard"] = "flat"
    frame["chunk_label"] = "flat or descent"
    frame["chunk_avg_grade"] = np.nan
    frame["chunk_median_grade"] = np.nan
    frame["chunk_dist_ft"] = np.nan
    frame["candidate_chunk_dist_ft"] = np.nan
    frame["chunk_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["chunk_base"] = "flat"
    frame["chunk_state"] = "flat or descent"

    grade = pd.to_numeric(frame["step_grade"], errors="coerce")
    frame.loc[grade >= climb_threshold, "chunk_base"] = "climb"
    frame.loc[grade >= hard_climb_threshold, "chunk_base"] = "hard_climb"

    next_chunk_id = 1

    def _finalize_chunk(indices: list[int], state: str | None) -> None:
        nonlocal next_chunk_id
        if state is None or not indices:
            return

        chunk = frame.loc[indices]
        chunk_distance_ft = float(chunk["step_dist_f"].fillna(0).sum())
        frame.loc[indices, "candidate_chunk_dist_ft"] = chunk_distance_ft
        if chunk_distance_ft <= min_chunk_dist_ft:
            return

        avg_grade = _weighted_grade_average(chunk)
        median_grade = _weighted_grade_median(chunk)
        climb_score = max(avg_grade, median_grade) if pd.notna(median_grade) else avg_grade
        if pd.isna(climb_score) or climb_score <= min_chunk_avg_grade:
            return
        if state == "hard_climb" and climb_score <= hard_climb_threshold:
            return
        if state == "hard_climb":
            hazard = "steep_climb"
            chunk_state = "climb (hard)"
            label = "climb (hard)"
        elif climb_score >= avg_climb_threshold:
            hazard = "climb"
            chunk_state = "climb (medium)"
            label = "climb (medium)"
        else:
            hazard = "mellow"
            chunk_state = "climb (easy)"
            label = "climb (easy)"

        frame.loc[indices, "hazard"] = hazard
        frame.loc[indices, "chunk_label"] = label
        frame.loc[indices, "chunk_avg_grade"] = avg_grade
        frame.loc[indices, "chunk_median_grade"] = median_grade
        frame.loc[indices, "chunk_dist_ft"] = chunk_distance_ft
        frame.loc[indices, "chunk_id"] = next_chunk_id
        frame.loc[indices, "chunk_state"] = chunk_state
        next_chunk_id += 1

    def _run_chunk_pass(state: str, start_threshold: float) -> None:
        active_state: str | None = None
        active_indices: list[int] = []
        recovery_indices: list[int] = []
        recovery_distance_ft = 0.0

        for idx, row in frame.iterrows():
            if pd.notna(frame.at[idx, "chunk_id"]):
                if recovery_indices:
                    active_indices.extend(recovery_indices)
                    recovery_indices = []
                    recovery_distance_ft = 0.0
                _finalize_chunk(active_indices, active_state)
                active_state = None
                active_indices = []
                continue

            row_grade = pd.to_numeric(row["step_grade"], errors="coerce")
            distance_ft = float(row["step_dist_f"]) if pd.notna(row["step_dist_f"]) else 0.0
            is_rest = pd.notna(row_grade) and float(row_grade) < 0.02
            starts_chunk = pd.notna(row_grade) and float(row_grade) >= start_threshold

            if active_state is None:
                if starts_chunk:
                    active_state = state
                    active_indices = [idx]
                continue

            if is_rest:
                recovery_indices.append(idx)
                recovery_distance_ft += distance_ft
                if recovery_distance_ft >= flat_recovery_ft:
                    _finalize_chunk(active_indices, active_state)
                    active_state = None
                    active_indices = []
                    recovery_indices = []
                    recovery_distance_ft = 0.0
                continue

            if recovery_indices:
                active_indices.extend(recovery_indices)
                recovery_indices = []
                recovery_distance_ft = 0.0

            if starts_chunk:
                active_indices.append(idx)
                continue

            active_indices.append(idx)

        if recovery_indices:
            active_indices.extend(recovery_indices)
        _finalize_chunk(active_indices, active_state)

    _run_chunk_pass("hard_climb", hard_climb_threshold)
    _run_chunk_pass("climb", climb_threshold)

    return frame


def analyze_steps(
    df: pd.DataFrame,
    rolling_window: int = 3,
    short_segment_threshold_m: float = DEFAULT_HAZARD_SHORT_SEGMENT_M,
) -> pd.DataFrame:
    """Compute step metrics and run hazard detection in one call."""
    return detect_hazards(
        compute_step_metrics(df),
        rolling_window=rolling_window,
        short_segment_threshold_m=short_segment_threshold_m,
    )


def analyze_chunks(
    df: pd.DataFrame,
    climb_threshold: float = DEFAULT_CHUNK_CLIMB_THRESHOLD,
    hard_climb_threshold: float = DEFAULT_CHUNK_HARD_CLIMB_THRESHOLD,
    avg_climb_threshold: float = DEFAULT_CHUNK_AVG_CLIMB_THRESHOLD,
    flat_recovery_ft: float = DEFAULT_CHUNK_FLAT_RECOVERY_FT,
    min_chunk_dist_ft: float = DEFAULT_CHUNK_MIN_DIST_FT,
    min_chunk_avg_grade: float = DEFAULT_CHUNK_MIN_AVG_GRADE,
) -> pd.DataFrame:
    """Compute step metrics and run chunk detection in one call."""
    return detect_chunks(
        compute_step_metrics(df),
        climb_threshold=climb_threshold,
        hard_climb_threshold=hard_climb_threshold,
        avg_climb_threshold=avg_climb_threshold,
        flat_recovery_ft=flat_recovery_ft,
        min_chunk_dist_ft=min_chunk_dist_ft,
        min_chunk_avg_grade=min_chunk_avg_grade,
    )
