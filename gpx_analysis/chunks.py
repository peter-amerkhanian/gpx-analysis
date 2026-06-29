import numpy as np
import pandas as pd

DEFAULT_CHUNK_CLIMB_THRESHOLD = 0.039
DEFAULT_CHUNK_HARD_CLIMB_THRESHOLD = 0.06
DEFAULT_CHUNK_AVG_CLIMB_THRESHOLD = 0.04
DEFAULT_CHUNK_FLAT_RECOVERY_FT = 700.0
DEFAULT_CHUNK_MIN_DIST_FT = .3 * 5280
DEFAULT_CHUNK_HARD_MIN_DIST_FT = .3 * 5280
DEFAULT_CHUNK_MIN_AVG_GRADE = 0.02


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


def detect_chunks(
    df: pd.DataFrame,
    climb_threshold: float = DEFAULT_CHUNK_CLIMB_THRESHOLD,
    hard_climb_threshold: float = DEFAULT_CHUNK_HARD_CLIMB_THRESHOLD,
    avg_climb_threshold: float = DEFAULT_CHUNK_AVG_CLIMB_THRESHOLD,
    flat_recovery_ft: float = DEFAULT_CHUNK_FLAT_RECOVERY_FT,
    min_chunk_dist_ft: float = DEFAULT_CHUNK_MIN_DIST_FT,
    hard_min_chunk_dist_ft: float = DEFAULT_CHUNK_HARD_MIN_DIST_FT,
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
        required_distance_ft = (
            hard_min_chunk_dist_ft if state == "hard_climb" else min_chunk_dist_ft
        )
        if chunk_distance_ft <= required_distance_ft:
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

    def _run_chunk_pass(
        state: str,
        start_threshold: float,
        recovery_threshold: float,
    ) -> None:
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
            is_rest = pd.notna(row_grade) and float(row_grade) < recovery_threshold
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

    _run_chunk_pass(
        "hard_climb",
        start_threshold=hard_climb_threshold,
        recovery_threshold=hard_climb_threshold,
    )
    _run_chunk_pass(
        "climb",
        start_threshold=climb_threshold,
        recovery_threshold=0.02,
    )

    return frame
