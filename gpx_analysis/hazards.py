import pandas as pd

DEFAULT_HAZARD_SHORT_SEGMENT_M = 80


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
        "light_descent": -0.03,
        "steep_descent": -0.069,
        "ultra_steep_descent": -0.159,
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
