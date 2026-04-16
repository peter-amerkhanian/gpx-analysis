import pandas as pd
import numpy as np
from .physics import compute_step_metrics

def detect_hazards(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """Classify each step into hazard categories from grade and turning signals."""
    frame = df.copy()
    frame["avg_step_grade"] = frame["step_grade"].rolling(rolling_window).mean()
    frame["avg_bearing_change"] = frame["step_turn"].rolling(rolling_window).mean()
    # thresholds = {
    #     "light_descent": -0.049,
    #     "steep_descent": -0.099,
    #     "ultra_steep_descent": -0.199,
    #     "turn": 19.9,
    #     "climb": 0.049,
    #     "steep_climb": 0.099,
    # }
    thresholds = {
        "light_descent": -0.049,
        "steep_descent": -0.099,
        "ultra_steep_descent": -0.249,
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