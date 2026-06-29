import pandas as pd

from .chunks import (
    DEFAULT_CHUNK_AVG_CLIMB_THRESHOLD,
    DEFAULT_CHUNK_CLIMB_THRESHOLD,
    DEFAULT_CHUNK_FLAT_RECOVERY_FT,
    DEFAULT_CHUNK_HARD_CLIMB_THRESHOLD,
    DEFAULT_CHUNK_HARD_MIN_DIST_FT,
    DEFAULT_CHUNK_MIN_AVG_GRADE,
    DEFAULT_CHUNK_MIN_DIST_FT,
    detect_chunks,
)
from .hazards import DEFAULT_HAZARD_SHORT_SEGMENT_M, detect_hazards
from .physics import compute_step_metrics


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
    hard_min_chunk_dist_ft: float = DEFAULT_CHUNK_HARD_MIN_DIST_FT,
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
        hard_min_chunk_dist_ft=hard_min_chunk_dist_ft,
        min_chunk_avg_grade=min_chunk_avg_grade,
    )
