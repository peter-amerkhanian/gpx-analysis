import pandas as pd

from .names import _levenshtein_distance, _normalize_match_text, _road_name_key
from .spatial import _bearing_difference_degrees, _overlap_length_m

def _score_mtc_match_candidates(
    matched: pd.DataFrame,
    overlap_buffer_m: float,
    match_preference_tolerance_m: float,
) -> pd.DataFrame:
    """Score route-to-street candidates using distance, name, overlap, and bearing."""
    if matched.empty:
        return matched

    matched = matched.copy()
    matched["_min_candidate_dist_m"] = matched.groupby("_segment_index")["_candidate_dist_m"].transform("min")
    matched["_within_pref_tolerance"] = (
        matched["_candidate_dist_m"] <= matched["_min_candidate_dist_m"] + match_preference_tolerance_m
    )
    matched["_bearing_diff_deg"] = matched.apply(
        lambda row: _bearing_difference_degrees(row["_route_geometry"], row["_candidate_geometry"]),
        axis=1,
    )
    matched["_bearing_diff_deg"] = matched["_bearing_diff_deg"].fillna(999.0)
    matched["_overlap_length_m"] = matched.apply(
        lambda row: _overlap_length_m(row["_route_geometry"], row["_candidate_geometry"], overlap_buffer_m),
        axis=1,
    )
    matched["_route_length_m"] = matched["_route_geometry"].apply(lambda geometry: getattr(geometry, "length", 0.0))
    matched["_osm_name_norm"] = matched.get("osm_name", pd.Series(index=matched.index, dtype="object")).apply(_normalize_match_text)
    matched["_mtc_road_name_norm"] = matched.get("road_name", pd.Series(index=matched.index, dtype="object")).apply(_normalize_match_text)
    matched["_osm_name_key"] = matched.get("osm_name", pd.Series(index=matched.index, dtype="object")).apply(_road_name_key)
    matched["_mtc_road_name_key"] = matched.get("road_name", pd.Series(index=matched.index, dtype="object")).apply(_road_name_key)
    matched["_name_distance"] = matched.apply(
        lambda row: _levenshtein_distance(row["_osm_name_key"], row["_mtc_road_name_key"]),
        axis=1,
    )
    matched["_has_name_distance"] = matched["_name_distance"].notna()
    matched["_name_distance_rank"] = matched["_name_distance"].fillna(999)
    matched["_name_key_match"] = (
        matched["_osm_name_key"].notna()
        & matched["_mtc_road_name_key"].notna()
        & (matched["_osm_name_key"] == matched["_mtc_road_name_key"])
    )
    matched["_name_key_match_rank"] = (~matched["_name_key_match"]).astype(int)
    matched["_tolerance_rank"] = (~matched["_within_pref_tolerance"]).astype(int)
    matched["_name_rank"] = (~matched["_has_name_distance"]).astype(int)
    return matched

def _score_osm_match_candidates(
    matched: pd.DataFrame,
    overlap_buffer_m: float,
    match_preference_tolerance_m: float,
) -> pd.DataFrame:
    """Score route-to-OSM-edge candidates using distance, highway type, overlap, and bearing."""
    if matched.empty:
        return matched

    scored = matched.copy()
    scored["_min_candidate_dist_m"] = scored.groupby("_segment_index")["_candidate_dist_m"].transform("min")
    scored["_within_pref_tolerance"] = (
        scored["_candidate_dist_m"] <= scored["_min_candidate_dist_m"] + match_preference_tolerance_m
    )
    scored["_candidate_priority"] = scored["_highway_priority"].where(scored["_within_pref_tolerance"], 999)
    scored["_bearing_diff_deg"] = scored.apply(
        lambda row: _bearing_difference_degrees(row["_route_geometry"], row["_candidate_geometry"]),
        axis=1,
    )
    scored["_bearing_diff_deg"] = scored["_bearing_diff_deg"].fillna(999.0)
    _ = overlap_buffer_m
    scored["_overlap_length_m"] = 0.0
    return scored

def _candidate_name_key(value: object) -> str | None:
    """Return the first normalized road name from scalar or semicolon-delimited OSM/MTC values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    first_name = text.split(";")[0].strip()
    return _road_name_key(first_name)

def _select_with_name_continuity(
    scored: pd.DataFrame,
    name_column: str,
    sort_columns: list[str],
    ascending: list[bool],
    match_preference_tolerance_m: float,
    max_continuity_bearing_diff_deg: float = 55.0,
) -> pd.DataFrame:
    """Select candidates while resisting one-segment jumps to crossing streets."""
    if scored.empty:
        return scored

    scored = scored.copy()
    name_values = scored[name_column] if name_column in scored.columns else pd.Series(pd.NA, index=scored.index)
    scored["_candidate_name_key"] = name_values.apply(_candidate_name_key)
    selected_rows: list[pd.Series] = []
    selected_by_segment: dict[object, pd.Series] = {}
    previous_name: str | None = None

    for segment_index, group in scored.groupby("_segment_index", sort=True):
        ranked = group.sort_values(
            by=sort_columns,
            ascending=ascending,
            kind="stable",
        )
        chosen = ranked.iloc[0]
        if previous_name is not None:
            continuity_candidates = ranked[
                (ranked["_candidate_name_key"] == previous_name)
                & ranked["_within_pref_tolerance"]
                & (ranked["_bearing_diff_deg"] <= max_continuity_bearing_diff_deg)
            ]
            if not continuity_candidates.empty:
                chosen = continuity_candidates.iloc[0]

        selected_by_segment[segment_index] = chosen
        selected_rows.append(chosen)
        if chosen["_candidate_name_key"] is not None:
            previous_name = chosen["_candidate_name_key"]

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return selected

    segment_indexes = list(selected["_segment_index"])
    for idx in range(1, len(segment_indexes) - 1):
        current_segment = segment_indexes[idx]
        previous_row = selected_by_segment[segment_indexes[idx - 1]]
        current_row = selected_by_segment[current_segment]
        next_row = selected_by_segment[segment_indexes[idx + 1]]
        previous_name = previous_row["_candidate_name_key"]
        current_name = current_row["_candidate_name_key"]
        next_name = next_row["_candidate_name_key"]
        if previous_name is None or previous_name != next_name or current_name == previous_name:
            continue

        group = scored[scored["_segment_index"] == current_segment].sort_values(
            by=sort_columns,
            ascending=ascending,
            kind="stable",
        )
        replacement_candidates = group[
            (group["_candidate_name_key"] == previous_name)
            & (
                group["_candidate_dist_m"]
                <= float(current_row["_candidate_dist_m"]) + match_preference_tolerance_m
            )
            & (group["_bearing_diff_deg"] <= max_continuity_bearing_diff_deg)
        ]
        if replacement_candidates.empty:
            continue

        replacement = replacement_candidates.iloc[0]
        selected_by_segment[current_segment] = replacement
        selected.iloc[idx] = replacement

    return selected.drop(columns=["_candidate_name_key"], errors="ignore")

def _select_best_mtc_match_per_segment(
    matched: pd.DataFrame,
    overlap_buffer_m: float,
    match_preference_tolerance_m: float,
) -> pd.DataFrame:
    """Return one best-scoring MTC candidate row per route segment."""
    scored = _score_mtc_match_candidates(
        matched,
        overlap_buffer_m=overlap_buffer_m,
        match_preference_tolerance_m=match_preference_tolerance_m,
    )
    if scored.empty:
        return scored
    overlap_ratio = scored["_overlap_length_m"].div(scored["_route_length_m"].where(scored["_route_length_m"] > 0))
    cross_street_mismatch = (
        scored["_osm_name_key"].notna()
        & (~scored["_name_key_match"])
        & (scored["_name_distance_rank"] > 4)
        & (
            (scored["_bearing_diff_deg"] > 60.0)
            | (overlap_ratio.fillna(0.0) < 0.45)
        )
    )
    scored = scored[~cross_street_mismatch].copy()
    if scored.empty:
        return scored

    return _select_with_name_continuity(
        scored,
        name_column="road_name",
        sort_columns=[
            "_tolerance_rank",
            "_name_key_match_rank",
            "_name_rank",
            "_name_distance_rank",
            "_overlap_length_m",
            "_bearing_diff_deg",
            "_candidate_dist_m",
        ],
        ascending=[True, True, True, True, False, True, True],
        match_preference_tolerance_m=match_preference_tolerance_m,
    )

def _select_best_osm_match_per_segment(
    matched: pd.DataFrame,
    overlap_buffer_m: float,
    match_preference_tolerance_m: float,
) -> pd.DataFrame:
    """Return one best-scoring OSM edge candidate row per route segment."""
    scored = _score_osm_match_candidates(
        matched,
        overlap_buffer_m=overlap_buffer_m,
        match_preference_tolerance_m=match_preference_tolerance_m,
    )
    if scored.empty:
        return scored

    return _select_with_name_continuity(
        scored,
        name_column="name",
        sort_columns=[
            "_candidate_priority",
            "_overlap_length_m",
            "_bearing_diff_deg",
            "_candidate_dist_m",
        ],
        ascending=[True, False, True, True],
        match_preference_tolerance_m=match_preference_tolerance_m,
    )
