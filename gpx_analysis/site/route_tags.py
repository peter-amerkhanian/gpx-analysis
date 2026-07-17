from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

DEFAULT_ROUTE_TAGS_PATH = "route_tags.yml"
ROUTE_TAG_ELEVATION_ARROW_THRESHOLD_FT = 200.0


def load_route_tag_thresholds(path: Path) -> dict[str, dict[str, object]]:
    """Load route tag segment thresholds from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_tags = payload.get("route_tags", payload)
    if not isinstance(raw_tags, list):
        raise ValueError(f"{path} must define a 'route_tags' list")

    thresholds: dict[str, dict[str, object]] = {}
    for index, raw_tag in enumerate(raw_tags, start=1):
        if not isinstance(raw_tag, dict):
            raise ValueError(f"Route tag entry #{index} in {path} must be a mapping")

        name = str(raw_tag.get("name", "")).strip()
        if not name:
            raise ValueError(f"Route tag entry #{index} in {path} is missing 'name'")
        if name in thresholds:
            raise ValueError(f"Duplicate route tag name '{name}' in {path}")

        threshold_ft = raw_tag.get("threshold_ft", raw_tag.get("threshold"))
        if threshold_ft is None:
            raise ValueError(f"Route tag '{name}' in {path} is missing 'threshold_ft'")

        config: dict[str, object] = {"threshold_ft": float(threshold_ft)}
        display_name = str(raw_tag.get("display_name", "")).strip()
        if display_name:
            config["display_name"] = display_name
        thresholds[name] = config

    return thresholds


def route_tag_segments_table(
    segments: pd.DataFrame,
    tag_thresholds_ft: dict[str, float | dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Return consecutive named road/trail runs used for route tag detection.

    Unnamed rows enclosed by the same road name are treated as part of that
    road's run. Unnamed rows at either end, or between different roads, remain
    unnamed.
    """
    columns = ["osm_name", "seg_id", "step_dist_f", "step_elevation_f"]
    if "osm_name" not in segments.columns or "step_dist_f" not in segments.columns:
        result = pd.DataFrame(columns=columns)
    else:
        source_columns = ["osm_name", "step_dist_f"]
        if "step_elevation_f" in segments.columns:
            source_columns.append("step_elevation_f")
        frame = segments[source_columns].copy()
        frame["osm_name"] = frame["osm_name"].astype("string").str.strip().replace("", pd.NA)
        previous_name = frame["osm_name"].ffill()
        next_name = frame["osm_name"].bfill()
        enclosed_by_same_name = (
            frame["osm_name"].isna()
            & previous_name.notna()
            & previous_name.eq(next_name)
        )
        frame.loc[enclosed_by_same_name, "osm_name"] = previous_name.loc[enclosed_by_same_name]
        frame["step_dist_f"] = pd.to_numeric(frame["step_dist_f"], errors="coerce").fillna(0)
        if "step_elevation_f" not in frame.columns:
            frame["step_elevation_f"] = 0.0
        frame["step_elevation_f"] = pd.to_numeric(frame["step_elevation_f"], errors="coerce").fillna(0)
        road_name_run_changed = frame["osm_name"].fillna("").ne(frame["osm_name"].shift().fillna(""))
        frame["seg_id"] = road_name_run_changed.astype("int64").cumsum()
        result = (
            frame[frame["osm_name"].notna() & frame["osm_name"].ne("")]
            .groupby(["osm_name", "seg_id"], sort=False)
            .agg(
                step_dist_f=("step_dist_f", "sum"),
                step_elevation_f=("step_elevation_f", "sum"),
            )
            .round(0)
            .reset_index()
            .sort_values(by="seg_id", kind="stable")
        )

    if tag_thresholds_ft is None:
        return result

    result = result.copy()
    threshold_values: list[float | None] = []
    display_names: list[str] = []
    qualifies: list[bool] = []
    for row in result.itertuples(index=False):
        road_name = str(row.osm_name)
        config = tag_thresholds_ft.get(road_name)
        threshold_ft: float | None = None
        display_name = road_name
        if isinstance(config, dict):
            raw_threshold = config.get("threshold_ft", config.get("threshold"))
            if raw_threshold is not None:
                threshold_ft = float(raw_threshold)
            display_name = str(config.get("display_name", road_name)).strip() or road_name
        elif config is not None:
            threshold_ft = float(config)

        threshold_values.append(threshold_ft)
        display_names.append(display_name)
        qualifies.append(threshold_ft is not None and float(row.step_dist_f) >= threshold_ft)

    result["display_name"] = display_names
    result["threshold_ft"] = threshold_values
    result["qualifies_tag"] = qualifies
    return result


def route_tags_from_segments(
    segments: pd.DataFrame,
    tag_thresholds_ft: dict[str, float | dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Return route tags for consecutive named road/trail runs above threshold."""
    thresholds = tag_thresholds_ft
    if "osm_name" not in segments.columns or "step_dist_f" not in segments.columns:
        return []
    if thresholds is None:
        thresholds = load_route_tag_thresholds(Path(DEFAULT_ROUTE_TAGS_PATH))

    road_runs = route_tag_segments_table(segments, thresholds)

    tags: list[dict[str, object]] = []
    for row in road_runs.itertuples(index=False):
        distance_ft = float(row.step_dist_f)
        threshold_ft = row.threshold_ft
        if pd.isna(threshold_ft) or distance_ft < float(threshold_ft):
            continue
        display_name = str(row.display_name)
        elevation_ft = float(row.step_elevation_f)
        if elevation_ft > ROUTE_TAG_ELEVATION_ARROW_THRESHOLD_FT:
            display_name = f"{display_name} \u2191"
        elif elevation_ft < -ROUTE_TAG_ELEVATION_ARROW_THRESHOLD_FT:
            display_name = f"{display_name} \u2193"
        tags.append(
            {
                "label": display_name,
                "distance_ft": round(distance_ft, 1),
                "elevation_ft": round(elevation_ft, 1),
                "threshold_ft": float(threshold_ft),
            }
        )
    return tags
