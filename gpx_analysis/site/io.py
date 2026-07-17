from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].astype("string").where(result[column].notna(), None)
    return result


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_geojson(path: Path, frame: gpd.GeoDataFrame) -> None:
    cleaned = json_ready_frame(frame)
    path.write_text(cleaned.to_json(), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
