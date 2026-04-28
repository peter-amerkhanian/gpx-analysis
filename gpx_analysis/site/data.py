from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .. import (
    aggregate_by_hazard,
    analyze_steps,
    make_route_map,
    points_to_segments,
    prepare_segment_display_columns,
    read_simple_gpx,
)
from ..geo import add_bart_station


@dataclass(frozen=True)
class RouteConfig:
    slug: str
    title: str
    source: str
    reverse: bool = False


ROUTES = [
    RouteConfig(slug="arlington", title="Arlington Gravel Loop", source="gpx_data\\arlington_gravel_loop.gpx"),
    RouteConfig(slug="grizzly", title="Grizzly Peak Loop", source="gpx_data\\classic_grizzly.gpx"),
    RouteConfig(slug="tunnel", title="Tunnel to Pinehurst Loop", source="gpx_data\\tunnel_to_pinehurst.gpx"),
    RouteConfig(slug="spruce_to_euclid", title="Spruce up, Euclid Down", source="gpx_data\\spruce_up_euclid_down.gpx"),
    RouteConfig(slug="wildwood", title="Wildwood to BBR Loop", source="gpx_data\\Wildwood_to_bbr_loop.gpx"),
    RouteConfig(slug="three_bears", title="Three Bears with Gravel Loop", source="gpx_data\\three_bears_with_gravel.gpx"),
    RouteConfig(slug="redwood", title="Redwood -- San Leandro to Macarthur BART", source="gpx_data\\redwood_to_macarthur.gpx"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def points_frame(points: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )


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


def compute_route_summary(points: pd.DataFrame, segments: gpd.GeoDataFrame) -> dict[str, object]:
    total_distance_m = float(points["step_dist_m"].sum())
    climbing = points["step_elevation_m"].diff().clip(lower=0)
    descending = points["step_elevation_m"].diff().clip(upper=0).abs()

    max_row = points.loc[points["elevation_m"].fillna(float("-inf")).idxmax()] if points["elevation_m"].notna().any() else None

    return {
        "point_count": int(len(points)),
        "segment_count": int(len(segments)),
        "distance_m": round(total_distance_m, 2),
        "distance_mi": round(total_distance_m / 1609.344, 2),
        "elevation_gain_m": round(float(climbing.sum()), 2),
        "elevation_gain_ft": round(float(climbing.sum()) * 3.28084, 2),
        "elevation_loss_m": round(float(descending.sum()), 2),
        "elevation_loss_ft": round(float(descending.sum()) * 3.28084, 2),
        "max_elevation_m": None if max_row is None else round(float(max_row["elevation_m"]), 2),
        "max_elevation_ft": None if max_row is None else round(float(max_row["elevation_f"]), 2),
        "start": {
            "lat": round(float(points.iloc[0]["lat"]), 6),
            "lon": round(float(points.iloc[0]["lon"]), 6),
        },
        "end": {
            "lat": round(float(points.iloc[-1]["lat"]), 6),
            "lon": round(float(points.iloc[-1]["lon"]), 6),
        },
    }


def build_route(
    route: RouteConfig,
    root: Path,
    routes_dir: Path,
    hazard_profile: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    source_path = root / route.source
    route_dir = routes_dir / route.slug
    ensure_dir(route_dir)

    points = read_simple_gpx(str(source_path), reverse=route.reverse)
    analyzed = analyze_steps(points, rolling_window=3)
    points_gdf = points_frame(analyzed)
    segments = points_to_segments(points_gdf)
    segments = prepare_segment_display_columns(segments, hazard_profile=hazard_profile)

    summary = compute_route_summary(analyzed, segments)
    summary["bart_station"] = add_bart_station(points_gdf, step=0)
    hazard_summary = aggregate_by_hazard(
        analyzed,
        column="step_dist_m",
        hazard_profile=hazard_profile,
    ).rename(
        columns={"step_dist_m": "distance_m"}
    )
    hazard_summary["distance_mi"] = (hazard_summary["distance_m"] / 1609.344).round(2)
    ride_cols = ["Ride Type", "Turn", "Grade", "More Details"]
    route_map = make_route_map(
        segments,
        popup_cols=ride_cols,
        hazard_profile=hazard_profile,
    )

    write_json(
        route_dir / "summary.json",
        {
            "route": asdict(route),
            "summary": summary,
        },
    )
    write_json(
        route_dir / "hazards.json",
        hazard_summary.to_dict(orient="records"),
    )
    write_geojson(route_dir / "points.geojson", points_gdf)
    write_geojson(route_dir / "segments.geojson", segments)
    route_map.save(str(route_dir / "map.html"))

    route_bundle = {
        "slug": route.slug,
        "title": route.title,
        "source": route.source,
        "reverse": route.reverse,
        "summary": summary,
        "paths": {
            "summary": f"data/routes/{route.slug}/summary.json",
            "hazards": f"data/routes/{route.slug}/hazards.json",
            "points": f"data/routes/{route.slug}/points.geojson",
            "segments": f"data/routes/{route.slug}/segments.geojson",
            "map": f"data/routes/{route.slug}/map.html",
            "page": f"routes/{route.slug}.qmd",
        },
        "hazards": hazard_summary.to_dict(orient="records"),
    }

    route_page_context = {
        "route": route,
        "route_facts_heading": f'{summary["distance_mi"]:,.1f} miles<br> {summary["elevation_gain_ft"]:,.1f} ft elevation gain',
        "summary_table": pd.DataFrame(
            [
                ["Distance (mi)", summary["distance_mi"]],
                ["Elevation gain (ft)", summary["elevation_gain_ft"]],
                ["Elevation loss (ft)", summary["elevation_loss_ft"]],
                ["Max elevation (ft)", summary["max_elevation_ft"]],
            ],
            columns=["Metric", "Value"],
        ),
        "hazards_table": hazard_summary[hazard_summary["hazard"] != "TOTAL"].copy().rename(
            columns={
                "hazard_label": "Hazard",
                "distance_mi": "Distance (mi)",
                "percent": "Percent",
            }
        )[["Hazard", "Distance (mi)", "Percent"]],
    }
    return route_bundle, route_page_context
