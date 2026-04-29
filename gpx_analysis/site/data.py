from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

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
class RouteLinks:
    strava_effort: str | None = None


@dataclass(frozen=True)
class RouteMedia:
    hero_image: str | None = None
    gallery: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteConfig:
    slug: str
    source: str
    title: str | None = None
    reverse: bool = False
    links: RouteLinks = field(default_factory=RouteLinks)
    media: RouteMedia = field(default_factory=RouteMedia)

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title

        stem = Path(self.source).stem.replace("_", " ").replace("-", " ")
        return " ".join(part.capitalize() for part in stem.split())

def load_routes(manifest_path: Path, root: Path) -> list[RouteConfig]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw_routes = payload.get("routes", [])
    if not isinstance(raw_routes, list):
        raise ValueError(f"{manifest_path} must define a top-level 'routes' list")

    routes: list[RouteConfig] = []
    seen_slugs: set[str] = set()
    for index, raw_route in enumerate(raw_routes, start=1):
        if not isinstance(raw_route, dict):
            raise ValueError(f"Route entry #{index} in {manifest_path} must be a mapping")

        slug = str(raw_route.get("slug", "")).strip()
        source = str(raw_route.get("source", "")).strip()
        if not slug:
            raise ValueError(f"Route entry #{index} in {manifest_path} is missing 'slug'")
        if slug in seen_slugs:
            raise ValueError(f"Duplicate route slug '{slug}' in {manifest_path}")
        if not source:
            raise ValueError(f"Route '{slug}' in {manifest_path} is missing 'source'")

        source_path = root / source
        if not source_path.exists():
            raise FileNotFoundError(f"Route '{slug}' source does not exist: {source_path}")

        raw_links = raw_route.get("links") or {}
        if not isinstance(raw_links, dict):
            raise ValueError(f"Route '{slug}' links must be a mapping")

        raw_media = raw_route.get("media") or {}
        if not isinstance(raw_media, dict):
            raise ValueError(f"Route '{slug}' media must be a mapping")

        hero_image = raw_media.get("hero_image")
        if hero_image is not None:
            hero_image = str(hero_image).strip() or None
            if hero_image and not (root / "quarto" / hero_image).exists():
                raise FileNotFoundError(
                    f"Route '{slug}' hero image does not exist under quarto/: {hero_image}"
                )

        gallery_items = tuple(str(item).strip() for item in raw_media.get("gallery", []) if str(item).strip())
        for image_path in gallery_items:
            if not (root / "quarto" / image_path).exists():
                raise FileNotFoundError(
                    f"Route '{slug}' gallery image does not exist under quarto/: {image_path}"
                )

        routes.append(
            RouteConfig(
                slug=slug,
                source=source,
                title=(str(raw_route.get("title", "")).strip() or None),
                reverse=bool(raw_route.get("reverse", False)),
                links=RouteLinks(
                    strava_effort=(str(raw_links.get("strava_effort", "")).strip() or None),
                ),
                media=RouteMedia(
                    hero_image=hero_image,
                    gallery=gallery_items,
                ),
            )
        )
        seen_slugs.add(slug)

    return routes


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
    summary["start_bart_station"] = add_bart_station(points_gdf, step=0)
    summary["end_bart_station"] = add_bart_station(points_gdf, step=len(points_gdf) - 1)
    summary["bart_station"] = summary["start_bart_station"]
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
        "title": route.display_title,
        "source": route.source,
        "reverse": route.reverse,
        "links": asdict(route.links),
        "media": {
            "hero_image": route.media.hero_image,
            "gallery": list(route.media.gallery),
        },
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
        "route_facts_heading": f"{summary['distance_mi']:,.1f} miles<br> {summary['elevation_gain_ft']:,.1f} ft elevation gain<br> Start: {summary['start_bart_station']} BART<br> End: {summary['end_bart_station']} BART",
        "summary_table": pd.DataFrame(
            [
                ["Distance (mi)", summary["distance_mi"]],
                ["Elevation gain (ft)", summary["elevation_gain_ft"]],
                ["Elevation loss (ft)", summary["elevation_loss_ft"]],
                ["Max elevation (ft)", summary["max_elevation_ft"]],
                ["Start BART", summary["start_bart_station"]],
                ["End BART", summary["end_bart_station"]],
            ],
            columns=["Metric", "Value"],
        ),
        "hazards_table": hazard_summary[hazard_summary["hazard"] != "TOTAL"]
        .copy()
        .rename(
            columns={
                "hazard_label": "Hazard",
                "distance_mi": "Distance (mi)",
                "percent": "Percent",
            }
        )[["Hazard", "Distance (mi)", "Percent"]],
    }
    return route_bundle, route_page_context
