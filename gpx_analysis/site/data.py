from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .. import (
    aggregate_by_hazard,
    aggregate_by_road_quality,
    analyze_steps,
    attach_chunk_section_details,
    compute_elevation_totals,
    points_frame,
    enrich_segments_with_osm_edges,
    enrich_segments_with_mtc_streets,
    make_chunk_map,
    make_road_quality_map,
    make_route_map,
    points_to_segments,
    prepare_segment_display_columns,
    read_simple_gpx,
    road_quality_score,
    summarize_chunk_sections,
)
from ..geo import add_bart_station

GRAVEL_TITLE_THRESHOLD_PERCENT = 10.0
CYCLEWAY_TITLE_THRESHOLD_PERCENT = 20.0
PROFILE_HIGHLIGHT_THRESHOLD_PERCENT = 10.0
PROFILE_FIXED_YLIM_MAX_ELEVATION_FT = 250.0
PROFILE_FIXED_YLIM_FT = (0.0, 500.0)
GRAVEL_HIGHLIGHT_COLOR = "chocolate"
CYCLEWAY_HIGHLIGHT_COLOR = "forestgreen"


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


def _route_elevation_ylim(elevation: pd.Series) -> tuple[float, float] | None:
    max_elevation_ft = pd.to_numeric(elevation, errors="coerce").max()
    if pd.notna(max_elevation_ft) and max_elevation_ft < PROFILE_FIXED_YLIM_MAX_ELEVATION_FT:
        return PROFILE_FIXED_YLIM_FT
    return None


def route_elevation_svg(segments: gpd.GeoDataFrame, debug=False) -> str:
    elevation = pd.to_numeric(segments.get("elevation_f"), errors="coerce")
    if elevation is None or elevation.notna().sum() < 2:
        return ""

    frame = segments.copy()
    color_map = {
        "gravel": GRAVEL_HIGHLIGHT_COLOR,
        "road": "tab:blue",
    }
    if "road_type" in frame.columns:
        frame["profile_surface"] = "road"
        total_segment_distance_m = float(
            pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0).sum()
        )
        gravel_percent = 0.0
        if total_segment_distance_m > 0:
            gravel_distance_m = float(
                pd.to_numeric(
                    frame.loc[frame["road_type"].eq("gravel"), "step_dist_m"],
                    errors="coerce",
                ).fillna(0).sum()
            )
            gravel_percent = gravel_distance_m / total_segment_distance_m * 100.0
        if gravel_percent > PROFILE_HIGHLIGHT_THRESHOLD_PERCENT:
            frame.loc[frame["road_type"].eq("gravel"), "profile_surface"] = "gravel"
        color_col = "profile_surface"
    elif "track" in frame.columns:
        color_col = "track"
    else:
        color_col = None
    frame["roll_elevation"] = elevation.interpolate(limit_direction="both").rolling(10, min_periods=1).mean()

    x = np.arange(len(frame))
    baseline = np.full(len(frame), float(np.nanmin(elevation)))

    fig, ax = plt.subplots(figsize=(4, 1))
    if color_col is None:
        ax.plot(x, frame["roll_elevation"], linewidth=2.5, alpha=0.9, color="tab:blue")
    else:
        frame["_profile_group"] = frame[color_col].ne(frame[color_col].shift()).cumsum()
        for _, subset in frame.groupby("_profile_group", sort=False):
            if len(subset) == 0:
                continue
            start = int(subset.index[0])
            stop = int(subset.index[-1])
            subset_x = x[start:stop + 1]
            ax.plot(
                subset_x,
                subset["roll_elevation"],
                linewidth=2.5,
                alpha=0.9,
                color=color_map.get(subset[color_col].iloc[-1], "tab:blue"),
            )

    ax.plot(
        x,
        baseline,
        linewidth=2,
        color="#8d99ae",
        linestyle=":",
        alpha=0.7,
    )
    profile_ylim = _route_elevation_ylim(elevation)
    if profile_ylim is not None:
        ax.set_ylim(profile_ylim)
    ax.set_axis_off()
    if debug:
        return ax
    else:
        svg_buffer = io.StringIO()
        fig.savefig(
            svg_buffer,
            format="svg",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        svg = svg_buffer.getvalue()
    return svg


def compute_route_summary(points: pd.DataFrame, segments: gpd.GeoDataFrame) -> dict[str, object]:
    total_distance_m = float(points["step_dist_m"].sum())
    elevation_totals = compute_elevation_totals(points)

    max_row = points.loc[points["elevation_m"].fillna(float("-inf")).idxmax()] if points["elevation_m"].notna().any() else None

    return {
        "point_count": int(len(points)),
        "segment_count": int(len(segments)),
        "distance_m": round(total_distance_m, 2),
        "distance_mi": round(total_distance_m / 1609.344, 2),
        "elevation_gain_m": round(elevation_totals["elevation_gain_m"], 2),
        "elevation_gain_ft": round(elevation_totals["elevation_gain_ft"], 2),
        "elevation_loss_m": round(elevation_totals["elevation_loss_m"], 2),
        "elevation_loss_ft": round(elevation_totals["elevation_loss_ft"], 2),
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


def format_duration_hhmm(minutes: float) -> str:
    """Return a duration like 1:30, preserving a leading zero hour for short rides."""
    total_minutes = max(0, int(round(minutes)))
    hours, remaining_minutes = divmod(total_minutes, 60)
    return f"{hours}:{remaining_minutes:02d}"


def total_estimated_time_minutes(chunk_sections_summary: pd.DataFrame) -> float:
    section_column = (
        "Section (avg grade)"
        if "Section (avg grade)" in chunk_sections_summary.columns
        else "Section"
    )
    total_rows = chunk_sections_summary[chunk_sections_summary[section_column].eq("TOTAL")]
    if total_rows.empty:
        return 0.0

    total_time = str(total_rows.iloc[0]["Time (Min)"]).strip()
    if not total_time:
        return 0.0
    return float(total_time.split()[0])


def route_display_title(
    base_title: str,
    gravel_percent: float,
    cycleway_percent: float,
) -> str:
    """Return the route title with gravel/cycleway suffixes when those route types are prominent."""
    notes: list[str] = []
    if gravel_percent > GRAVEL_TITLE_THRESHOLD_PERCENT:
        notes.append(f"{round(gravel_percent)}% Gravel")
    if cycleway_percent > CYCLEWAY_TITLE_THRESHOLD_PERCENT:
        notes.append(f"{round(cycleway_percent)}% Cycleway")
    if notes:
        return f"{base_title} ({', '.join(notes)})"
    return base_title


def route_display_title_html(
    base_title: str,
    gravel_percent: float,
    cycleway_percent: float,
) -> str:
    """Return HTML title markup with colored gravel/cycleway suffixes when applicable."""
    notes: list[str] = []
    if gravel_percent > GRAVEL_TITLE_THRESHOLD_PERCENT:
        notes.append(
            f'<span style="color: {GRAVEL_HIGHLIGHT_COLOR};">{round(gravel_percent)}% Gravel</span>'
        )
    if cycleway_percent > CYCLEWAY_TITLE_THRESHOLD_PERCENT:
        notes.append(
            f'<span style="color: {CYCLEWAY_HIGHLIGHT_COLOR};">{round(cycleway_percent)}% Cycleway</span>'
        )
    if notes:
        return f'{base_title} <span>({", ".join(notes)})</span>'
    return base_title


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
    segments = enrich_segments_with_osm_edges(segments)
    segments = enrich_segments_with_mtc_streets(segments)
    segments = prepare_segment_display_columns(segments, hazard_profile=hazard_profile)
    segments = attach_chunk_section_details(segments)

    summary = compute_route_summary(analyzed, segments)
    total_segment_distance_m = float(pd.to_numeric(segments.get("step_dist_m"), errors="coerce").fillna(0).sum())
    gravel_distance_m = float(
        pd.to_numeric(
            segments.loc[segments.get("road_type").eq("gravel"), "step_dist_m"],
            errors="coerce",
        ).fillna(0).sum()
    ) if "road_type" in segments.columns else 0.0
    cycleway_distance_m = float(
        pd.to_numeric(
            segments.loc[segments.get("osm_highway").eq("cycleway"), "step_dist_m"],
            errors="coerce",
        ).fillna(0).sum()
    ) if "osm_highway" in segments.columns else 0.0
    gravel_percent = (gravel_distance_m / total_segment_distance_m * 100.0) if total_segment_distance_m > 0 else 0.0
    cycleway_percent = (cycleway_distance_m / total_segment_distance_m * 100.0) if total_segment_distance_m > 0 else 0.0
    summary["gravel_percent"] = round(gravel_percent, 1)
    summary["cycleway_percent"] = round(cycleway_percent, 1)
    summary["road_quality_score"] = int(round(road_quality_score(segments) * 100))
    summary["start_bart_station"] = add_bart_station(points_gdf, step=0)
    summary["end_bart_station"] = add_bart_station(points_gdf, step=len(points_gdf) - 1)
    summary["bart_station"] = summary["start_bart_station"]
    display_title = route_display_title(route.display_title, gravel_percent, cycleway_percent)
    display_title_html = route_display_title_html(route.display_title, gravel_percent, cycleway_percent)
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
    road_quality_map = make_road_quality_map(segments)
    chunk_map = make_chunk_map(segments)
    road_quality_summary = aggregate_by_road_quality(segments).reset_index()
    chunk_sections_summary = summarize_chunk_sections(segments)
    climb_only_sections_summary = summarize_chunk_sections(segments, include_rest_periods=False)
    estimated_time_min = total_estimated_time_minutes(chunk_sections_summary)
    summary["estimated_time_min"] = round(estimated_time_min, 0)
    summary["estimated_time_display"] = format_duration_hhmm(estimated_time_min)

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
    elevation_profile_svg = route_elevation_svg(segments)
    write_text(route_dir / "profile.svg", elevation_profile_svg)
    route_map.save(str(route_dir / "map.html"))
    road_quality_map.save(str(route_dir / "road_quality_map.html"))
    chunk_map.save(str(route_dir / "chunk_map.html"))

    route_bundle = {
        "slug": route.slug,
        "title": display_title,
        "title_html": display_title_html,
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
            "road_quality_map": f"data/routes/{route.slug}/road_quality_map.html",
            "chunk_map": f"data/routes/{route.slug}/chunk_map.html",
            "profile_svg": f"data/routes/{route.slug}/profile.svg",
            "page": f"routes/{route.slug}.qmd",
        },
        "hazards": hazard_summary.to_dict(orient="records"),
    }

    route_page_context = {
        "route": route,
        "route_bundle": route_bundle,
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
        "road_quality_table": road_quality_summary,
        "chunk_sections_table": chunk_sections_summary,
        "climb_only_sections_table": climb_only_sections_summary,
    }
    return route_bundle, route_page_context
