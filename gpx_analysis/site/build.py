from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .. import (
    aggregate_by_hazard,
    aggregate_by_road_quality,
    attach_chunk_section_details,
    compute_coast_speed,
    compute_step_metrics,
    detect_hazards,
    make_chunk_map,
    make_descent_chunk_map,
    make_grade_map,
    make_road_quality_map,
    make_route_overview_map,
    points_frame,
    points_to_segments,
    prepare_segment_display_columns,
    priority_descent_chunk_sections,
    read_simple_gpx,
    road_quality_score,
    summarize_chunk_sections,
    summarize_descent_chunk_sections,
)
from ..geo import add_bart_station
from .cache import ENRICHED_SEGMENTS_CACHE_NAME, load_or_build_enriched_segments
from .io import ensure_dir, write_geojson, write_json, write_text
from .profile import route_elevation_svg
from .route_tags import DEFAULT_ROUTE_TAGS_PATH, load_route_tag_thresholds, route_tags_from_segments
from .routes import RouteConfig
from .summary import (
    compute_route_summary,
    format_duration_hhmm,
    route_display_title,
    route_display_title_html,
    total_estimated_time_minutes,
)


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
    step_metrics = compute_coast_speed(compute_step_metrics(points))
    points_gdf = points_frame(step_metrics)
    step_segments = points_to_segments(points_gdf)
    segments = load_or_build_enriched_segments(
        step_segments,
        route_dir / ENRICHED_SEGMENTS_CACHE_NAME,
    )
    for column in ["coast_speed_mps", "coast_speed_mph"]:
        if column not in segments.columns and column in step_segments.columns and len(segments) == len(step_segments):
            segments[column] = step_segments[column].to_numpy()
    segments = detect_hazards(segments, rolling_window=3)
    hazard_segments = segments.copy()
    segments = prepare_segment_display_columns(segments, hazard_profile=hazard_profile)
    segments = attach_chunk_section_details(segments)

    summary = compute_route_summary(step_metrics, segments)
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
    route_tag_thresholds = load_route_tag_thresholds(root / DEFAULT_ROUTE_TAGS_PATH)
    summary["route_tags"] = route_tags_from_segments(segments, route_tag_thresholds)
    summary["start_bart_station"] = add_bart_station(points_gdf, step=0)
    summary["end_bart_station"] = add_bart_station(points_gdf, step=len(points_gdf) - 1)
    summary["bart_station"] = summary["start_bart_station"]
    display_title = route_display_title(route.display_title, gravel_percent, cycleway_percent)
    display_title_html = route_display_title_html(route.display_title, gravel_percent, cycleway_percent)
    hazard_summary = aggregate_by_hazard(
        hazard_segments,
        column="step_dist_m",
        hazard_profile=hazard_profile,
    ).rename(
        columns={"step_dist_m": "distance_m"}
    )
    hazard_summary["distance_mi"] = (hazard_summary["distance_m"] / 1609.344).round(2)
    overview_map = make_route_overview_map(segments)
    route_map = make_grade_map(segments)
    descent_chunk_map = make_descent_chunk_map(segments)
    road_quality_map = make_road_quality_map(segments)
    chunk_map = make_chunk_map(segments)
    road_quality_summary = aggregate_by_road_quality(segments).reset_index()
    descent_chunks_summary = summarize_descent_chunk_sections(segments)
    priority_descents_summary = priority_descent_chunk_sections(descent_chunks_summary)
    summary["priority_descent_mi"] = round(
        float(pd.to_numeric(priority_descents_summary["Distance (mi)"], errors="coerce").fillna(0).sum()),
        2,
    )
    chunk_sections_summary = summarize_chunk_sections(segments)
    climb_only_sections_summary = summarize_chunk_sections(segments, include_rest_periods=False)
    estimated_time_min = total_estimated_time_minutes(chunk_sections_summary)
    summary["estimated_time_min"] = round(estimated_time_min, 0)
    summary["estimated_time_display"] = format_duration_hhmm(round(estimated_time_min, -1))

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
    overview_map.save(str(route_dir / "overview_map.html"))
    route_map.save(str(route_dir / "map.html"))
    descent_chunk_map.save(str(route_dir / "descent_chunk_map.html"))
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
            "overview_map": f"data/routes/{route.slug}/overview_map.html",
            "descent_chunk_map": f"data/routes/{route.slug}/descent_chunk_map.html",
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
        "descent_chunks_table": descent_chunks_summary,
        "priority_descents_table": priority_descents_summary,
        "chunk_sections_table": chunk_sections_summary,
        "climb_only_sections_table": climb_only_sections_summary,
    }
    return route_bundle, route_page_context
