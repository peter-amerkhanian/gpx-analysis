from __future__ import annotations

from .build import build_route
from .cache import (
    ENRICHED_SEGMENTS_CACHE_NAME,
    ENRICHED_SEGMENTS_DERIVED_COLUMNS,
    ENRICHED_SEGMENTS_DERIVED_PREFIXES,
    load_or_build_enriched_segments,
    strip_enriched_segment_derived_columns,
)
from .io import ensure_dir, json_ready_frame, write_geojson, write_json, write_text
from .profile import (
    GRAVEL_HIGHLIGHT_COLOR,
    PROFILE_FIXED_YLIM_FT,
    PROFILE_FIXED_YLIM_MAX_ELEVATION_FT,
    PROFILE_HIGHLIGHT_THRESHOLD_PERCENT,
    _route_elevation_ylim,
    route_elevation_svg,
)
from .route_tags import (
    DEFAULT_ROUTE_TAGS_PATH,
    ROUTE_TAG_ELEVATION_ARROW_THRESHOLD_FT,
    load_route_tag_thresholds,
    route_tag_segments_table,
    route_tags_from_segments,
)
from .routes import RouteConfig, RouteLinks, RouteMedia, load_routes
from .summary import (
    CYCLEWAY_HIGHLIGHT_COLOR,
    CYCLEWAY_TITLE_THRESHOLD_PERCENT,
    GRAVEL_TITLE_THRESHOLD_PERCENT,
    compute_route_summary,
    format_duration_hhmm,
    route_display_title,
    route_display_title_html,
    total_estimated_time_minutes,
)

__all__ = [
    "CYCLEWAY_HIGHLIGHT_COLOR",
    "CYCLEWAY_TITLE_THRESHOLD_PERCENT",
    "DEFAULT_ROUTE_TAGS_PATH",
    "ENRICHED_SEGMENTS_CACHE_NAME",
    "ENRICHED_SEGMENTS_DERIVED_COLUMNS",
    "ENRICHED_SEGMENTS_DERIVED_PREFIXES",
    "GRAVEL_HIGHLIGHT_COLOR",
    "GRAVEL_TITLE_THRESHOLD_PERCENT",
    "PROFILE_FIXED_YLIM_FT",
    "PROFILE_FIXED_YLIM_MAX_ELEVATION_FT",
    "PROFILE_HIGHLIGHT_THRESHOLD_PERCENT",
    "ROUTE_TAG_ELEVATION_ARROW_THRESHOLD_FT",
    "RouteConfig",
    "RouteLinks",
    "RouteMedia",
    "_route_elevation_ylim",
    "build_route",
    "compute_route_summary",
    "ensure_dir",
    "format_duration_hhmm",
    "json_ready_frame",
    "load_or_build_enriched_segments",
    "load_route_tag_thresholds",
    "load_routes",
    "route_display_title",
    "route_display_title_html",
    "route_elevation_svg",
    "route_tag_segments_table",
    "route_tags_from_segments",
    "strip_enriched_segment_derived_columns",
    "total_estimated_time_minutes",
    "write_geojson",
    "write_json",
    "write_text",
]
