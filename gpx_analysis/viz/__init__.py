from __future__ import annotations

from .palettes import (
    CHUNK_STATE_COLORS,
    DEFAULT_HAZARD_COLORS,
    DEFAULT_HAZARD_PROFILE,
    DETAILED_HAZARD_COLORS,
    HAZARD_PROFILE_COLORS,
    HAZARD_PROFILE_LABELS,
    HAZARD_PROFILE_REMAPS,
    ROAD_QUALITY_COLORS,
    SIMPLIFIED_HAZARD_COLORS,
    SIMPLIFIED_ROAD_QUALITY_COLORS,
    HazardProfileName,
    resolve_hazard_profile,
    resolve_road_quality_profile,
    resolve_simplified_road_quality_profile,
    simplify_road_quality_category,
)
from .formatters import make_google_maps_link
from .columns import (
    _add_google_maps_details,
    _select_present_columns,
    apply_hazard_profile,
    prepare_osm_columns,
    prepare_segment_display_columns,
)
from .geometry import _frames_share_route_overlap, _route_overlap_pass_indexes
from .folium_base import add_map_elements
from .maps import (
    make_grade_map,
    make_hazard_map,
    make_road_quality_map,
    make_route_overview_map,
)
from .chunk_maps import make_chunk_map

__all__ = [
    "CHUNK_STATE_COLORS",
    "DEFAULT_HAZARD_COLORS",
    "DEFAULT_HAZARD_PROFILE",
    "DETAILED_HAZARD_COLORS",
    "HAZARD_PROFILE_COLORS",
    "HAZARD_PROFILE_LABELS",
    "HAZARD_PROFILE_REMAPS",
    "ROAD_QUALITY_COLORS",
    "SIMPLIFIED_HAZARD_COLORS",
    "SIMPLIFIED_ROAD_QUALITY_COLORS",
    "HazardProfileName",
    "_add_google_maps_details",
    "_frames_share_route_overlap",
    "_route_overlap_pass_indexes",
    "_select_present_columns",
    "add_map_elements",
    "apply_hazard_profile",
    "make_chunk_map",
    "make_google_maps_link",
    "make_grade_map",
    "make_hazard_map",
    "make_road_quality_map",
    "make_route_overview_map",
    "prepare_osm_columns",
    "prepare_segment_display_columns",
    "resolve_hazard_profile",
    "resolve_road_quality_profile",
    "resolve_simplified_road_quality_profile",
    "simplify_road_quality_category",
]

