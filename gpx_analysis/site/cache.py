from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from ..geo import enrich_segments_with_mtc_streets, enrich_segments_with_osm_edges
from .io import write_geojson

ENRICHED_SEGMENTS_CACHE_NAME = "segments_enriched.geojson"
ENRICHED_SEGMENTS_DERIVED_PREFIXES = ("chunk_", "candidate_chunk_", "section_")
ENRICHED_SEGMENTS_DERIVED_COLUMNS = {
    "Segment",
    "More Details",
    "Turn",
    "Grade",
    "Hazard Grade",
    "Ride Type",
    "avg_bearing_change",
    "avg_step_grade",
    "hazard",
    "hazard_grade",
    "hazard_label",
    "hazard_raw",
    "Road Name",
    "Elevation (ft)",
    "Road Type",
    "Speed Limit",
    "Section",
    "Distance (mi)",
    "Climb (ft)",
    "Average Grade",
    "Median Grade",
    "Section Road Name",
    "Section Distance (mi)",
    "Section Time (min)",
    "Chunk Avg Grade",
    "Chunk Distance (ft)",
    "Candidate Chunk Distance (ft)",
    "_display_color",
    "section_id",
    "route_part",
    "section_label",
}


def load_or_build_enriched_segments(
    segments: gpd.GeoDataFrame,
    cache_path: Path,
) -> gpd.GeoDataFrame:
    """Load cached OSM/MTC-enriched segments, or build and cache them."""
    if cache_path.exists():
        return strip_enriched_segment_derived_columns(gpd.read_file(cache_path))

    enriched_segments = enrich_segments_with_osm_edges(segments)
    enriched_segments = enrich_segments_with_mtc_streets(enriched_segments)
    write_geojson(cache_path, enriched_segments)
    return enriched_segments


def strip_enriched_segment_derived_columns(
    segments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Remove columns recomputed after OSM/MTC enrichment."""
    drop_columns = [
        column
        for column in segments.columns
        if column in ENRICHED_SEGMENTS_DERIVED_COLUMNS
        or any(column.startswith(prefix) for prefix in ENRICHED_SEGMENTS_DERIVED_PREFIXES)
    ]
    if not drop_columns:
        return segments
    return segments.drop(columns=drop_columns)
