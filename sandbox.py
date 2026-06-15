import gpx_analysis as gpxa
from gpx_analysis import compute_elevation_totals

def test_data_set(paths):
    points = gpxa.io.read_simple_gpx(paths[0])
    points = gpxa.physics.compute_step_metrics(points)
    points = gpxa.analytics.analyze_steps(points, rolling_window = 5, short_segment_threshold_m=80.0)
    points_gdf = gpxa.geo.points_frame(points)
    segments = gpxa.geo.points_to_segments(points_gdf)
    segments = gpxa.geo.enrich_segments_with_osm_edges(
        segments, corridor_m=5, match_max_distance_m=1
    )
    segments = gpxa.geo.enrich_segments_with_mtc_streets(
        segments,
        corridor_m=10.0,
        match_max_distance_m=25.0,
        match_preference_tolerance_m=8.0,
        match_window_size=10,
    )
    return segments

# site.route_elevation_svg(segments, debug=True);
# gpxa.viz.make_route_map(segments, hazard_profile="simplified")
# raw = compute_elevation_totals(segments)
# thresholded = compute_elevation_totals(
#     segments.assign(
#         step_elevation_m=segments["step_elevation_m"].where(
#             segments["step_elevation_m"].abs() >= 0.8, 0
#         )
#     )
# )
# gpxa.reporting.summarize_chunk_sections(segments, include_rest_periods=True)
# gpxa.viz.make_chunk_map(segments)
# gpxa.viz.make_road_quality_map(segments)
# gpxa.reporting.aggregate_by_road_quality(segments)
# gpxa.reporting.road_quality_score(segments)