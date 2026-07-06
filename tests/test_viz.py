import unittest

import geopandas as gpd
from shapely.geometry import LineString

from gpx_analysis.viz import (
    _frames_share_route_overlap,
    _route_overlap_pass_indexes,
    make_chunk_map,
    make_grade_map,
    make_hazard_map,
    make_road_quality_map,
    make_route_overview_map,
)


class RouteOverlapTests(unittest.TestCase):
    def test_route_overlap_pass_indexes_only_include_overlapping_passes(self) -> None:
        frame = gpd.GeoDataFrame(
            {},
            geometry=[
                LineString([(-200, 0), (-100, 0)]),
                LineString([(0, 0), (100, 0)]),
                LineString([(200, 0), (300, 0)]),
                LineString([(100, 0), (0, 0)]),
                LineString([(400, 0), (500, 0)]),
            ],
            index=["unique-start", "earlier-overlap", "unique-middle", "later-overlap", "unique-end"],
            crs=3857,
        )

        earlier, later = _route_overlap_pass_indexes(frame)

        self.assertEqual(earlier, {"earlier-overlap"})
        self.assertEqual(later, {"later-overlap"})

    def test_flat_shared_overlap_does_not_trigger_direction_split(self) -> None:
        left = gpd.GeoDataFrame(
            {"hazard": ["flat"]},
            geometry=[LineString([(0, 0), (40, 0)])],
            crs=3857,
        )
        right = gpd.GeoDataFrame(
            {"hazard": ["flat"]},
            geometry=[LineString([(0, 0), (40, 0)])],
            crs=3857,
        )

        self.assertFalse(
            _frames_share_route_overlap(
                left,
                right,
                column="hazard",
                ignore_value="flat",
            )
        )

    def test_non_flat_shared_overlap_still_triggers_direction_split(self) -> None:
        left = gpd.GeoDataFrame(
            {"hazard": ["descent"]},
            geometry=[LineString([(0, 0), (40, 0)])],
            crs=3857,
        )
        right = gpd.GeoDataFrame(
            {"hazard": ["descent"]},
            geometry=[LineString([(0, 0), (40, 0)])],
            crs=3857,
        )

        self.assertTrue(
            _frames_share_route_overlap(
                left,
                right,
                column="hazard",
                ignore_value="flat",
            )
        )


class RouteMapTests(unittest.TestCase):
    def test_overview_map_shows_road_name_and_elevation_interactions(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step_dist_m": [100.0],
                "osm_name": ["Pinehurst Road"],
                "elevation_f": [742.4],
            },
            geometry=[LineString([(-122.0, 37.0), (-122.1, 37.1)])],
            crs=4326,
        )

        html = make_route_overview_map(frame).get_root().render()

        self.assertIn('"Road Name"', html)
        self.assertIn('"Elevation (ft)"', html)
        self.assertIn("Pinehurst Road", html)
        self.assertIn("742 ft", html)
        self.assertIn('"More Details"', html)
        self.assertIn("google.com/maps", html)

    def test_hazard_map_shows_road_name_in_tooltip_and_popup(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1],
                "lat": [37.0],
                "lon": [-122.0],
                "step_dist_m": [100.0],
                "step_turn": [0.0],
                "step_grade": [0.04],
                "hazard_grade": [0.04],
                "hazard": ["climb"],
                "osm_name": ["Pinehurst Road"],
            },
            geometry=[LineString([(-122.0, 37.0), (-122.1, 37.1)])],
            crs=4326,
        )

        html = make_hazard_map(frame).get_root().render()

        self.assertIn('"Road Name"', html)
        self.assertIn('"More Details"', html)
        self.assertIn("Pinehurst Road", html)
        self.assertIn("google.com/maps", html)

    def test_gravel_overlay_is_opt_in_and_dashed(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1],
                "lat": [37.0],
                "lon": [-122.0],
                "step_dist_m": [100.0],
                "step_turn": [0.0],
                "step_grade": [0.0],
                "hazard_grade": [0.0],
                "hazard": ["flat"],
                "osm_name": ["Pinehurst Road"],
                "road_type": ["gravel"],
            },
            geometry=[LineString([(-122.0, 37.0), (-122.1, 37.1)])],
            crs=4326,
        )

        default_html = make_hazard_map(frame).get_root().render()
        overlay_html = make_hazard_map(frame, show_gravel_overlay=True).get_root().render()

        self.assertNotIn("route-gravel-overlay", default_html)
        self.assertIn("route-gravel-overlay", overlay_html)
        self.assertIn("style.zIndex = 390", overlay_html)
        self.assertIn('"color": "#b37400"', overlay_html)
        self.assertIn('"weight": 9', overlay_html)

    def test_grade_map_renders_smoothed_grade(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1, 2, 3],
                "step_dist_m": [100.0, 100.0, 100.0],
                "step_grade": [-0.08, 0.0, 0.08],
                "avg_step_grade": [-0.06, 0.0, 0.06],
                "osm_name": ["Pinehurst Road"] * 3,
                "road_type": ["road", "gravel", "road"],
            },
            geometry=[
                LineString([(-122.0, 37.0), (-122.01, 37.01)]),
                LineString([(-122.01, 37.01), (-122.02, 37.02)]),
                LineString([(-122.02, 37.02), (-122.03, 37.03)]),
            ],
            crs=4326,
        )

        html = make_grade_map(frame, smoothing_window_m=250.0).get_root().render()

        self.assertIn("Smoothed Grade", html)
        self.assertIn("smooth_grade", html)
        self.assertIn("Pinehurst Road", html)
        self.assertIn('"More Details"', html)
        self.assertIn("google.com/maps", html)
        self.assertIn("light_all", html)
        self.assertIn("route-gravel-overlay", html)

    def test_grade_map_adds_route_pass_control_for_overlaps(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1, 2, 3, 4],
                "step_dist_m": [1000.0, 1000.0, 1000.0, 1000.0],
                "step_grade": [0.02, 0.06, 0.01, -0.04],
                "avg_step_grade": [0.02, 0.06, 0.01, -0.04],
                "osm_name": ["Pinehurst Road"] * 4,
            },
            geometry=[
                LineString([(-122.0, 37.0), (-122.01, 37.0)]),
                LineString([(-122.01, 37.0), (-122.02, 37.0)]),
                LineString([(-122.02, 37.0), (-122.03, 37.0)]),
                LineString([(-122.02, 37.0), (-122.01, 37.0)]),
            ],
            crs=4326,
        )

        html = make_grade_map(frame).get_root().render()

        self.assertIn("Route Pass", html)
        self.assertIn("Outbound", html)
        self.assertIn("Return", html)

    def test_road_quality_map_does_not_show_gravel_overlay_by_default(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1],
                "lat": [37.0],
                "lon": [-122.0],
                "step_dist_m": [100.0],
                "step_turn": [0.0],
                "step_grade": [0.0],
                "hazard": ["flat"],
                "osm_name": ["Pinehurst Road"],
                "road_type": ["gravel"],
                "mtc_road_name": ["Pinehurst Road"],
                "mtc_pci_info": ["Gravel"],
                "mtc_pci_date": ["2026"],
            },
            geometry=[LineString([(-122.0, 37.0), (-122.1, 37.1)])],
            crs=4326,
        )

        html = make_road_quality_map(frame).get_root().render()

        self.assertIn('"More Details"', html)
        self.assertIn("google.com/maps", html)
        self.assertNotIn("route-gravel-overlay", html)


class ChunkMapTests(unittest.TestCase):
    def test_chunk_touch_target_uses_section_popup_fields(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1, 2],
                "lat": [37.0, 37.1],
                "lon": [-122.0, -122.1],
                "osm_name": ["Pinehurst Road", "Pinehurst Road"],
                "step_turn": [0.0, 1.0],
                "step_grade": [0.05, 0.05],
                "step_dist_m": [304.8, 304.8],
                "step_dist_f": [1000.0, 1000.0],
                "step_elevation_f": [50.0, 50.0],
                "chunk_state": ["climb (medium)", "climb (medium)"],
                "chunk_avg_grade": [0.05, 0.05],
                "chunk_median_grade": [0.05, 0.05],
                "chunk_dist_ft": [2000.0, 2000.0],
                "candidate_chunk_dist_ft": [2000.0, 2000.0],
                "chunk_id": [1, 1],
                "section_id": [1, 1],
                "section_label": ["1. Pinehurst Road: climb (medium)"] * 2,
                "section_climb_gain_ft": [100.0, 100.0],
                "section_distance_mi": [0.4, 0.4],
                "section_time_min": ["4 +/- 1", "4 +/- 1"],
            },
            geometry=[
                LineString([(-122.0, 37.0), (-122.1, 37.1)]),
                LineString([(-122.1, 37.1), (-122.2, 37.2)]),
            ],
            crs=4326,
        )

        html = make_chunk_map(frame).get_root().render()

        self.assertIn('"className": "route-touch-target"', html)
        self.assertIn('"More Details"', html)
        self.assertIn("google.com/maps", html)
        self.assertIn('"Section Time (min)"', html)
        self.assertIn("4 +/- 1", html)
        self.assertIn("1. Pinehurst Road (5% avg)", html)
        self.assertNotIn("1. Pinehurst Road: climb (medium)", html)

    def test_chunk_map_prefers_section_label_road_name_over_segment_start(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1, 2, 3],
                "lat": [37.0, 37.1, 37.2],
                "lon": [-122.0, -122.1, -122.2],
                "osm_name": ["Caldecott Lane", "Caldecott Lane", "Tunnel Road"],
                "step_turn": [0.0, 1.0, 1.0],
                "step_grade": [0.05, 0.05, 0.05],
                "step_dist_m": [304.8, 304.8, 304.8],
                "step_dist_f": [1000.0, 1000.0, 1000.0],
                "step_elevation_f": [50.0, 50.0, 50.0],
                "chunk_state": ["climb (medium)", "climb (medium)", "climb (medium)"],
                "chunk_avg_grade": [0.05, 0.05, 0.05],
                "chunk_median_grade": [0.05, 0.05, 0.05],
                "chunk_dist_ft": [3000.0, 3000.0, 3000.0],
                "candidate_chunk_dist_ft": [3000.0, 3000.0, 3000.0],
                "chunk_id": [1, 1, 1],
                "section_id": [1, 1, 1],
                "section_label": ["1. Tunnel Road: climb (medium)"] * 3,
                "section_climb_gain_ft": [150.0, 150.0, 150.0],
                "section_distance_mi": [0.6, 0.6, 0.6],
                "section_time_min": ["6 +/- 2", "6 +/- 2", "6 +/- 2"],
            },
            geometry=[
                LineString([(-122.0, 37.0), (-122.1, 37.1)]),
                LineString([(-122.1, 37.1), (-122.2, 37.2)]),
                LineString([(-122.2, 37.2), (-122.3, 37.3)]),
            ],
            crs=4326,
        )

        html = make_chunk_map(frame).get_root().render()

        self.assertIn("1. Tunnel Road (5% avg)", html)
        self.assertNotIn("1. Caldecott Lane (5% avg)", html)

    def test_chunk_map_adds_route_pass_control_for_overlaps(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1, 2, 3, 4],
                "lat": [37.0, 37.0, 37.0, 37.0],
                "lon": [-122.0, -122.01, -122.02, -122.01],
                "osm_name": ["Pinehurst Road"] * 4,
                "step_turn": [0.0, 0.0, 0.0, 0.0],
                "step_grade": [0.05, 0.05, -0.05, -0.05],
                "step_dist_m": [1000.0, 1000.0, 1000.0, 1000.0],
                "step_dist_f": [3280.8, 3280.8, 3280.8, 3280.8],
                "step_elevation_f": [164.0, 164.0, -164.0, -164.0],
                "chunk_state": [
                    "climb (medium)",
                    "climb (medium)",
                    "flat or descent",
                    "flat or descent",
                ],
                "chunk_avg_grade": [0.05, 0.05, None, None],
                "chunk_median_grade": [0.05, 0.05, None, None],
                "chunk_dist_ft": [6561.6, 6561.6, None, None],
                "candidate_chunk_dist_ft": [6561.6, 6561.6, None, None],
                "chunk_id": [1, 1, None, None],
                "section_id": [1, 1, 2, 2],
                "section_label": [
                    "1. Pinehurst Road: climb (medium)",
                    "1. Pinehurst Road: climb (medium)",
                    "flat or descent",
                    "flat or descent",
                ],
                "section_climb_gain_ft": [328.0, 328.0, 0.0, 0.0],
                "section_distance_mi": [1.2, 1.2, 1.2, 1.2],
                "section_time_min": ["8 +/- 2", "8 +/- 2", "6", "6"],
            },
            geometry=[
                LineString([(-122.0, 37.0), (-122.01, 37.0)]),
                LineString([(-122.01, 37.0), (-122.02, 37.0)]),
                LineString([(-122.02, 37.0), (-122.01, 37.0)]),
                LineString([(-122.01, 37.0), (-122.0, 37.0)]),
            ],
            crs=4326,
        )

        html = make_chunk_map(frame).get_root().render()

        self.assertIn("Route Pass", html)
        self.assertIn("Outbound", html)
        self.assertIn("Return", html)
        self.assertIn("1. Pinehurst Road (5% avg)", html)

    def test_chunk_map_split_layers_have_section_fields_without_seeded_section_id(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1, 2, 3, 4],
                "lat": [37.0, 37.0, 37.0, 37.0],
                "lon": [-122.0, -122.01, -122.02, -122.01],
                "osm_name": ["Pinehurst Road"] * 4,
                "road_type": ["road", "gravel", "gravel", "road"],
                "step_turn": [0.0, 0.0, 0.0, 0.0],
                "step_grade": [0.05, 0.05, -0.05, -0.05],
                "step_dist_m": [1000.0, 1000.0, 1000.0, 1000.0],
                "step_dist_f": [3280.8, 3280.8, 3280.8, 3280.8],
                "step_elevation_f": [164.0, 164.0, -164.0, -164.0],
                "chunk_state": [
                    "climb (medium)",
                    "climb (medium)",
                    "flat or descent",
                    "flat or descent",
                ],
                "chunk_avg_grade": [0.05, 0.05, None, None],
                "chunk_median_grade": [0.05, 0.05, None, None],
                "chunk_dist_ft": [6561.6, 6561.6, None, None],
                "candidate_chunk_dist_ft": [6561.6, 6561.6, None, None],
                "chunk_id": [1, 1, None, None],
            },
            geometry=[
                LineString([(-122.0, 37.0), (-122.01, 37.0)]),
                LineString([(-122.01, 37.0), (-122.02, 37.0)]),
                LineString([(-122.02, 37.0), (-122.01, 37.0)]),
                LineString([(-122.01, 37.0), (-122.0, 37.0)]),
            ],
            crs=4326,
        )

        html = make_chunk_map(frame).get_root().render()

        self.assertIn("Route Pass", html)
        self.assertIn('"Section"', html)
        self.assertIn("route-gravel-overlay", html)


if __name__ == "__main__":
    unittest.main()
