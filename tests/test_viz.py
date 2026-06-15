import unittest

import geopandas as gpd
from shapely.geometry import LineString

from gpx_analysis.viz import _frames_share_route_overlap, make_chunk_map


class RouteOverlapTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
