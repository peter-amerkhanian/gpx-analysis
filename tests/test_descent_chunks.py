import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from gpx_analysis.descent_chunks import detect_descent_chunks, make_descent_chunk_map


class DetectDescentChunksTests(unittest.TestCase):
    def test_detects_descent_chunk_that_reaches_peak_speed_and_min_distance(self) -> None:
        frame = pd.DataFrame(
            {
                "step": range(6),
                "coast_speed_mph": [0.0, 11.0, 16.0, 21.0, 18.0, 10.0],
                "step_dist_f": [0.0, 400.0, 400.0, 400.0, 400.0, 200.0],
            }
        )

        result = detect_descent_chunks(frame)

        self.assertEqual(result.loc[1, "descent_chunk_state"], "descent")
        self.assertEqual(result.loc[4, "descent_chunk_state"], "descent")
        self.assertEqual(result.loc[5, "descent_chunk_state"], "other")
        self.assertEqual(result.loc[1, "descent_chunk_id"], 1)
        self.assertEqual(result.loc[1, "descent_chunk_dist_ft"], 1600.0)
        self.assertEqual(result.loc[1, "descent_chunk_max_speed_mph"], 21.0)

    def test_rejects_run_that_never_reaches_peak_speed(self) -> None:
        frame = pd.DataFrame(
            {
                "coast_speed_mph": [11.0, 15.0, 19.9, 14.0, 10.0],
                "step_dist_f": [400.0] * 5,
            }
        )

        result = detect_descent_chunks(frame)

        self.assertTrue(result["descent_chunk_state"].eq("other").all())
        self.assertTrue(result["descent_chunk_id"].isna().all())

    def test_rejects_run_shorter_than_min_distance(self) -> None:
        frame = pd.DataFrame(
            {
                "coast_speed_mph": [11.0, 22.0, 16.0, 10.0],
                "step_dist_f": [200.0, 200.0, 200.0, 200.0],
            }
        )

        result = detect_descent_chunks(frame)

        self.assertTrue(result["descent_chunk_state"].eq("other").all())
        self.assertEqual(result.loc[0, "descent_candidate_chunk_dist_ft"], 600.0)


class DescentChunkMapTests(unittest.TestCase):
    def test_make_descent_chunk_map_renders_descent_sections(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "step": [1, 2, 3, 4, 5],
                "lat": [37.0, 37.01, 37.02, 37.03, 37.04],
                "lon": [-122.0, -122.01, -122.02, -122.03, -122.04],
                "osm_name": ["Pinehurst Road"] * 5,
                "coast_speed_mph": [11.0, 18.0, 22.0, 16.0, 10.0],
                "step_dist_f": [400.0, 400.0, 400.0, 400.0, 100.0],
                "step_dist_m": [121.92, 121.92, 121.92, 121.92, 30.48],
            },
            geometry=[
                LineString([(-122.0, 37.0), (-122.01, 37.01)]),
                LineString([(-122.01, 37.01), (-122.02, 37.02)]),
                LineString([(-122.02, 37.02), (-122.03, 37.03)]),
                LineString([(-122.03, 37.03), (-122.04, 37.04)]),
                LineString([(-122.04, 37.04), (-122.05, 37.05)]),
            ],
            crs=4326,
        )

        html = make_descent_chunk_map(frame, tiles="openstreetmap").get_root().render()

        self.assertIn("descent", html)
        self.assertIn("Max Coast Speed", html)
        self.assertIn("1. Pinehurst Road", html)


if __name__ == "__main__":
    unittest.main()
