import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from gpx_analysis.descent_chunks import (
    detect_descent_chunks,
    make_descent_chunk_map,
    summarize_descent_chunk_sections,
)


class DetectDescentChunksTests(unittest.TestCase):
    def test_detects_descent_chunk_that_reaches_peak_speed_and_min_distance(self) -> None:
        frame = pd.DataFrame(
            {
                "step": range(6),
                "coast_speed_mph": [0.0, 11.0, 16.0, 21.0, 18.0, 10.0],
                "step_dist_f": [0.0, 400.0, 400.0, 400.0, 400.0, 200.0],
                "step_grade": [0.0, -0.03, -0.04, -0.05, -0.04, 0.0],
            }
        )

        result = detect_descent_chunks(frame, end_speed_mph=10.0)

        self.assertEqual(result.loc[1, "descent_chunk_state"], "light descent")
        self.assertEqual(result.loc[4, "descent_chunk_state"], "light descent")
        self.assertEqual(result.loc[5, "descent_chunk_state"], "other")
        self.assertEqual(result.loc[1, "descent_chunk_id"], 1)
        self.assertEqual(result.loc[1, "descent_chunk_dist_ft"], 1600.0)
        self.assertEqual(result.loc[1, "descent_chunk_max_speed_mph"], 21.0)
        self.assertAlmostEqual(result.loc[1, "descent_chunk_avg_speed_mph"], 16.5)
        self.assertAlmostEqual(result.loc[1, "descent_chunk_avg_grade"], -0.04)

    def test_classifies_descent_chunks_by_max_speed(self) -> None:
        frame = pd.DataFrame(
            {
                "coast_speed_mph": [21.0, 10.0, 35.0, 10.0, 45.0, 10.0, 55.0, 10.0],
                "step_dist_f": [1400.0, 100.0, 1400.0, 100.0, 1400.0, 100.0, 1400.0, 100.0],
            }
        )

        result = detect_descent_chunks(frame, end_speed_mph=10.0)

        self.assertEqual(result.loc[0, "descent_chunk_state"], "light descent")
        self.assertEqual(result.loc[2, "descent_chunk_state"], "descent")
        self.assertEqual(result.loc[4, "descent_chunk_state"], "steep descent")
        self.assertEqual(result.loc[6, "descent_chunk_state"], "dangerous descent")

    def test_rejects_run_that_never_reaches_peak_speed(self) -> None:
        frame = pd.DataFrame(
            {
                "coast_speed_mph": [11.0, 15.0, 19.9, 14.0, 10.0],
                "step_dist_f": [400.0] * 5,
            }
        )

        result = detect_descent_chunks(frame, end_speed_mph=10.0)

        self.assertTrue(result["descent_chunk_state"].eq("other").all())
        self.assertTrue(result["descent_chunk_id"].isna().all())

    def test_rejects_run_shorter_than_min_distance(self) -> None:
        frame = pd.DataFrame(
            {
                "coast_speed_mph": [11.0, 22.0, 16.0, 10.0],
                "step_dist_f": [200.0, 200.0, 200.0, 200.0],
            }
        )

        result = detect_descent_chunks(frame, end_speed_mph=10.0)

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
                "step_grade": [-0.02, -0.03, -0.04, -0.03, 0.0],
                "step_dist_f": [400.0, 400.0, 400.0, 400.0, 100.0],
                "step_dist_m": [121.92, 121.92, 121.92, 121.92, 30.48],
                "road_type": ["road", "gravel", "gravel", "road", "road"],
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
        self.assertIn("Average Coast Speed", html)
        self.assertIn("Average Grade", html)
        self.assertIn("1. Pinehurst Road: light descent", html)
        self.assertIn("route-gravel-overlay", html)


class SummarizeDescentChunkSectionsTests(unittest.TestCase):
    def test_summarizes_descent_sections_with_road_quality_score(self) -> None:
        frame = pd.DataFrame(
            {
                "descent_chunk_state": ["light descent", "light descent", "other"],
                "coast_speed_mph": [22.0, 28.0, 4.0],
                "step_grade": [-0.04, -0.06, 0.0],
                "step_dist_f": [500.0, 500.0, 100.0],
                "osm_name": ["Pinehurst Road", "Pinehurst Road", "Pinehurst Road"],
                "mtc_pci_info": ["Excellent", "Fair", "Excellent"],
            }
        )

        result = summarize_descent_chunk_sections(frame)

        self.assertEqual(result.columns.tolist(), [
            "Section",
            "Average Grade",
            "Max Coast Speed",
            "Good+ Pavement",
            "Distance (mi)",
        ])
        self.assertEqual(result.iloc[0]["Section"], "TOTAL")
        self.assertEqual(result.iloc[0]["Good+ Pavement"], "100%")
        self.assertEqual(result.iloc[1]["Section"], "1. Pinehurst Road: light descent")
        self.assertEqual(result.iloc[1]["Average Grade"], "-5.0%")
        self.assertEqual(result.iloc[1]["Max Coast Speed"], "28.0 mph")
        self.assertEqual(result.iloc[1]["Good+ Pavement"], "100%")

    def test_summarizes_gravel_descent_as_gravel(self) -> None:
        frame = pd.DataFrame(
            {
                "descent_chunk_state": ["descent", "descent"],
                "coast_speed_mph": [32.0, 35.0],
                "step_grade": [-0.06, -0.07],
                "step_dist_f": [700.0, 700.0],
                "osm_name": ["Grizzly Peak Trail", "Grizzly Peak Trail"],
                "road_type": ["gravel", "gravel"],
                "mtc_pci_info": ["Roadway (Unknown)", "Roadway (Unknown)"],
            }
        )

        result = summarize_descent_chunk_sections(frame)

        self.assertEqual(result.iloc[0]["Good+ Pavement"], "Gravel")
        self.assertEqual(result.iloc[1]["Good+ Pavement"], "Gravel")

    def test_mixed_gravel_descent_reports_good_pavement_percentage(self) -> None:
        frame = pd.DataFrame(
            {
                "descent_chunk_state": ["descent", "descent"],
                "coast_speed_mph": [32.0, 35.0],
                "step_grade": [-0.06, -0.07],
                "step_dist_f": [700.0, 700.0],
                "osm_name": ["Mixed Road", "Mixed Road"],
                "road_type": ["gravel", "road"],
                "mtc_pci_info": ["Gravel", "Good"],
            }
        )

        result = summarize_descent_chunk_sections(frame)

        self.assertEqual(result.iloc[0]["Good+ Pavement"], "50%")
        self.assertEqual(result.iloc[1]["Good+ Pavement"], "50%")


if __name__ == "__main__":
    unittest.main()
