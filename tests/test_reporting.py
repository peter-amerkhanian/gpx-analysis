import unittest

import pandas as pd

from gpx_analysis.reporting import road_quality_score, summarize_chunk_sections


class RoadQualityScoreTests(unittest.TestCase):
    def test_excludes_unknown_road_miles_from_score(self) -> None:
        frame = pd.DataFrame(
            {
                "step_dist_f": [5280.0, 5280.0, 8 * 5280.0],
                "pci_available": ["PCI Available", "PCI Available", "PCI Unknown"],
                "mtc_pci_info": ["Good", "Poor", "Roadway (Unknown)"],
            }
        )

        self.assertEqual(road_quality_score(frame), 0.5)

    def test_returns_zero_when_no_road_miles_are_rated(self) -> None:
        frame = pd.DataFrame(
            {
                "step_dist_f": [5280.0, 5280.0],
                "pci_available": ["PCI Unknown", "PCI Unknown"],
                "mtc_pci_info": ["Roadway (Unknown)", "Cycleway (Unknown)"],
            }
        )

        self.assertEqual(road_quality_score(frame), 0.0)


class SummarizeChunkSectionsTests(unittest.TestCase):
    def test_climb_section_label_uses_average_grade(self) -> None:
        frame = pd.DataFrame(
            {
                "chunk_state": ["climb (medium)", "climb (medium)"],
                "chunk_avg_grade": [0.052, 0.052],
                "step_dist_f": [1000.0, 1000.0],
                "step_elevation_f": [52.0, 52.0],
                "elevation_f": [100.0, 152.0],
                "osm_name": ["Pinehurst Road", "Pinehurst Road"],
                "hazard": ["climb", "climb"],
            }
        )

        result = summarize_chunk_sections(frame, include_rest_periods=False)

        self.assertIn("Section (avg grade)", result.columns)
        self.assertNotIn("Section", result.columns)
        self.assertEqual(result.iloc[1]["Section (avg grade)"], "1. Pinehurst Road (5% avg)")

    def test_chunk_climb_uses_adjusted_elevation_gain(self) -> None:
        frame = pd.DataFrame(
            {
                "chunk_state": ["climb (easy)"] * 7,
                "chunk_avg_grade": [0.01] * 7,
                "step_dist_m": [0.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
                "step_dist_f": [0.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0],
                "elevation_m": [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0],
                "osm_name": ["Pinehurst Road"] * 7,
                "hazard": ["climb"] * 7,
            }
        )

        result = summarize_chunk_sections(frame, include_rest_periods=False)

        self.assertEqual(result.iloc[0]["Climb (ft)"], "0")


if __name__ == "__main__":
    unittest.main()
