import unittest

import pandas as pd

from gpx_analysis.reporting import summarize_chunk_sections


class SummarizeChunkSectionsTests(unittest.TestCase):
    def test_climb_section_label_uses_average_grade(self) -> None:
        frame = pd.DataFrame(
            {
                "chunk_state": ["climb (medium)", "climb (medium)"],
                "chunk_avg_grade": [0.052, 0.052],
                "step_dist_f": [1000.0, 1000.0],
                "step_elevation_f": [52.0, 52.0],
                "osm_name": ["Pinehurst Road", "Pinehurst Road"],
                "hazard": ["climb", "climb"],
            }
        )

        result = summarize_chunk_sections(frame, include_rest_periods=False)

        self.assertIn("Section (avg grade)", result.columns)
        self.assertNotIn("Section", result.columns)
        self.assertEqual(result.iloc[1]["Section (avg grade)"], "1. Pinehurst Road (5% avg)")


if __name__ == "__main__":
    unittest.main()
