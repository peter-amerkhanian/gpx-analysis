import unittest

import pandas as pd

from gpx_analysis.chunks import detect_chunks
from gpx_analysis.hazards import detect_hazards


class DetectHazardsTests(unittest.TestCase):
    def test_short_segment_grade_spike_is_smoothed(self) -> None:
        frame = pd.DataFrame(
            {
                "step_grade": [0.05, 0.051, -0.20, 0.052, 0.053],
                "step_dist_m": [40.0, 45.0, 3.0, 42.0, 44.0],
                "step_turn": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

        result = detect_hazards(frame, rolling_window=3, short_segment_threshold_m=12.0)

        self.assertGreater(result.loc[2, "hazard_grade"], 0.039)
        self.assertEqual(result.loc[2, "hazard"], "climb")

    def test_long_segment_grade_spike_is_not_smoothed_away(self) -> None:
        frame = pd.DataFrame(
            {
                "step_grade": [0.05, 0.051, -0.20, 0.052, 0.053],
                "step_dist_m": [40.0, 45.0, 30.0, 42.0, 44.0],
                "step_turn": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

        result = detect_hazards(frame, rolling_window=3, short_segment_threshold_m=12.0)

        self.assertAlmostEqual(result.loc[2, "hazard_grade"], -0.20)
        self.assertEqual(result.loc[2, "hazard"], "ultra_steep_descent")

    def test_turn_hazard_uses_same_step_effective_grade_threshold(self) -> None:
        frame = pd.DataFrame(
            {
                "step_grade": [0.0, -0.10, 0.0],
                "step_dist_m": [25.0, 25.0, 25.0],
                "step_turn": [0.0, 30.0, 0.0],
            }
        )

        result = detect_hazards(frame, rolling_window=3, short_segment_threshold_m=12.0)

        self.assertEqual(result.loc[1, "hazard"], "turn_on_steep_descent")


class DetectChunksTests(unittest.TestCase):
    def test_hard_climb_is_split_from_long_gentle_climb(self) -> None:
        frame = pd.DataFrame(
            {
                "step": range(10),
                "step_grade": [0.10] * 4 + [0.02] * 6,
                "step_dist_f": [200.0] * 10,
                "step_dist_m": [60.96] * 10,
            }
        )

        result = detect_chunks(
            frame,
            flat_recovery_ft=300.0,
            min_chunk_dist_ft=500.0,
            hard_min_chunk_dist_ft=500.0,
        )

        self.assertEqual(result.loc[0, "chunk_state"], "climb (hard)")
        self.assertEqual(result.loc[3, "chunk_state"], "climb (hard)")
        self.assertEqual(result.loc[4, "chunk_state"], "flat or descent")


if __name__ == "__main__":
    unittest.main()
