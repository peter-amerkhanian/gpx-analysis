import unittest

import pandas as pd

from gpx_analysis.analytics import detect_hazards


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


if __name__ == "__main__":
    unittest.main()
