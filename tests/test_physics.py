import unittest

import pandas as pd

from gpx_analysis.io import read_simple_gpx
from gpx_analysis.physics import compute_elevation_totals, compute_step_metrics


class ComputeElevationTotalsTests(unittest.TestCase):
    def test_small_noise_does_not_accumulate_as_climbing(self) -> None:
        frame = pd.DataFrame(
            {
                "elevation_m": [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0],
                "step_dist_m": [0.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
            }
        )

        result = compute_elevation_totals(frame, smoothing_window_m=0, reversal_threshold_m=4.0)

        self.assertEqual(result["elevation_gain_m"], 0.0)
        self.assertEqual(result["elevation_loss_m"], 0.0)

    def test_sustained_climb_and_descent_are_counted(self) -> None:
        frame = pd.DataFrame(
            {
                "elevation_m": [100.0, 105.0, 112.0, 119.0, 108.0],
                "step_dist_m": [0.0, 50.0, 50.0, 50.0, 50.0],
            }
        )

        result = compute_elevation_totals(frame, smoothing_window_m=0, reversal_threshold_m=4.0)

        self.assertEqual(result["elevation_gain_m"], 19.0)
        self.assertEqual(result["elevation_loss_m"], 11.0)

    def test_default_totals_are_close_to_reference_routes(self) -> None:
        route_targets_ft = {
            "gpx_data/Three_Wild_Bears.gpx": 2953.0,
            "gpx_data/OAK_Hills_Skyline_Loop.gpx": 2062.0,
        }

        for route_path, target_ft in route_targets_ft.items():
            with self.subTest(route_path=route_path):
                points = compute_step_metrics(read_simple_gpx(route_path))
                result = compute_elevation_totals(points)

                self.assertLess(abs(result["elevation_gain_ft"] - target_ft), 100.0)


if __name__ == "__main__":
    unittest.main()
