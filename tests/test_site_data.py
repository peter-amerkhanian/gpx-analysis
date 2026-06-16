import unittest

import pandas as pd

from gpx_analysis.site.data import _route_elevation_ylim, route_tags_from_segments


class RouteElevationSvgTests(unittest.TestCase):
    def test_low_max_elevation_uses_fixed_ylim(self) -> None:
        elevation = pd.Series([10.0, 125.0, 249.9])

        self.assertEqual(_route_elevation_ylim(elevation), (0.0, 500.0))

    def test_high_max_elevation_uses_automatic_ylim(self) -> None:
        elevation = pd.Series([10.0, 125.0, 250.0])

        self.assertIsNone(_route_elevation_ylim(elevation))


class RouteTagsFromSegmentsTests(unittest.TestCase):
    def test_returns_tags_above_thresholds_in_distance_order(self) -> None:
        segments = pd.DataFrame(
            {
                "osm_name": [
                    "Redwood Road",
                    "Wildcat Creek Trail",
                    "Redwood Road",
                    "Meadows Canyon Trail",
                ],
                "step_dist_f": [40000, 21000, 15000, 7500],
            }
        )

        result = route_tags_from_segments(segments)

        self.assertEqual(
            [tag["label"] for tag in result],
            ["Redwood Road", "Wildcat Creek Trail", "Meadows Canyon Trail"],
        )
        self.assertEqual(result[0]["distance_ft"], 55000.0)
        self.assertEqual(result[0]["threshold_ft"], 54000.0)

    def test_excludes_roads_below_thresholds(self) -> None:
        segments = pd.DataFrame(
            {
                "osm_name": ["Redwood Road", "Wildcat Creek Trail"],
                "step_dist_f": [53999, 19999],
            }
        )

        self.assertEqual(route_tags_from_segments(segments), [])

    def test_missing_columns_returns_empty_tags(self) -> None:
        segments = pd.DataFrame({"osm_name": ["Redwood Road"]})

        self.assertEqual(route_tags_from_segments(segments), [])


if __name__ == "__main__":
    unittest.main()
