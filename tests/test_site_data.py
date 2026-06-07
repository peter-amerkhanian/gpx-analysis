import unittest

import pandas as pd

from gpx_analysis.site.data import _route_elevation_ylim


class RouteElevationSvgTests(unittest.TestCase):
    def test_low_max_elevation_uses_fixed_ylim(self) -> None:
        elevation = pd.Series([10.0, 125.0, 249.9])

        self.assertEqual(_route_elevation_ylim(elevation), (0.0, 500.0))

    def test_high_max_elevation_uses_automatic_ylim(self) -> None:
        elevation = pd.Series([10.0, 125.0, 250.0])

        self.assertIsNone(_route_elevation_ylim(elevation))


if __name__ == "__main__":
    unittest.main()
