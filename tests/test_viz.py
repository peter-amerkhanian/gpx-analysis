import unittest

import geopandas as gpd
from shapely.geometry import LineString

from gpx_analysis.viz import _frames_share_route_overlap


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


if __name__ == "__main__":
    unittest.main()
