import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from gpx_analysis.geo import _finalize_mtc_unknowns


class MtcUnknownFallbackTests(unittest.TestCase):
    def test_missing_mtc_and_osm_still_gets_unknown_roadway_label(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "mtc_pci_info": [pd.NA],
                "osm_highway": [pd.NA],
            },
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=4326,
        )

        result = _finalize_mtc_unknowns(frame)

        self.assertEqual(result.loc[0, "pci_available"], "PCI Unknown")
        self.assertEqual(result.loc[0, "mtc_pci_info"], "Roadway (Unknown)")

    def test_missing_mtc_with_osm_highway_gets_typed_unknown_label(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "mtc_pci_info": [pd.NA],
                "osm_highway": ["secondary"],
            },
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=4326,
        )

        result = _finalize_mtc_unknowns(frame)

        self.assertEqual(result.loc[0, "pci_available"], "PCI Unknown")
        self.assertEqual(result.loc[0, "mtc_pci_info"], "Secondary (Unknown)")


if __name__ == "__main__":
    unittest.main()
