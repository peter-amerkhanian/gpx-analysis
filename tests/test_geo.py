import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from gpx_analysis.geo import (
    _finalize_mtc_unknowns,
    _fill_mtc_gaps_from_osm_continuity,
    _select_best_mtc_match_per_segment,
    _select_best_osm_match_per_segment,
)


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


class SegmentStreetMatchingTests(unittest.TestCase):
    def test_osm_match_prefers_continuing_main_street_over_perpendicular_side_street(self) -> None:
        route_geometries = {
            0: LineString([(0, 0), (20, 0)]),
            1: LineString([(20, 0), (40, 0)]),
            2: LineString([(40, 0), (60, 0)]),
        }
        arlington = LineString([(-10, 1), (70, 1)])
        madera = LineString([(30, -10), (30, 10)])
        matched = pd.DataFrame(
            [
                {
                    "_segment_index": 0,
                    "name": "Arlington Boulevard",
                    "highway": "secondary",
                    "_highway_priority": 3,
                    "_route_geometry": route_geometries[0],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
                {
                    "_segment_index": 1,
                    "name": "Madera Street",
                    "highway": "residential",
                    "_highway_priority": 6,
                    "_route_geometry": route_geometries[1],
                    "_candidate_geometry": madera,
                    "_candidate_dist_m": 0.0,
                },
                {
                    "_segment_index": 1,
                    "name": "Arlington Boulevard",
                    "highway": "secondary",
                    "_highway_priority": 3,
                    "_route_geometry": route_geometries[1],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
                {
                    "_segment_index": 2,
                    "name": "Arlington Boulevard",
                    "highway": "secondary",
                    "_highway_priority": 3,
                    "_route_geometry": route_geometries[2],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
            ]
        )

        result = _select_best_osm_match_per_segment(
            matched,
            overlap_buffer_m=3.0,
            match_preference_tolerance_m=4.0,
        )

        self.assertEqual(result["name"].tolist(), ["Arlington Boulevard"] * 3)

    def test_mtc_match_prefers_continuing_main_street_over_perpendicular_side_street(self) -> None:
        route_geometries = {
            0: LineString([(0, 0), (20, 0)]),
            1: LineString([(20, 0), (40, 0)]),
            2: LineString([(40, 0), (60, 0)]),
        }
        arlington = LineString([(-10, 1), (70, 1)])
        madera = LineString([(30, -10), (30, 10)])
        matched = pd.DataFrame(
            [
                {
                    "_segment_index": 0,
                    "osm_name": "Arlington Boulevard",
                    "road_name": "Arlington Boulevard",
                    "_route_geometry": route_geometries[0],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
                {
                    "_segment_index": 1,
                    "osm_name": "Arlington Boulevard",
                    "road_name": "Madera Street",
                    "_route_geometry": route_geometries[1],
                    "_candidate_geometry": madera,
                    "_candidate_dist_m": 0.0,
                },
                {
                    "_segment_index": 1,
                    "osm_name": "Arlington Boulevard",
                    "road_name": "Arlington Boulevard",
                    "_route_geometry": route_geometries[1],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
                {
                    "_segment_index": 2,
                    "osm_name": "Arlington Boulevard",
                    "road_name": "Arlington Boulevard",
                    "_route_geometry": route_geometries[2],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
            ]
        )

        result = _select_best_mtc_match_per_segment(
            matched,
            overlap_buffer_m=3.0,
            match_preference_tolerance_m=8.0,
        )

        self.assertEqual(result["road_name"].tolist(), ["Arlington Boulevard"] * 3)

    def test_mtc_match_treats_suffix_variants_as_same_road_across_side_street_cluster(self) -> None:
        route_geometries = {
            i: LineString([(i * 20, 0), ((i + 1) * 20, 0)])
            for i in range(6)
        }
        arlington = LineString([(-10, 1), (130, 1)])
        side_streets = {
            "TAMALPAIS AVENUE": LineString([(30, -10), (30, 10)]),
            "SCENIC STREET": LineString([(50, -10), (50, 10)]),
            "MADERA DRIVE": LineString([(70, -10), (70, 10)]),
        }
        rows = [
            {
                "_segment_index": 0,
                "osm_name": "Arlington Boulevard",
                "road_name": "ARLINGTON",
                "_route_geometry": route_geometries[0],
                "_candidate_geometry": arlington,
                "_candidate_dist_m": 1.0,
            }
        ]
        for segment_index, side_name in enumerate(side_streets, start=1):
            rows.extend(
                [
                    {
                        "_segment_index": segment_index,
                        "osm_name": "Arlington Boulevard",
                        "road_name": side_name,
                        "_route_geometry": route_geometries[segment_index],
                        "_candidate_geometry": side_streets[side_name],
                        "_candidate_dist_m": 0.0,
                    },
                    {
                        "_segment_index": segment_index,
                        "osm_name": "Arlington Boulevard",
                        "road_name": "Arlington Boulevard",
                        "_route_geometry": route_geometries[segment_index],
                        "_candidate_geometry": arlington,
                        "_candidate_dist_m": 1.0,
                    },
                ]
            )
        rows.extend(
            [
                {
                    "_segment_index": 4,
                    "osm_name": "Arlington Avenue",
                    "road_name": "ARLINGTON AVE",
                    "_route_geometry": route_geometries[4],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
                {
                    "_segment_index": 5,
                    "osm_name": "Arlington Avenue",
                    "road_name": "Arlington Avenue",
                    "_route_geometry": route_geometries[5],
                    "_candidate_geometry": arlington,
                    "_candidate_dist_m": 1.0,
                },
            ]
        )
        matched = pd.DataFrame(rows)

        result = _select_best_mtc_match_per_segment(
            matched,
            overlap_buffer_m=3.0,
            match_preference_tolerance_m=8.0,
        )

        self.assertEqual(
            result["road_name"].tolist(),
            [
                "ARLINGTON",
                "Arlington Boulevard",
                "Arlington Boulevard",
                "Arlington Boulevard",
                "ARLINGTON AVE",
                "Arlington Avenue",
            ],
        )

    def test_mtc_match_rejects_perpendicular_side_street_when_osm_name_disagrees(self) -> None:
        route = LineString([(0, 0), (20, 0)])
        matched = pd.DataFrame(
            [
                {
                    "_segment_index": 0,
                    "osm_name": "Arlington Boulevard",
                    "road_name": "TAMALPAIS AVENUE",
                    "_route_geometry": route,
                    "_candidate_geometry": LineString([(10, -10), (10, 10)]),
                    "_candidate_dist_m": 0.0,
                },
                {
                    "_segment_index": 0,
                    "osm_name": "Arlington Boulevard",
                    "road_name": "SCENIC STREET",
                    "_route_geometry": route,
                    "_candidate_geometry": LineString([(12, -10), (12, 10)]),
                    "_candidate_dist_m": 0.0,
                },
            ]
        )

        result = _select_best_mtc_match_per_segment(
            matched,
            overlap_buffer_m=3.0,
            match_preference_tolerance_m=8.0,
        )

        self.assertTrue(result.empty)


class MtcContinuityFillTests(unittest.TestCase):
    def test_fills_mtc_gap_when_osm_and_mtc_neighbors_share_road_key(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "osm_name": [
                    "Arlington Boulevard",
                    "Arlington Boulevard",
                    "Arlington Boulevard",
                    "Arlington Avenue",
                ],
                "mtc_road_name": ["ARLINGTON", pd.NA, pd.NA, "ARLINGTON AVE"],
                "mtc_pci_info": ["Good", pd.NA, pd.NA, "Good"],
                "mtc_pci_date": ["2024", pd.NA, pd.NA, "2024"],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
                LineString([(2, 0), (3, 0)]),
                LineString([(3, 0), (4, 0)]),
            ],
            crs=4326,
        )

        result = _fill_mtc_gaps_from_osm_continuity(frame)

        self.assertEqual(result.loc[1, "mtc_road_name"], "ARLINGTON")
        self.assertEqual(result.loc[2, "mtc_road_name"], "ARLINGTON")
        self.assertEqual(result.loc[1, "mtc_pci_info"], "Good")
        self.assertEqual(result.loc[2, "mtc_pci_date"], "2024")

    def test_does_not_fill_mtc_gap_when_neighbors_are_different_roads(self) -> None:
        frame = gpd.GeoDataFrame(
            {
                "osm_name": ["Arlington Boulevard", "Arlington Boulevard", "The Circle"],
                "mtc_road_name": ["ARLINGTON", pd.NA, "THE CIRCLE"],
                "mtc_pci_info": ["Good", pd.NA, "Fair"],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
                LineString([(2, 0), (3, 0)]),
            ],
            crs=4326,
        )

        result = _fill_mtc_gaps_from_osm_continuity(frame)

        self.assertTrue(pd.isna(result.loc[1, "mtc_road_name"]))
        self.assertTrue(pd.isna(result.loc[1, "mtc_pci_info"]))


if __name__ == "__main__":
    unittest.main()
