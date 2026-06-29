import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from gpx_analysis.site.data import (
    _route_elevation_ylim,
    load_route_tag_thresholds,
    load_or_build_enriched_segments,
    route_tags_from_segments,
    strip_enriched_segment_derived_columns,
    write_geojson,
)


class RouteElevationSvgTests(unittest.TestCase):
    def test_low_max_elevation_uses_fixed_ylim(self) -> None:
        elevation = pd.Series([10.0, 125.0, 249.9])

        self.assertEqual(_route_elevation_ylim(elevation), (0.0, 500.0))

    def test_high_max_elevation_uses_automatic_ylim(self) -> None:
        elevation = pd.Series([10.0, 125.0, 250.0])

        self.assertIsNone(_route_elevation_ylim(elevation))


class RouteTagsFromSegmentsTests(unittest.TestCase):
    def test_loads_route_tag_thresholds_from_yaml(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_tags.yml"
            path.write_text(
                "\n".join(
                    [
                        "route_tags:",
                        "  - name: Test Road",
                        "    threshold_ft: 1234",
                        "    display_name: Test",
                    ]
                ),
                encoding="utf-8",
            )

            result = load_route_tag_thresholds(path)

        self.assertEqual(
            result,
            {"Test Road": {"threshold_ft": 1234.0, "display_name": "Test"}},
        )

    def test_returns_consecutive_run_tags_above_thresholds_in_route_order(self) -> None:
        segments = pd.DataFrame(
            {
                "osm_name": [
                    "Wildcat Creek Trail",
                    "Redwood Road",
                    "Redwood Road",
                    "Meadows Canyon Trail",
                    "Wildcat Creek Trail",
                ],
                "step_dist_f": [21000, 40000, 15000, 7500, 22000],
                "step_elevation_f": [100.0, 100.0, 100.0, 100.0, -250.0],
            }
        )

        result = route_tags_from_segments(segments)

        self.assertEqual(
            [tag["label"] for tag in result],
            ["Wildcat Creek", "Redwood Road", "Meadows Canyon", "Wildcat Creek \u2193"],
        )
        self.assertEqual(result[1]["distance_ft"], 55000.0)
        self.assertEqual(result[1]["elevation_ft"], 200.0)
        self.assertEqual(result[1]["threshold_ft"], 54000.0)

    def test_does_not_combine_nonconsecutive_road_runs(self) -> None:
        segments = pd.DataFrame(
            {
                "osm_name": ["Redwood Road", "Other Road", "Redwood Road"],
                "step_dist_f": [40000.0, 100.0, 15000.0],
                "step_elevation_f": [100.0, 0.0, 100.0],
            }
        )

        result = route_tags_from_segments(segments)

        self.assertEqual(result, [])

    def test_uses_optional_display_name_and_appends_climb_arrow(self) -> None:
        segments = pd.DataFrame(
            {
                "osm_name": ["Redwood Road", "Redwood Road"],
                "step_dist_f": [30000.0, 25000.0],
                "step_elevation_f": [350.0, 200.1],
            }
        )

        result = route_tags_from_segments(
            segments,
            tag_thresholds_ft={
                "Redwood Road": {
                    "threshold_ft": 54000,
                    "display_name": "Big Redwood",
                }
            },
        )

        self.assertEqual(result[0]["label"], "Big Redwood \u2191")
        self.assertEqual(result[0]["elevation_ft"], 550.0)

    def test_appends_descent_arrow_to_default_label(self) -> None:
        segments = pd.DataFrame(
            {
                "osm_name": ["Claremont Avenue", "Claremont Avenue"],
                "step_dist_f": [5000.0, 5100.0],
                "step_elevation_f": [-250.0, -251.0],
            }
        )

        result = route_tags_from_segments(
            segments,
            tag_thresholds_ft={"Claremont Avenue": 10000},
        )

        self.assertEqual(result[0]["label"], "Claremont Avenue \u2193")

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


class EnrichedSegmentsCacheTests(unittest.TestCase):
    def test_loads_cached_enriched_segments_when_present(self) -> None:
        cached = gpd.GeoDataFrame(
            {"osm_name": ["Cached Road"], "chunk_state": ["flat or descent"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=4326,
        )
        source = gpd.GeoDataFrame(
            {"osm_name": ["Source Road"]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs=4326,
        )

        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "segments_enriched.geojson"
            write_geojson(cache_path, cached)

            with patch("gpx_analysis.site.data.enrich_segments_with_osm_edges") as osm_edges:
                result = load_or_build_enriched_segments(source, cache_path)

            osm_edges.assert_not_called()
            self.assertEqual(result.loc[0, "osm_name"], "Cached Road")
            self.assertNotIn("chunk_state", result.columns)

    def test_builds_and_writes_enriched_segments_when_cache_missing(self) -> None:
        source = gpd.GeoDataFrame(
            {"osm_name": ["Source Road"]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs=4326,
        )
        osm_result = source.assign(osm_highway="residential")
        mtc_result = osm_result.assign(mtc_pci_info="Good")

        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "segments_enriched.geojson"
            with (
                patch("gpx_analysis.site.data.enrich_segments_with_osm_edges", return_value=osm_result) as osm_edges,
                patch("gpx_analysis.site.data.enrich_segments_with_mtc_streets", return_value=mtc_result) as mtc_streets,
            ):
                result = load_or_build_enriched_segments(source, cache_path)

            osm_edges.assert_called_once()
            mtc_streets.assert_called_once()
            self.assertTrue(cache_path.exists())
            self.assertEqual(result.loc[0, "mtc_pci_info"], "Good")

    def test_strips_derived_columns_from_seeded_final_segments(self) -> None:
        segments = gpd.GeoDataFrame(
            {
                "osm_name": ["Cached Road"],
                "chunk_state": ["flat or descent"],
                "section_id": [1],
                "Ride Type": ["Steep"],
            },
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=4326,
        )

        result = strip_enriched_segment_derived_columns(segments)

        self.assertIn("osm_name", result.columns)
        self.assertNotIn("chunk_state", result.columns)
        self.assertNotIn("section_id", result.columns)
        self.assertNotIn("Ride Type", result.columns)


if __name__ == "__main__":
    unittest.main()
