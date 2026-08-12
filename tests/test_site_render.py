import unittest

from gpx_analysis.site.render import mobile_summary_cards, summary_card


class SummaryCardTests(unittest.TestCase):
    def test_displays_priority_descent_chunk_miles(self) -> None:
        route = {
            "title": "Test Route",
            "paths": {"page": "routes/test.qmd", "profile_svg": "data/test.svg"},
            "summary": {
                "bart_station": "Test",
                "distance_mi": 20.0,
                "elevation_gain_ft": 2000,
                "estimated_time_min": 120,
                "estimated_time_display": "2:00",
                "road_quality_score": 75,
                "gravel_percent": 0,
                "priority_descent_mi": 2.4,
            },
            "hazards": [
                {"hazard": "steep_descent", "distance_mi": 9.9},
            ],
        }

        html = "".join(summary_card(route))

        self.assertIn("Danger Zone", html)
        self.assertIn("2.40 mi", html)
        self.assertIn('data-danger-zone="2.40"', html)
        self.assertNotIn("Steep Descent", html)
        self.assertNotIn("9.90 mi", html)

    def test_offers_danger_zone_sorting(self) -> None:
        route = {
            "title": "Test Route",
            "paths": {"page": "routes/test.qmd", "profile_svg": "data/test.svg"},
            "summary": {
                "bart_station": "Test",
                "distance_mi": 20.0,
                "elevation_gain_ft": 2000,
                "estimated_time_min": 120,
                "estimated_time_display": "2:00",
                "road_quality_score": 75,
                "gravel_percent": 0,
                "priority_descent_mi": 2.4,
            },
        }

        html = mobile_summary_cards([route])

        self.assertIn('value="danger_zone_asc"', html)
        self.assertIn('value="danger_zone_desc"', html)
