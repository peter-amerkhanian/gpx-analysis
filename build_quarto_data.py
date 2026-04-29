from __future__ import annotations

from pathlib import Path

from gpx_analysis import DEFAULT_HAZARD_PROFILE
from gpx_analysis.site.data import build_route, ensure_dir, load_routes, write_json
from gpx_analysis.site.render import (
    html_table,
    remove_stale_children,
    write_dashboard_page,
    write_quarto_config,
    write_route_page,
    write_route_pages_index,
)


ROOT = Path(__file__).resolve().parent
QUARTO_DIR = ROOT / "quarto"
ROUTES_MANIFEST_PATH = ROOT / "routes.yml"
DATA_DIR = QUARTO_DIR / "data"
ROUTES_DIR = DATA_DIR / "routes"
ROUTE_PAGES_DIR = QUARTO_DIR / "routes"
QUARTO_CONFIG_PATH = QUARTO_DIR / "_quarto.yml"
INDEX_PAGE_PATH = QUARTO_DIR / "index.qmd"
DASHBOARD_PAGE_PATH = QUARTO_DIR / "routes-dashboard.qmd"
HAZARD_PROFILE = DEFAULT_HAZARD_PROFILE


def main() -> None:
    routes_config = load_routes(ROUTES_MANIFEST_PATH, ROOT)
    ensure_dir(ROUTES_DIR)
    remove_stale_children(ROUTES_DIR, keep={route.slug for route in routes_config})
    write_route_pages_index(ROUTE_PAGES_DIR, routes_config)

    routes: list[dict[str, object]] = []
    for route in routes_config:
        route_bundle, route_page_context = build_route(route, ROOT, ROUTES_DIR, HAZARD_PROFILE)
        write_route_page(
            route_page_context["route"],
            route_page_context["route_facts_heading"],
            html_table(route_page_context["summary_table"]),
            html_table(route_page_context["hazards_table"]),
            ROUTE_PAGES_DIR,
        )
        routes.append(route_bundle)

    write_json(DATA_DIR / "routes.json", {"routes": routes})
    write_dashboard_page(routes, INDEX_PAGE_PATH, "Routes")
    if DASHBOARD_PAGE_PATH.exists():
        DASHBOARD_PAGE_PATH.unlink()
    write_quarto_config(routes_config, QUARTO_CONFIG_PATH)
    print(f"Built {len(routes)} route bundles in {ROUTES_DIR}")


if __name__ == "__main__":
    main()
