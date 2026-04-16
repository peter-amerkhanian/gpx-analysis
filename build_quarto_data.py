from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from gpx_analysis import (
    aggregate_by_hazard,
    analyze_steps,
    DEFAULT_HAZARD_PROFILE,
    make_route_map,
    points_to_segments,
    prepare_segment_display_columns,
    read_simple_gpx,
)


ROOT = Path(__file__).resolve().parent
QUARTO_DIR = ROOT / "quarto"
DATA_DIR = QUARTO_DIR / "data"
ROUTES_DIR = DATA_DIR / "routes"
ROUTE_PAGES_DIR = QUARTO_DIR / "routes"
QUARTO_CONFIG_PATH = QUARTO_DIR / "_quarto.yml"
INDEX_PAGE_PATH = QUARTO_DIR / "index.qmd"
DASHBOARD_PAGE_PATH = QUARTO_DIR / "routes-dashboard.qmd"
HAZARD_PROFILE = DEFAULT_HAZARD_PROFILE


@dataclass(frozen=True)
class RouteConfig:
    slug: str
    title: str
    source: str
    reverse: bool = False


ROUTES = [
    RouteConfig(slug="arlington", title="Arlington Gravel Loop", source="gpx_data\\arlington_gravel_loop.gpx"),
    RouteConfig(slug="grizzly", title="Grizzly Peak", source="gpx_data\\classic_grizzly.gpx"),
    RouteConfig(slug="tunnel", title="Tunnel to Pinehurst", source="gpx_data\\tunnel_to_pinehurst.gpx"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def points_frame(points: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )


def json_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].astype("string").where(result[column].notna(), None)
    return result


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_geojson(path: Path, frame: gpd.GeoDataFrame) -> None:
    cleaned = json_ready_frame(frame)
    path.write_text(cleaned.to_json(), encoding="utf-8")


def html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(
        index=False,
        border=0,
        classes=["table", "table-striped", "table-sm"],
        justify="left",
        escape=False,
    )


def remove_stale_children(parent: Path, keep: set[str], suffix: str | None = None) -> None:
    if not parent.exists():
        return

    for child in parent.iterdir():
        if suffix is not None and child.suffix != suffix:
            continue
        if child.stem in keep or child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def route_page_content(
    route: RouteConfig,
    route_facts_heading: str,
    summary_table_html: str,
    hazards_table_html: str,
) -> str:
    return f"""---
title: "{route.title}"
---
**{route_facts_heading}**  

## Map
<iframe
  src="../data/routes/{route.slug}/map.html"
  style="width:100%; height:min(70vh, 560px); min-height:360px; border:none;"
  loading="lazy"
></iframe>

## Data
{hazards_table_html}
"""


def write_route_page(
    route: RouteConfig,
    route_facts_heading: str,
    summary_table_html: str,
    hazards_table_html: str,
) -> None:
    ensure_dir(ROUTE_PAGES_DIR)
    (ROUTE_PAGES_DIR / f"{route.slug}.qmd").write_text(
        route_page_content(route, route_facts_heading, summary_table_html, hazards_table_html),
        encoding="utf-8",
    )


def write_route_pages_index() -> None:
    ensure_dir(ROUTE_PAGES_DIR)
    keep = {route.slug for route in ROUTES}
    remove_stale_children(ROUTE_PAGES_DIR, keep=keep, suffix=".qmd")


def write_dashboard_page(routes: list[dict[str, object]], output_path: Path, title: str) -> None:
    summary_table = pd.DataFrame(
        [
            {
                "Route": f'<a href="{route["paths"]["page"].replace(".qmd", ".html")}">{route["title"]}</a>',
                "Miles": route["summary"]["distance_mi"],
                "Elevation Gain (ft)": route["summary"]["elevation_gain_ft"],
                "Steep Climbing Miles": next(
                    (
                        row["distance_mi"]
                        for row in route["hazards"]
                        if row["hazard"] == "steep_climb"
                    ),
                    0,
                ),
                "Max Elevation (ft)": route["summary"]["max_elevation_ft"]
            }
            for route in routes
        ]
    )

    hazard_rows: list[dict[str, object]] = []
    for route in routes:
        for row in route["hazards"]:
            if row["hazard"] == "TOTAL":
                continue
            hazard_rows.append(
                {
                    "Route": f'<a href="{route["paths"]["page"].replace(".qmd", ".html")}">{route["title"]}</a>',
                    "Hazard": row["hazard_label"],
                    "Miles": row["distance_mi"],
                    "Percent": row["percent"],
                }
            )
    output_path.write_text(
        f"""---
title: "{title}"
format: dashboard
---

## Snapshot

### Route Summaries

{html_table(summary_table)}

## Notes

### Editorial Direction

- Use this page for cross-route comparison.
- Use Route Notes for maps and more route-specific data.
""",
        encoding="utf-8",
    )


def write_quarto_config() -> None:
    config = {
        "project": {
            "type": "website",
            "output-dir": "../docs",
            "resources": ["data/**"],
        },
        "website": {
            "title": "GPX Analysis",
            "navbar": {
                "left": [
                    {"href": "index.qmd", "text": "Routes"},
                    {
                        "text": "Route Notes",
                        "menu": [
                            {
                                "href": f"routes/{route.slug}.qmd",
                                "text": route.title,
                            }
                            for route in ROUTES
                        ],
                    },
                ],
            },
        },
        "format": {
            "html": {
                "theme": "cosmo",
                "toc": True,
                "code-fold": False,
            },
        },
        "execute": {
            "echo": False,
            "warning": False,
            "message": False,
            "freeze": "auto",
        },
    }
    QUARTO_CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def compute_route_summary(points: pd.DataFrame, segments: gpd.GeoDataFrame) -> dict[str, object]:
    total_distance_m = float(points["step_dist_m"].sum())
    climbing = points["step_elevation_m"].diff().clip(lower=0)
    descending = points["step_elevation_m"].diff().clip(upper=0).abs()

    max_row = points.loc[points["elevation_m"].fillna(float("-inf")).idxmax()] if points["elevation_m"].notna().any() else None

    return {
        "point_count": int(len(points)),
        "segment_count": int(len(segments)),
        "distance_m": round(total_distance_m, 2),
        "distance_mi": round(total_distance_m / 1609.344, 2),
        "elevation_gain_m": round(float(climbing.sum()), 2),
        "elevation_gain_ft": round(float(climbing.sum()) * 3.28084, 2),
        "elevation_loss_m": round(float(descending.sum()), 2),
        "elevation_loss_ft": round(float(descending.sum()) * 3.28084, 2),
        "max_elevation_m": None if max_row is None else round(float(max_row["elevation_m"]), 2),
        "max_elevation_ft": None if max_row is None else round(float(max_row["elevation_f"]), 2),
        "start": {
            "lat": round(float(points.iloc[0]["lat"]), 6),
            "lon": round(float(points.iloc[0]["lon"]), 6),
        },
        "end": {
            "lat": round(float(points.iloc[-1]["lat"]), 6),
            "lon": round(float(points.iloc[-1]["lon"]), 6),
        },
    }


def build_route(route: RouteConfig) -> dict[str, object]:
    source_path = ROOT / route.source
    route_dir = ROUTES_DIR / route.slug
    ensure_dir(route_dir)

    points = read_simple_gpx(str(source_path), reverse=route.reverse)
    analyzed = analyze_steps(points, rolling_window=3)
    points_gdf = points_frame(analyzed)
    segments = points_to_segments(points_gdf)
    segments = prepare_segment_display_columns(segments, hazard_profile=HAZARD_PROFILE)

    summary = compute_route_summary(analyzed, segments)
    hazard_summary = aggregate_by_hazard(
        analyzed,
        column="step_dist_m",
        hazard_profile=HAZARD_PROFILE,
    ).rename(
        columns={"step_dist_m": "distance_m"}
    )
    hazard_summary["distance_mi"] = (hazard_summary["distance_m"] / 1609.344).round(2)
    ride_cols = ["Ride Type", "Turn", "Grade", "More Details"]
    route_map = make_route_map(
        segments,
        popup_cols=ride_cols,
        hazard_profile=HAZARD_PROFILE,
    )

    write_json(
        route_dir / "summary.json",
        {
            "route": asdict(route),
            "summary": summary,
        },
    )
    write_json(
        route_dir / "hazards.json",
        hazard_summary.to_dict(orient="records"),
    )
    write_geojson(route_dir / "points.geojson", points_gdf)
    write_geojson(route_dir / "segments.geojson", segments)
    route_map.save(str(route_dir / "map.html"))

    summary_table = pd.DataFrame(
        [
            ["Distance (mi)", summary["distance_mi"]],
            ["Elevation gain (ft)", summary["elevation_gain_ft"]],
            ["Elevation loss (ft)", summary["elevation_loss_ft"]],
            ["Max elevation (ft)", summary["max_elevation_ft"]],
        ],
        columns=["Metric", "Value"],
    )
    route_facts_heading = f'{summary["distance_mi"]:,.1f} miles<br> {summary["elevation_gain_ft"]:,.1f} ft elevation gain'
    hazards_table = hazard_summary[hazard_summary["hazard"] != "TOTAL"].copy()
    hazards_table = hazards_table.rename(
        columns={
            "hazard_label": "Hazard",
            "distance_mi": "Distance (mi)",
            "percent": "Percent",
        }
    )[["Hazard", "Distance (mi)", "Percent"]]
    write_route_page(
        route,
        route_facts_heading,
        html_table(summary_table),
        html_table(hazards_table),
    )

    return {
        "slug": route.slug,
        "title": route.title,
        "source": route.source,
        "reverse": route.reverse,
        "summary": summary,
        "paths": {
            "summary": f"data/routes/{route.slug}/summary.json",
            "hazards": f"data/routes/{route.slug}/hazards.json",
            "points": f"data/routes/{route.slug}/points.geojson",
            "segments": f"data/routes/{route.slug}/segments.geojson",
            "map": f"data/routes/{route.slug}/map.html",
            "page": f"routes/{route.slug}.qmd",
        },
        "hazards": hazard_summary.to_dict(orient="records"),
    }


def main() -> None:
    ensure_dir(ROUTES_DIR)
    remove_stale_children(ROUTES_DIR, keep={route.slug for route in ROUTES})
    write_route_pages_index()
    routes = [build_route(route) for route in ROUTES]
    write_json(DATA_DIR / "routes.json", {"routes": routes})
    write_dashboard_page(routes, INDEX_PAGE_PATH, "Routes")
    if DASHBOARD_PAGE_PATH.exists():
        DASHBOARD_PAGE_PATH.unlink()
    write_quarto_config()
    print(f"Built {len(routes)} route bundles in {ROUTES_DIR}")


if __name__ == "__main__":
    main()
