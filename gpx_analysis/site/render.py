from __future__ import annotations
import shutil
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from itables import to_html_datatable
import yaml
from .data import RouteConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


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


def html_table(frame: pd.DataFrame, table_id: str | None = None) -> str:
    table_html = frame.to_html(
        index=False,
        border=0,
        classes=["table", "table-striped", "table-sm"],
        justify="left",
        escape=False,
        table_id=table_id,
    )
    return f"```{{=html}}\n{table_html}\n```"


def interactive_table_html(frame: pd.DataFrame) -> str:
    """Render a DataFrame as an ITables table for Quarto HTML output."""
    table_html = to_html_datatable(
        frame,
        allow_html=True,
        columnControl=["order"],
        display_logo_when_loading=False,
        paging=False,
        scrollY="72vh",
        scrollCollapse=True,
        order=[[2, "desc"]],
        layout={"topStart": "search", "topEnd": None},
        showIndex=False,
        style={"width": "100%"},
    )
    return f"```{{=html}}\n{table_html}\n```"


def render_template(template_root: Path, template_name: str, **context: object) -> str:
    environment = Environment(
        loader=FileSystemLoader(str(template_root)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(template_name).render(**context).strip() + "\n"


def road_quality_color(score: float) -> str:
    """Return the summary-card background color for a road quality percentage."""
    if score < 10:
        return "#d73027"
    if score < 25:
        return "#fc8d59"
    if score < 40:
        return "#fee08b"
    if score < 55:
        return "#ffffbf"
    if score < 70:
        return "#d9ef8b"
    if score < 85:
        return "#91cf60"
    return "#1a9850"


def route_hazard_miles(route: dict[str, object], hazard: str) -> float:
    return float(
        next(
            (
                row["distance_mi"]
                for row in route["hazards"]
                if row["hazard"] == hazard
            ),
            0,
        )
    )


def summary_card(route: dict[str, object], path_prefix: str = "", title=True) -> list[str]:
    page_href = f'{path_prefix}{route["paths"]["page"].replace(".qmd", ".html")}'
    profile_src = f'{path_prefix}{route["paths"]["profile_svg"]}'
    title_html = str(route.get("title_html", route["title"]))
    road_quality_score = float(route["summary"]["road_quality_score"])
    road_quality_style = f"background-color:{road_quality_color(road_quality_score)};"
    steep_descent_miles = route_hazard_miles(route, "steep_descent")
    return [(
        f'<article class="mobile-route-card" '
        f'data-bart="{route["summary"]["bart_station"]}" '
        f'data-miles="{float(route["summary"]["distance_mi"]):.2f}" '
        f'data-elevation="{float(route["summary"]["elevation_gain_ft"]):.2f}" '
        f'data-time="{float(route["summary"].get("estimated_time_min", 0)):.0f}" '
        f'data-has-gravel="{str(float(route["summary"]["gravel_percent"]) > 10).lower()}">'
    ),
    (
        f'<p class="mobile-route-title"><a style="color:DodgerBlue;" href="{page_href}">'
        f'{title_html}</a></p>'
    ) if title else '',
    (
        '<div class="mobile-route-elevation" aria-hidden="true">'
        f'<img src="{profile_src}" alt="" loading="lazy">'
        "</div>"
    ),
    '<div class="mobile-route-metrics">',
    (
        f'<p><span class="mobile-route-label">BART</span><br>'
        f'{route["summary"]["bart_station"]}</p>'
    ),
    (
        f'<p><span class="mobile-route-label">Miles</span><br>'
        f'{route["summary"]["distance_mi"]}</p>'
    ),
    (
        f'<p><span class="mobile-route-label">Elev. Gain</span><br>'
        f'{route["summary"]["elevation_gain_ft"]} ft</p>'
    ),
    (
        f'<p><span class="mobile-route-label">Time</span><br>'
        f'~{route["summary"].get("estimated_time_display", "0:00")}</p>'
    ),
    (
        f'<p><span class="mobile-route-label">Steep Descent</span><br>'
        f'{steep_descent_miles:.2f} mi</p>'
    ),
    (
        f'<p><span class="mobile-route-label">Road Quality</span><br>'
        f'<span class="mobile-route-quality" style="{road_quality_style}">'
        f'{route["summary"]["road_quality_score"]}%</span></p>'
    ),
    "</div>",
    "</article>"]

def mobile_summary_cards(routes: list[dict[str, object]]) -> str:
    """Render compact route cards for small screens as raw HTML inside Quarto blocks."""
    bart_options = [
        '<option value="">All stations</option>',
        *[
            f'<option value="{station}">{station}</option>'
            for station in {str(route["summary"]["bart_station"]) for route in routes}
        ],
    ]
    cards: list[str] = [
        "```{=html}",
        '<div class="route-browser">',
        '<div class="mobile-route-controls">',
        '<div class="mobile-route-control">',
        '<label for="mobile-route-sort">Sort routes</label>',
        '<select id="mobile-route-sort" class="form-select">',
        '<option value="miles_asc">Miles: shortest</option>',
        '<option value="miles_desc">Miles: longest</option>',
        '<option value="elev_asc">Elevation: lowest</option>',
        '<option value="elev_desc">Elevation: highest</option>',
        '<option value="time_asc">Time: shortest</option>',
        '<option value="time_desc">Time: longest</option>',
        "</select>",
        "</div>",
        '<div class="mobile-route-control">',
        '<label for="mobile-route-bart">BART station</label>',
        '<select id="mobile-route-bart" class="form-select">',
        *bart_options,
        "</select>",
        "</div>",
        '<div class="mobile-route-control">',
        '<fieldset class="mobile-route-radio-control">',
        '<legend>Gravel</legend>',
        '<label><input type="radio" name="mobile-route-gravel" value="include" checked> W/ gravel</label>',
        '<label><input type="radio" name="mobile-route-gravel" value="exclude"> No gravel</label>',
        "</fieldset>",
        "</div>",
        "</div>",
        '<div class="mobile-route-grid" id="mobile-route-grid">',
    ]
    sorted_routes = sorted(routes, key=lambda item: float(item["summary"]["distance_mi"]))
    for route in sorted_routes:
        cards.extend(
            summary_card(route)
        )
    cards.extend(["</div>", "</div>", "```"])
    return "\n".join(cards)


def write_route_page(
    quarto_dir: Path,
    route: RouteConfig,
    route_bundle: dict[str, object],
    hazards_table_html: str,
    road_quality_table_html: str,
    climb_only_sections_table_html: str,
    chunk_sections_table_html: str,
    route_pages_dir: Path,
) -> None:
    ensure_dir(route_pages_dir)
    write_text(
        route_pages_dir / f"{route.slug}.qmd",
        render_template(
            quarto_dir,
            "templates/route-page.qmd.j2",
            title=str(route_bundle["title"]),
            hero_image=route.media.hero_image,
            strava_effort=route.links.strava_effort,
            summary_card_html="\n".join(summary_card(route_bundle, path_prefix="../", title=False)),
            map_src=f"../{route_bundle['paths']['map']}",
            road_quality_map_src=f"../{route_bundle['paths']['road_quality_map']}",
            chunk_map_src=f"../{route_bundle['paths']['chunk_map']}",
            hazards_table_html=hazards_table_html,
            road_quality_table_html=road_quality_table_html,
            climb_only_sections_table_html=climb_only_sections_table_html,
            chunk_sections_table_html=chunk_sections_table_html,
            gallery_images=route.media.gallery,
            gallery_title=str(route_bundle["title"]),
        ),
    )


def write_route_pages_index(route_pages_dir: Path, routes: list[RouteConfig]) -> None:
    ensure_dir(route_pages_dir)
    keep = {route.slug for route in routes}
    remove_stale_children(route_pages_dir, keep=keep, suffix=".qmd")


def write_dashboard_page(
    quarto_dir: Path,
    routes: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    summary_table = pd.DataFrame(
        [
            {
                "Route": (
                    f'<a href="{route["paths"]["page"].replace(".qmd", ".html")}">'
                    f'{route.get("title_html", route["title"])}</a>'
                ),
                "BART": route["summary"]["bart_station"],
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
                "Dangerous Descent Miles": next(
                    (
                        row["distance_mi"]
                        for row in route["hazards"]
                        if row["hazard"] == "danger_zone"
                    ),
                    0,
                ),
            }
            for route in routes
        ]
    ).sort_values(by="Miles")

    write_text(
        output_path,
        render_template(
            quarto_dir,
            "templates/dashboard.qmd.j2",
            title=title,
            desktop_table_html=interactive_table_html(summary_table.sort_values(by=["BART", "Miles"])),
            mobile_cards_html=mobile_summary_cards(routes),
        ),
    )


def write_quarto_config(routes: list[RouteConfig], quarto_config_path: Path) -> None:
    config = {
        "project": {
            "type": "website",
            "output-dir": "../docs",
            "resources": ["data/**", "images/**"],
            "render": ["index.qmd", "routes/*.qmd"],
        },
        "website": {
            "title": "🚇BART Rides🚲",
            "navbar": {
                "left": [
                    {"href": "index.qmd", "text": "Routes"},
                    {
                        "text": "Route Notes",
                        "menu": [
                            {
                                "href": f"routes/{route.slug}.qmd",
                                "text": route.display_title,
                            }
                            for route in routes
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
                "css": "styles.css",
                "include-in-header": {
                    "text": '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css">',
                },
            },
        },
        "execute": {
            "echo": False,
            "warning": False,
            "message": False,
            "freeze": "auto",
        },
    }
    quarto_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
