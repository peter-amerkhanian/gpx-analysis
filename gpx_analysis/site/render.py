from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from itables import to_html_datatable
import yaml

from .data import RouteConfig


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    return frame.to_html(
        index=False,
        border=0,
        classes=["table", "table-striped", "table-sm"],
        justify="left",
        escape=False,
        table_id=table_id,
    )


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
        '<div class="mobile-route-controls">',
        '<div class="mobile-route-control">',
        '<label for="mobile-route-sort">Sort routes</label>',
        '<select id="mobile-route-sort" class="form-select">',
        '<option value="miles_asc">Miles: shortest</option>',
        '<option value="miles_desc">Miles: longest</option>',
        '<option value="elev_asc">Elevation: lowest</option>',
        '<option value="elev_desc">Elevation: highest</option>',
        "</select>",
        "</div>",
        '<div class="mobile-route-control">',
        '<label for="mobile-route-bart">BART station</label>',
        '<select id="mobile-route-bart" class="form-select">',
        *bart_options,
        "</select>",
        "</div>",
        "</div>",
        '<div class="mobile-route-grid" id="mobile-route-grid">',
    ]
    sorted_routes = sorted(routes, key=lambda item: float(item["summary"]["distance_mi"]))
    for route in sorted_routes:
        steep_climbing = next((row["distance_mi"] for row in route["hazards"] if row["hazard"] == "steep_climb"), 0)
        dangerous_descent = next((row["distance_mi"] for row in route["hazards"] if row["hazard"] == "danger_zone"), 0)
        cards.extend(
            [
                (
                    f'<article class="mobile-route-card" '
                    f'data-bart="{route["summary"]["bart_station"]}" '
                    f'data-miles="{float(route["summary"]["distance_mi"]):.2f}" '
                    f'data-elevation="{float(route["summary"]["elevation_gain_ft"]):.2f}">'
                ),
                (
                    f'<p class="mobile-route-title"><a style="color:DodgerBlue;" href="{route["paths"]["page"].replace(".qmd", ".html")}">'
                    f'{route["title"]}</a></p>'
                ),
                (
                    '<div class="mobile-route-elevation" aria-hidden="true">'
                    f'{route.get("elevation_profile_svg", "")}'
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
                    f'<p><span class="mobile-route-label">Elevation Gain</span><br>'
                    f'{route["summary"]["elevation_gain_ft"]} ft</p>'
                ),
                # (
                #     f'<p><span class="mobile-route-label">Steep Climbing</span><br>'
                #     f'{steep_climbing} mi</p>'
                # ),
                (
                    f'<p><span class="mobile-route-label">Tech Descents</span><br>'
                    f'{dangerous_descent} mi</p>'
                ),
                "</div>",
                "</article>",
            ]
        )
    cards.extend(["</div>", "```"])
    return "\n".join(cards)


def route_page_content(
    route: RouteConfig,
    route_facts_heading: str,
    summary_table_html: str,
    hazards_table_html: str,
) -> str:
    hero_html = ""
    if route.media.hero_image:
        hero_html = f"""
## Photo
![{route.display_title}]({route.media.hero_image})
"""

    links_lines: list[str] = []
    if route.links.strava_effort:
        links_lines.append(
            '<a href="'
            f'{route.links.strava_effort}'
            '" target="_blank" rel="noopener noreferrer">'
            '<i class="fa-brands fa-strava"></i> Example Strava effort'
            "</a>"
        )
    links_html = ""
    if links_lines:
        links_html = "\n".join(links_lines) + "\n"

    gallery_html = ""
    if route.media.gallery:
        gallery_blocks = "\n".join(
            f"![{route.display_title}]({image_path})"
            for image_path in route.media.gallery
        )
        gallery_html = f"""
## Gallery
{gallery_blocks}
"""

    return f"""---
title: "{route.display_title}"
---
**{route_facts_heading}**  

{hero_html}

{links_html}

## Map
<iframe
  src="../data/routes/{route.slug}/map.html"
  style="width:100%; height:min(70vh, 560px); min-height:360px; border:none;"
  loading="lazy"
  allowfullscreen
></iframe>

## Data
{hazards_table_html}

{gallery_html}
"""


def write_route_page(
    route: RouteConfig,
    route_facts_heading: str,
    summary_table_html: str,
    hazards_table_html: str,
    route_pages_dir: Path,
) -> None:
    ensure_dir(route_pages_dir)
    (route_pages_dir / f"{route.slug}.qmd").write_text(
        route_page_content(route, route_facts_heading, summary_table_html, hazards_table_html),
        encoding="utf-8",
    )


def write_route_pages_index(route_pages_dir: Path, routes: list[RouteConfig]) -> None:
    ensure_dir(route_pages_dir)
    keep = {route.slug for route in routes}
    remove_stale_children(route_pages_dir, keep=keep, suffix=".qmd")


def write_dashboard_page(routes: list[dict[str, object]], output_path: Path, title: str) -> None:
    summary_table = pd.DataFrame(
        [
            {
                "Route": f'<a href="{route["paths"]["page"].replace(".qmd", ".html")}">{route["title"]}</a>',
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

    output_path.write_text(
        f"""---
title: "{title}"
format: dashboard
---

<style>
.desktop-only {{
  display: block;
}}

.mobile-only {{
  display: none;
}}

.mobile-route-grid {{
  display: grid;
  gap: 0.9rem;
}}

.mobile-route-controls {{
  display: grid;
  gap: 0.75rem;
  margin-bottom: 0.9rem;
}}

.mobile-route-control {{
  min-width: 0;
}}

.mobile-route-controls label {{
  display: block;
  font-weight: 600;
  margin-bottom: 0.35rem;
}}

.mobile-route-card {{
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 0.85rem;
  padding: 0.95rem 1rem 0.85rem;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.05);
}}

.mobile-route-title {{
  margin-top: 0;
  margin-bottom: 0.65rem;
  font-size: 1rem;
  line-height: 1.25;
  font-weight: 700;
}}

.mobile-route-title a {{
  color: inherit;
  text-decoration: none;
}}

.mobile-route-title a:hover {{
  text-decoration: underline;
}}

.mobile-route-elevation {{
  margin-bottom: 0.75rem;
}}

.mobile-route-elevation svg {{
  display: block;
  width: 100%;
  height: auto;
  max-height: 18vh;
}}

.mobile-route-metrics {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.65rem 0.9rem;
}}

.mobile-route-metrics p {{
  margin: 0;
  line-height: 1.25;
}}

.mobile-route-label {{
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: rgba(0, 0, 0, 0.62);
}}

@media (max-width: 900px) {{
  .desktop-only {{
    display: none;
  }}

  .mobile-only {{
    display: block;
  }}
}}
</style>

## Snapshot

### Route Summaries

::: {{.desktop-only}}
{interactive_table_html(summary_table.sort_values(by=["BART", "Miles"]))}
:::

::: {{.mobile-only}}
{mobile_summary_cards(routes)}
:::

```{{=html}}
<script>
window.addEventListener("load", function () {{
  const sortSelect = document.getElementById("mobile-route-sort");
  const bartSelect = document.getElementById("mobile-route-bart");
  const grid = document.getElementById("mobile-route-grid");
  if (!sortSelect || !bartSelect || !grid) {{
    return;
  }}

  const sorters = {{
    miles_asc: (a, b) => Number(a.dataset.miles) - Number(b.dataset.miles),
    miles_desc: (a, b) => Number(b.dataset.miles) - Number(a.dataset.miles),
    elev_asc: (a, b) => Number(a.dataset.elevation) - Number(b.dataset.elevation),
    elev_desc: (a, b) => Number(b.dataset.elevation) - Number(a.dataset.elevation)
  }};

  function applyMobileControls() {{
    const cards = Array.from(grid.querySelectorAll(".mobile-route-card"));
    const bart = bartSelect.value;
    cards.forEach((card) => {{
      const visible = !bart || card.dataset.bart === bart;
      card.style.display = visible ? "" : "none";
    }});
    const sorter = sorters[sortSelect.value] || sorters.miles_asc;
    cards.sort(sorter);
    cards.forEach((card) => grid.appendChild(card));
  }}

  sortSelect.addEventListener("change", applyMobileControls);
  bartSelect.addEventListener("change", applyMobileControls);
}});
</script>
```
""",
        encoding="utf-8",
    )


def write_quarto_config(routes: list[RouteConfig], quarto_config_path: Path) -> None:
    config = {
        "project": {
            "type": "website",
            "output-dir": "../docs",
            "resources": ["data/**", "images/**"],
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
