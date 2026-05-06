# gpx-analysis

This repo turns curated GPX rides into a published Quarto site of BART-accessible route notes, maps, and comparison tables.

The project has three layers:

- `gpx_analysis/` is the Python package that parses GPX tracks, computes route metrics, classifies ride hazards, enriches segments with local OpenStreetMap data, and prepares site-facing outputs.
- `routes.yml` is the manifest that defines which routes are included in the site.
- `quarto/` and `docs/` are the publishing layers: generated Quarto source lives in `quarto/`, and rendered site output is written to `docs/`.

## End-To-End Flow

The main build path is:

1. Add or update GPX files under `gpx_data/`.
2. Register each published route in `routes.yml` with a `slug`, `source`, and optional metadata like title, reverse direction, Strava link, and images.
3. Run `build_quarto_data.py`.
4. The build script loads the manifest, analyzes each GPX route, writes per-route data artifacts into `quarto/data/routes/<slug>/`, and regenerates the Quarto pages and navbar config.
5. Quarto renders the generated site from `quarto/` into `docs/`.

In code, `build_quarto_data.py` orchestrates the pipeline:

- `gpx_analysis.io.read_simple_gpx()` parses GPX track points into a flat DataFrame.
- `gpx_analysis.physics.compute_step_metrics()` computes per-step distance, elevation delta, grade, bearing, and turn signals.
- `gpx_analysis.analytics.detect_hazards()` labels each step with ride/hazard categories.
- `gpx_analysis.geo.points_to_segments()` converts points into route segments.
- `gpx_analysis.geo.enrich_segments_with_osm_edges()` matches route segments to a locally cached Bay Area OSM network and adds attributes like `osm_highway`, `osm_name`, and `road_type`.
- `gpx_analysis.geo.add_bart_station()` finds the nearest BART station to the start and end of each route.
- `gpx_analysis.reporting.aggregate_by_hazard()` summarizes route distance by hazard class.
- `gpx_analysis.viz.make_route_map()` builds a Folium map colored by hazard profile.
- `gpx_analysis.site.data.build_route()` writes the data bundle for each route.
- `gpx_analysis.site.render.*` regenerates route pages, the dashboard page, and `_quarto.yml`.

## Data Pipeline Outputs

For each route slug, the build writes a folder under `quarto/data/routes/<slug>/` containing:

- `summary.json`: route metadata, total distance, climbing, max elevation, and start/end coordinates and BART stations
- `hazards.json`: aggregate miles and percentages by hazard bucket
- `points.geojson`: per-point GPX data after analysis
- `segments.geojson`: per-segment geometry plus hazard and OSM enrichment columns
- `map.html`: embeddable Folium route map

It also writes:

- `quarto/data/routes.json`: site-wide route index consumed by the dashboard
- `quarto/routes/*.qmd`: generated per-route Quarto pages
- `quarto/index.qmd`: generated routes dashboard page
- `quarto/_quarto.yml`: generated site navbar config

When you run Quarto, those generated sources render into `docs/`, which is the publishable static site.

## Repo Structure

- `gpx_analysis/`: reusable route-analysis package
- `gpx_analysis/site/`: site-specific build helpers for route bundles and Quarto page generation
- `gpx_data/`: curated GPX source files
- `data/`: non-GPX reference data, including BART stations and the local OSM cache
- `quarto/`: generated Quarto source plus checked-in site assets
- `docs/`: rendered Quarto output
- `build_quarto_data.py`: main data/site generation script
- `download_bay_area_osm.py`: one-time or occasional builder for the local Bay Area OSM cache
- `routes.yml`: route manifest
- `sandbox.ipynb`: exploratory notebook for local analysis
- `DRAFT_coast_speed_openstax_explained.qmd`: standalone draft document, not part of the main route site

## Setup

This project uses `uv` and requires Python 3.12+.

```powershell
uv sync
```

## OSM Cache Prerequisite

Route enrichment depends on a local OpenStreetMap cache under `data/osm/`. If those parquet files are missing, route builds that need OSM enrichment will fail.

Build or refresh that cache with:

```powershell
uv run python download_bay_area_osm.py
```

That script downloads the configured Alameda and Contra Costa county network with OSMnx, writes GeoParquet node/edge files, and also creates tiled parquet shards so route-scoped reads stay fast.

## Build Commands

Regenerate all route data and Quarto source:

```powershell
uv run python build_quarto_data.py
```

Preview the site locally:

```powershell
quarto preview quarto
```

Render the site into `docs/` without starting preview:

```powershell
quarto render quarto
```

## Editing Guide

Edit these by hand:

- `routes.yml` to add, remove, rename, reverse, or annotate routes
- `gpx_data/*.gpx` to change the source rides
- `gpx_analysis/` when changing analysis logic, hazard rules, enrichment, or visual output
- checked-in assets under `quarto/images/` if you add route media

Treat these as generated artifacts:

- `quarto/routes/*.qmd`
- `quarto/index.qmd`
- `quarto/_quarto.yml`
- `quarto/data/**`
- `docs/**`

If you want a structural change to the route pages, dashboard, or navbar to persist, change the generators in `gpx_analysis/site/` or `build_quarto_data.py`, then rebuild.

## Route Manifest

`routes.yml` is the source of truth for published routes. Each route supports:

- `slug`: stable output id used for filenames and URLs
- `source`: GPX path relative to the repo root
- `title`: optional display title
- `reverse`: optional boolean to reverse point order before analysis
- `links.strava_effort`: optional external ride link shown on the route page
- `media.hero_image`: optional image path relative to `quarto/`
- `media.gallery`: optional list of image paths relative to `quarto/`

The loader validates duplicate slugs, missing GPX files, and missing media files before the build continues.

## Hazard Model

The current site build uses the default `simplified` hazard profile from `gpx_analysis.viz`.

Raw step-level classifications include things like:

- `steep_climb`
- `climb`
- `flat`
- `light_descent`
- `steep_descent`
- `ultra_steep_descent`
- `turn_on_descent`
- `turn_on_steep_descent`

For the published site, those are collapsed into:

- `steep_climb`
- `climb`
- `mellow`
- `descent`
- `danger_zone`

That simplified profile is used consistently for route summaries, tables, and Folium map colors.
