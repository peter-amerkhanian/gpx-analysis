# gpx-analysis

GPX route analysis project with a Python package for analytics, curated route inputs, and a Quarto website for publishing route dashboards.

## What This Repo Contains

- `gpx_analysis/` is the reusable Python package. It holds the analysis, geometry, IO, reporting, physics, and visualization code used to turn GPX tracks into route summaries and map-ready outputs.
- `gpx_data/` is the curated source data folder. It contains the GPX files that serve as the inputs for route analysis.
- `quarto/` is the Quarto website source. It contains the homepage, routes comparison dashboard, generated per-route pages, site config, and generated data artifacts under `quarto/data/`.
- `docs/` is the rendered website output directory. Quarto writes the built site here for publishing.

## Working Model

1. GPX files live in `gpx_data/`.
2. `build_quarto_data.py` reads the configured routes, runs the analysis package, and exports summaries, GeoJSON, maps, and generated Quarto pages.
3. Quarto reads the generated files in `quarto/` and renders the website into `docs/`.

## Setup

This project uses `uv` for dependency management.

```powershell
uv sync
```

## Build The Quarto Data

```powershell
uv run python build_quarto_data.py
```

## Preview The Website

```powershell
quarto preview quarto
```

## Project Layout

- `gpx_analysis/` - installable Python package with the reusable route-analysis logic.
- `gpx_data/` - GPX route inputs used to build the site data and route pages.
- `quarto/` - Quarto website source, generated route pages, and route data artifacts.
- `docs/` - rendered Quarto website output.
- `build_quarto_data.py` - project build script that analyzes routes and regenerates Quarto content.
- `main.py` - lightweight top-level script entry point for local experimentation.
- `main.ipynb` - exploratory notebook for analysis and prototyping.
- `DRAFT_coast_speed_openstax_explained.qmd` - draft Quarto document that is not part of the main site navigation.
- `pyproject.toml` - project metadata, dependencies, and packaging configuration.
- `uv.lock` - locked dependency versions for reproducible installs with `uv`.
- `.python-version` - local Python version hint for tooling.
- `.gitignore` - git ignore rules for generated files and local environment artifacts.
- `.venv/` - local virtual environment created by `uv sync`.
- `cache/` - local cache or scratch output generated during development.

## Notes

- The Quarto homepage and route dashboard pages are generated artifacts. If you want structural changes to those pages to persist, update `build_quarto_data.py` and then rebuild.
- The navbar and route pages are generated from the route list in `build_quarto_data.py`, so route additions or removals should start there.
