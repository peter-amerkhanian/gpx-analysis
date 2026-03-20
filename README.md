# gpx-analysis

GPX route analysis project with a **Quarto-first publishing workflow**.

## Working Model

- `gpx_analysis/` holds reusable Python analysis code.
- `gpx_data/` holds curated GPX files.
- `scripts/build_quarto_data.py` exports route summaries and GeoJSON into `quarto/data/`.
- `quarto/` holds the source documents and dashboard pages.
- `docs/` is the rendered output target for Quarto.

## Current Focus

The project is being reoriented away from a custom web app and toward:

- Quarto documents for route notes and explainers
- Quarto dashboard pages for cross-route comparisons
- Python scripts for deterministic, precomputed analytics

## Setup

This project uses `uv` for dependency management.

```powershell
uv sync
```

## Build Quarto Data

```powershell
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\build_quarto_data.py
```

## Preview Quarto Site

```powershell
quarto preview quarto
```

## Project Layout

- `main.ipynb` - exploratory notebook
- `gpx_analysis/` - reusable analysis package
- `gpx_data/` - curated GPX inputs
- `scripts/build_quarto_data.py` - exports data artifacts for Quarto
- `quarto/` - Quarto source files
- `docs/` - rendered site output

## Immediate Next Steps

1. Tighten the route-by-route Quarto pages with real narrative.
2. Add route-specific charts such as elevation profile or hazard bands.
3. Decide whether GitHub Pages should publish rendered `docs/` or a Quarto render workflow.
