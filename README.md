# gpx-analysis

Early-stage Python project for analyzing GPX tracks and visualizing route behavior.

## Current Status

This repository is in scaffold/prototyping phase:

- Environment and dependencies are configured in `pyproject.toml`.
- Exploratory work is notebook-first (`main.ipynb`).
- Reusable GPX functions now live in a lightweight package (`gpx_analysis/`).
- Data directories exist for inputs and cached artifacts (`gpx_data/`, `cache/`).

## Planned Scope

The dependency stack suggests a workflow around:

- Parsing GPX files (`gpxpy`)
- Geospatial processing (`geopandas`, `shapely`, `osmnx`)
- Data analysis and modeling (`pandas`, `numpy`, `scikit-learn`)
- Mapping and visualization (`folium`, `matplotlib`)

## Project Layout

- `main.ipynb` - exploratory analysis notebook
- `gpx_analysis/io.py` - GPX parsing helpers
- `gpx_analysis/analytics.py` - step metrics and hazard detection
- `gpx_analysis/geo.py` - point-to-segment geospatial helpers
- `main.py` - simple script entry point using the package
- `hello_world.py` - simple smoke test script
- `gpx_data/` - GPX input files
- `cache/` - intermediate/generated artifacts

## Setup

This project uses `uv` for dependency management.

```powershell
uv sync
```

Run analysis using the default first GPX file found in `gpx_data/`:

```powershell
uv run python main.py
```

Run analysis on a specific file:

```powershell
uv run python main.py --path gpx_data/arlington.gpx
```

## Notebook/Script Usage

```python
from gpx_analysis import read_simple_gpx, analyze_steps, points_to_segments_lonlat

points = read_simple_gpx("gpx_data/arlington.gpx")
steps = analyze_steps(points, rolling_window=3)
```

## Next Steps

1. Define concrete analysis goals (distance, speed profile, elevation gain, clustering, etc.).
2. Continue extracting notebook-only logic into `gpx_analysis/` modules.
3. Add a richer CLI for selecting outputs and saving summary artifacts.
4. Add tests for GPX parsing and core metric calculations.
