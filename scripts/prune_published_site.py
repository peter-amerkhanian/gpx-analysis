from __future__ import annotations

import shutil
from pathlib import Path


PUBLISHED_ROUTE_ARTIFACTS_TO_PRUNE = {
    "hazards.json",
    "points.geojson",
    "segments.geojson",
    "segments_enriched.geojson",
    "summary.json",
}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_routes_dir = repo_root / "quarto" / "data" / "routes"
    routes_dir = repo_root / "docs" / "data" / "routes"
    rendered_routes_dir = repo_root / "docs" / "routes"
    if not routes_dir.exists():
        print(f"No published route data to prune at {routes_dir}")
        return

    current_slugs = {
        path.name
        for path in source_routes_dir.iterdir()
        if path.is_dir()
    } if source_routes_dir.exists() else set()

    stale_count = 0
    if current_slugs:
        for route_dir in routes_dir.iterdir():
            if route_dir.is_dir() and route_dir.name not in current_slugs:
                shutil.rmtree(route_dir)
                stale_count += 1

        if rendered_routes_dir.exists():
            for route_page in rendered_routes_dir.glob("*.html"):
                if route_page.stem not in current_slugs:
                    route_page.unlink()
                    stale_count += 1

    removed_count = 0
    removed_bytes = 0
    for path in routes_dir.rglob("*"):
        if not path.is_file() or path.name not in PUBLISHED_ROUTE_ARTIFACTS_TO_PRUNE:
            continue
        removed_bytes += path.stat().st_size
        path.unlink()
        removed_count += 1

    removed_mb = removed_bytes / 1024 / 1024
    print(
        f"Pruned {removed_count} unpublished route artifact(s), "
        f"removed {stale_count} stale route output(s), saving {removed_mb:.1f} MB"
    )


if __name__ == "__main__":
    main()
