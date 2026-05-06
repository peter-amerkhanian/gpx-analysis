from __future__ import annotations

import json
import shutil
from datetime import datetime, UTC

import geopandas as gpd
import numpy as np
import osmnx as ox
from pathlib import Path

from gpx_analysis.geo import (
    LOCAL_OSM_EDGES_PATH,
    LOCAL_OSM_EDGES_TILE_DIR,
    LOCAL_OSM_NETWORK_TYPE,
    LOCAL_OSM_NODES_PATH,
    LOCAL_OSM_NODES_TILE_DIR,
    LOCAL_OSM_TILE_SIZE_DEG,
    OSM_DATA_DIR,
)

BAY_AREA_COUNTIES = [
    "Alameda County, California, USA",
    "Contra Costa County, California, USA",
]
BOUNDARY_PATH = OSM_DATA_DIR / "sf_bay_area_boundary.geojson"
METADATA_PATH = OSM_DATA_DIR / "sf_bay_area_all_public.metadata.json"
PARQUET_ROW_GROUP_SIZE = 50_000


def _spatially_sort_nodes(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cluster nearby nodes together so parquet bbox reads skip more unrelated data."""
    return nodes.sort_values(["x", "y"]).reset_index(drop=True)


def _spatially_sort_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cluster nearby edges together before writing parquet row groups."""
    bounds = edges.geometry.bounds.rename(
        columns={"minx": "_minx", "miny": "_miny", "maxx": "_maxx", "maxy": "_maxy"}
    )
    sorted_edges = edges.join(bounds).sort_values(["_minx", "_miny", "_maxx", "_maxy"])
    return sorted_edges.drop(columns=["_minx", "_miny", "_maxx", "_maxy"]).reset_index(drop=True)


def _tile_ids_for_frame(frame: gpd.GeoDataFrame, tile_size: float = LOCAL_OSM_TILE_SIZE_DEG) -> gpd.GeoDataFrame:
    """Attach a coarse lon/lat tile id to each row based on geometry bounds."""
    bounds = frame.geometry.bounds
    x_tile = np.floor(bounds["minx"] / tile_size).astype(int)
    y_tile = np.floor(bounds["miny"] / tile_size).astype(int)
    return frame.assign(_tile_id="x" + x_tile.astype(str) + "_y" + y_tile.astype(str))


def _write_tiled_parquet(frame: gpd.GeoDataFrame, out_dir: Path) -> int:
    """Write one parquet file per spatial tile and return the tile count."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiled = _tile_ids_for_frame(frame)
    tile_count = 0
    for tile_id, tile_frame in tiled.groupby("_tile_id", sort=True):
        tile_out = tile_frame.drop(columns=["_tile_id"]).reset_index(drop=True)
        tile_out.to_parquet(
            out_dir / f"{tile_id}.parquet",
            index=False,
            write_covering_bbox=True,
        )
        tile_count += 1
    return tile_count


def main() -> None:
    OSM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Geocoding county boundaries...")
    boundary_gdf = ox.geocode_to_gdf(BAY_AREA_COUNTIES)
    boundary = boundary_gdf.geometry.union_all()
    boundary_crs = boundary_gdf.crs

    print(f"Downloading {LOCAL_OSM_NETWORK_TYPE} network for configured counties...")
    graph = ox.graph_from_polygon(
        boundary,
        network_type=LOCAL_OSM_NETWORK_TYPE,
        simplify=False,
        retain_all=True,
    )

    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes_out = _spatially_sort_nodes(nodes.reset_index())
    edges_out = _spatially_sort_edges(edges.reset_index())

    print(f"Saving nodes to {LOCAL_OSM_NODES_PATH}...")
    nodes_out.to_parquet(
        LOCAL_OSM_NODES_PATH,
        index=False,
        write_covering_bbox=True,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )
    print(f"Saving edges to {LOCAL_OSM_EDGES_PATH}...")
    edges_out.to_parquet(
        LOCAL_OSM_EDGES_PATH,
        index=False,
        write_covering_bbox=True,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )
    print(f"Saving tiled nodes to {LOCAL_OSM_NODES_TILE_DIR}...")
    node_tile_count = _write_tiled_parquet(nodes_out, LOCAL_OSM_NODES_TILE_DIR)
    print(f"Saving tiled edges to {LOCAL_OSM_EDGES_TILE_DIR}...")
    edge_tile_count = _write_tiled_parquet(edges_out, LOCAL_OSM_EDGES_TILE_DIR)

    boundary_out = gpd.GeoDataFrame(
        {"name": ["Configured Bay Area Counties"], "county_count": [len(BAY_AREA_COUNTIES)]},
        geometry=[boundary],
        crs=boundary_crs,
    )
    boundary_out.to_file(BOUNDARY_PATH, driver="GeoJSON")

    metadata = {
        "nodes_path": str(LOCAL_OSM_NODES_PATH),
        "edges_path": str(LOCAL_OSM_EDGES_PATH),
        "boundary_path": str(BOUNDARY_PATH),
        "network_type": LOCAL_OSM_NETWORK_TYPE,
        "counties": BAY_AREA_COUNTIES,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "created_utc": datetime.now(UTC).isoformat(),
        "osmnx_version": ox.__version__,
        "build_method": "osmnx Overpass download -> GeoParquet cache",
        "parquet_row_group_size": PARQUET_ROW_GROUP_SIZE,
        "tile_size_degrees": LOCAL_OSM_TILE_SIZE_DEG,
        "node_tile_count": node_tile_count,
        "edge_tile_count": edge_tile_count,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Finished.")
    print(f"Nodes: {metadata['node_count']:,}")
    print(f"Edges: {metadata['edge_count']:,}")
    print(f"Boundary: {BOUNDARY_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
