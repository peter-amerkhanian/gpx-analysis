from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, UTC

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
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

@dataclass(frozen=True)
class CountyBoundary:
    name: str
    osmid: str


BAY_AREA_COUNTIES = [
    CountyBoundary(name="Marin County, California, USA", osmid="R396461"),
    CountyBoundary(name="San Francisco County, California, USA", osmid="R111968"),
    CountyBoundary(name="Alameda County, California, USA", osmid="R396499"),
    CountyBoundary(name="Contra Costa County, California, USA", osmid="R396462"),
]
BOUNDARY_PATH = OSM_DATA_DIR / "sf_bay_area_boundary.geojson"
METADATA_PATH = OSM_DATA_DIR / "sf_bay_area_all_public.metadata.json"
PARQUET_ROW_GROUP_SIZE = 50_000


def _load_existing_metadata() -> dict[str, object]:
    """Return previously built cache metadata when available."""
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _county_names() -> list[str]:
    """Return configured county names in build order."""
    return [county.name for county in BAY_AREA_COUNTIES]


def _geocode_county_boundary(county: CountyBoundary) -> gpd.GeoDataFrame:
    """Resolve a county boundary via a stable OSM relation ID."""
    return ox.geocode_to_gdf(county.osmid, by_osmid=True)


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


def _combine_nodes(existing: gpd.GeoDataFrame, new: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge node frames and keep one row per OSM node id."""
    if existing.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return existing.reset_index(drop=True)
    combined = gpd.GeoDataFrame(
        pd.concat([existing, new], ignore_index=True),
        geometry="geometry",
        crs=existing.crs or new.crs,
    )
    return combined.drop_duplicates(subset=["osmid"], keep="first").reset_index(drop=True)


def _combine_edges(existing: gpd.GeoDataFrame, new: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge edge frames and keep one row per OSM edge key."""
    if existing.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return existing.reset_index(drop=True)
    combined = gpd.GeoDataFrame(
        pd.concat([existing, new], ignore_index=True),
        geometry="geometry",
        crs=existing.crs or new.crs,
    )
    return combined.drop_duplicates(subset=["u", "v", "key"], keep="first").reset_index(drop=True)


def _download_county_graph(county: CountyBoundary) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download one county network and return node/edge GeoDataFrames."""
    print(f"Geocoding boundary for {county.name}...")
    county_gdf = _geocode_county_boundary(county)
    county_boundary = county_gdf.geometry.union_all()
    print(f"Downloading {LOCAL_OSM_NETWORK_TYPE} network for {county.name}...")
    graph = ox.graph_from_polygon(
        county_boundary,
        network_type=LOCAL_OSM_NETWORK_TYPE,
        simplify=False,
        retain_all=True,
    )
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    return nodes.reset_index(), edges.reset_index()


def main() -> None:
    OSM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Geocoding county boundaries...")
    boundary_gdf = gpd.GeoDataFrame(
        pd.concat([_geocode_county_boundary(county) for county in BAY_AREA_COUNTIES], ignore_index=True),
        geometry="geometry",
    )
    boundary = boundary_gdf.geometry.union_all()
    boundary_crs = boundary_gdf.crs

    existing_metadata = _load_existing_metadata()
    existing_counties = set(existing_metadata.get("counties", []))
    configured_counties = set(_county_names())
    existing_cache_available = (
        LOCAL_OSM_NODES_PATH.exists()
        and LOCAL_OSM_EDGES_PATH.exists()
        and existing_counties.issubset(configured_counties)
    )
    if existing_cache_available:
        print(f"Loading existing nodes from {LOCAL_OSM_NODES_PATH}...")
        nodes_out = gpd.read_parquet(LOCAL_OSM_NODES_PATH)
        print(f"Loading existing edges from {LOCAL_OSM_EDGES_PATH}...")
        edges_out = gpd.read_parquet(LOCAL_OSM_EDGES_PATH)
    else:
        if existing_counties:
            print("Existing metadata does not match current configured counties; rebuilding configured cache.")
        nodes_out = gpd.GeoDataFrame(geometry=[], crs=boundary_crs)
        edges_out = gpd.GeoDataFrame(geometry=[], crs=boundary_crs)
        existing_counties = set()

    missing_counties = [county for county in BAY_AREA_COUNTIES if county.name not in existing_counties]
    if missing_counties:
        print(f"Downloading {len(missing_counties)} county network(s) one at a time...")
    else:
        print("Configured counties already exist in the local cache. Rewriting outputs from existing data.")

    for county in missing_counties:
        county_nodes, county_edges = _download_county_graph(county)
        nodes_out = _combine_nodes(nodes_out, county_nodes)
        edges_out = _combine_edges(edges_out, county_edges)
        print(
            f"Merged {county.name}: {len(county_nodes):,} nodes, {len(county_edges):,} edges. "
            f"Current cache: {len(nodes_out):,} nodes, {len(edges_out):,} edges."
        )

    nodes_out = _spatially_sort_nodes(nodes_out)
    edges_out = _spatially_sort_edges(edges_out)

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
        "counties": _county_names(),
        "node_count": int(len(nodes_out)),
        "edge_count": int(len(edges_out)),
        "created_utc": datetime.now(UTC).isoformat(),
        "osmnx_version": ox.__version__,
        "build_method": "osmnx county-by-county Overpass download -> merged GeoParquet cache",
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
