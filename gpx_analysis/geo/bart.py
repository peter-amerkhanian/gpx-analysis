from functools import lru_cache
from typing import cast

import fiona
import geopandas as gpd

from .constants import BART_KML_PATH, BART_STATION_LAYER, PROJECTED_CRS

def _load_bart_stations() -> gpd.GeoDataFrame:
    """Load BART station points from the repo KML in projected CRS."""
    fiona.drvsupport.supported_drivers.setdefault("KML", "rw")
    stations = gpd.read_file(BART_KML_PATH, driver="KML", layer=BART_STATION_LAYER)
    return stations.to_crs(PROJECTED_CRS)

def add_bart_station(gdf: gpd.GeoDataFrame, step: int = 0) -> str:
    """Return the nearest BART station name to the selected route step."""
    if gdf.crs is None:
        raise ValueError("gdf must have a CRS.")
    if gdf.empty:
        raise ValueError("gdf must not be empty.")
    if not (-len(gdf) <= step < len(gdf)):
        raise IndexError(f"step {step} is out of bounds for gdf with {len(gdf)} rows.")

    route_geometry = gdf.to_crs(PROJECTED_CRS).geometry.iloc[step]
    stations = _load_bart_stations()
    nearest_station = stations.loc[stations.geometry.distance(route_geometry).idxmin(), "Name"]
    return cast(str, nearest_station)
