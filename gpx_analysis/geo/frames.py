import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from .constants import LOCAL_OSM_CRS

def points_frame(points: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs=f"EPSG:{LOCAL_OSM_CRS}",
    )

def points_to_segments(gdf: gpd.GeoDataFrame, lon: str = "lon", lat: str = "lat", sort_col: str | None = None) -> gpd.GeoDataFrame:
    """Convert ordered point rows into consecutive line segments in lon/lat space."""
    frame = gdf.sort_values(sort_col) if sort_col else gdf.sort_index()
    x0, y0 = frame[lon].to_numpy()[:-1], frame[lat].to_numpy()[:-1]
    x1, y1 = frame[lon].to_numpy()[1:], frame[lat].to_numpy()[1:]
    segment_geometry = [LineString([(a, b), (c, d)]) for a, b, c, d in zip(x0, y0, x1, y1)]
    segments = gpd.GeoDataFrame(
        {"start_i": frame.index[:-1], "end_i": frame.index[1:]},
        geometry=segment_geometry,
        crs=gdf.crs,
    )
    segments = segments.merge(gdf.drop(columns="geometry"), left_on="end_i", right_on="step", how="left")
    return segments
