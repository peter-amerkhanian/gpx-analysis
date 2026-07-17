import geopandas as gpd

from .constants import LOCAL_OSM_NETWORK_TYPE, PROJECTED_CRS
from .osm import build_route_graph

def stop_signs_on_segments(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str = LOCAL_OSM_NETWORK_TYPE,
    corridor_m: float = 6.0,
    segment_buffer_m: float = 8.0,
    retain_all: bool = True
) -> gpd.GeoDataFrame:
    """Find stop/traffic light controls near route segments from the OpenStreetMap network."""
    projected_segments, nodes, _ = build_route_graph(gdf_segments, network_type, corridor_m, retain_all)

    if nodes.empty or "highway" not in nodes.columns:
        return nodes.iloc[0:0].to_crs(gdf_segments.crs)

    allowed = {"stop", "traffic_signals"}
    controls = nodes[nodes["highway"].apply(lambda value: value in allowed or (isinstance(value, list) and any(tag in allowed for tag in value)))].copy()

    if controls.empty:
        return controls.to_crs(gdf_segments.crs)

    projected_route_buffer = projected_segments.geometry.union_all().buffer(segment_buffer_m)
    projected_controls = controls.to_crs(PROJECTED_CRS)
    controls_on_route = projected_controls[projected_controls.geometry.intersects(projected_route_buffer)].copy()
    controls_on_route = controls_on_route.to_crs(gdf_segments.crs)

    if controls_on_route.empty:
        return controls_on_route

    controls_wgs84 = controls_on_route.to_crs(4326)
    controls_on_route["More Details"] = (
        '<a href="https://www.google.com/maps?q='
        + controls_wgs84.geometry.y.astype(str)
        + ","
        + controls_wgs84.geometry.x.astype(str)
        + '" target="_blank">📍 Open in Google Maps</a>'
    )
    return controls_on_route
