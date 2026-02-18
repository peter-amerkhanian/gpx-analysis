import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString


def points_to_segments_lonlat(gdf: gpd.GeoDataFrame, lon: str = "lon", lat: str = "lat", sort_col: str | None = None) -> gpd.GeoDataFrame:
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


def stop_signs_on_segments(
    gdf_segments: gpd.GeoDataFrame,
    network_type: str = "drive",
    corridor_m: float = 6.0,
    segment_buffer_m: float = 8.0,
    include_traffic_signals: bool = True,
    retain_all: bool = True
) -> gpd.GeoDataFrame:
    if gdf_segments.crs is None:
        raise ValueError("gdf_segments must have a CRS.")

    if gdf_segments.empty:
        return gpd.GeoDataFrame({"highway": []}, geometry=[], crs=gdf_segments.crs)

    segs_3857 = gdf_segments.to_crs(3857)
    route_poly_3857 = segs_3857.geometry.union_all().buffer(corridor_m)
    route_poly = gpd.GeoSeries([route_poly_3857], crs=3857).to_crs(gdf_segments.crs).iloc[0]

    graph = ox.graph_from_polygon(route_poly, network_type=network_type, simplify=False, retain_all=retain_all)
    nodes, _ = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    if "highway" not in nodes.columns:
        return nodes.iloc[0:0].to_crs(gdf_segments.crs)

    allowed = {"stop"}
    if include_traffic_signals:
        allowed.add("traffic_signals")

    controls = nodes[nodes["highway"].apply(lambda value: value in allowed or (isinstance(value, list) and any(tag in allowed for tag in value)))].copy()

    if controls.empty:
        return controls.to_crs(gdf_segments.crs)

    route_buffer = segs_3857.geometry.union_all().buffer(segment_buffer_m)
    controls_3857 = controls.to_crs(3857)
    controls_on_route = controls_3857[controls_3857.geometry.intersects(route_buffer)].copy()
    return controls_on_route.to_crs(gdf_segments.crs)
