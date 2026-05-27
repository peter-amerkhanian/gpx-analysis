from __future__ import annotations

from typing import Literal, Mapping
from shapely.geometry import LineString, Point
import geopandas as gpd
import numpy as np
import pandas as pd
import folium
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen

ROAD_QUALITY_COLORS = {
    'Excellent': '#1a9850',
    'Very Good': '#91cf60',
    'Good': '#d9ef8b',
    'Fair': '#ffffbf',
    'At Risk': '#fee08b',
    'Poor': '#fc8d59',
    'Failed': '#d73027',
    'Gravel': "#712f00",
    'Cycleway': "#0078da",
}

SIMPLIFIED_ROAD_QUALITY_COLORS = {
    "Great": "#1a9850",
    "Good": "#d9ef8b",
    "Ok": "#fee08b",
    "Poor": "#d73027",
    "Roadway (Unknown)": "#8a8a8a",
    "Gravel": "#712f00",
    "Cycleway": "#0078da",
    "Cycleway (Unknown)": "#0078da",
}

DETAILED_HAZARD_COLORS = {
    "steep_climb": "#012C22",
    "climb": "#2D9966",
    "flat": "#31D492",
    "light_descent": "#fee08b",
    "steep_descent": "#f46d43",
    "ultra_steep_descent": "#9F0712",
    "turn_on_descent": "#4F39F6",
    "turn_on_steep_descent": "#8A0194",
}

SIMPLIFIED_HAZARD_COLORS = {
    "steep_climb": "#012C22",
    "climb": "#29865B",
    "mellow": "#31D492",
    "descent": "#f46d43",
    "danger_zone": "#9F0712",
}

CHUNK_STATE_COLORS = {
    "flat or descent": "#bdbdbd",
    "climb (easy)": "#9bd770",
    "climb (medium)": "#0C9000",
    "climb (hard)": "#052C01",
}

DEFAULT_HAZARD_COLORS = DETAILED_HAZARD_COLORS
DEFAULT_HAZARD_PROFILE = "simplified"

HAZARD_PROFILE_LABELS = {
    "detailed": {
        "steep_climb": "Steep Climb",
        "climb": "Climb",
        "flat": "Flat",
        "light_descent": "Light Descent",
        "steep_descent": "Steep Descent",
        "ultra_steep_descent": "Ultra Steep Descent",
        "turn_on_descent": "Turn On Descent",
        "turn_on_steep_descent": "Turn On Steep Descent",
    },
    "simplified": {
        "steep_climb": "Steep Climb",
        "climb": "Climb",
        "mellow": "Mellow",
        "descent": "Descent",
        "danger_zone": "Danger Zone",
    },
}

HAZARD_PROFILE_REMAPS = {
    "detailed": {
        "steep_climb": "steep_climb",
        "climb": "climb",
        "flat": "flat",
        "light_descent": "light_descent",
        "steep_descent": "steep_descent",
        "ultra_steep_descent": "ultra_steep_descent",
        "turn_on_descent": "turn_on_descent",
        "turn_on_steep_descent": "turn_on_steep_descent",
    },
    "simplified": {
        "steep_climb": "steep_climb",
        "climb": "climb",
        "flat": "mellow",
        "light_descent": "mellow",
        "steep_descent": "descent",
        "turn_on_descent": "descent",
        "ultra_steep_descent": "danger_zone",
        "turn_on_steep_descent": "danger_zone",
    },
}

HAZARD_PROFILE_COLORS = {
    "detailed": DETAILED_HAZARD_COLORS,
    "simplified": SIMPLIFIED_HAZARD_COLORS,
}

HazardProfileName = Literal["detailed", "simplified"]


def resolve_hazard_profile(
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
    hazard_colors: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    remap = dict(HAZARD_PROFILE_REMAPS[hazard_profile])
    colors = dict(HAZARD_PROFILE_COLORS[hazard_profile])
    if hazard_colors:
        colors.update(hazard_colors)
    labels = dict(HAZARD_PROFILE_LABELS[hazard_profile])
    return remap, colors, labels

def resolve_road_quality_profile(gdf_segments: gpd.GeoDataFrame):
    colors = ROAD_QUALITY_COLORS.copy()
    for val in gdf_segments['mtc_pci_info'].unique():
        if val not in ROAD_QUALITY_COLORS:
            colors[val] = "#8a8a8a"
    for pci, _ in ROAD_QUALITY_COLORS.items():
        if pci not in gdf_segments['mtc_pci_info'].unique():
            del colors[pci]
    return colors


def simplify_road_quality_category(value: object) -> str | object:
    """Collapse detailed PCI labels into simpler map categories."""
    if value is None or pd.isna(value):
        return value

    text = str(value)
    if text in {"Gravel", "Cycleway", "Cycleway (Unknown)"}:
        return text
    if text in {"Excellent", "Very Good"}:
        return "Great"
    if text in {"Good", "Fair"}:
        return "Good"
    if text == "At Risk":
        return "Ok"
    if text in {"Poor", "Failed"}:
        return "Poor"
    if text.endswith(" (Unknown)"):
        return "Roadway (Unknown)"
    return text


def resolve_simplified_road_quality_profile(gdf_segments: gpd.GeoDataFrame) -> dict[str, str]:
    """Return colors for simplified road-quality map categories present in the frame."""
    colors = SIMPLIFIED_ROAD_QUALITY_COLORS.copy()
    present = set(gdf_segments["road_quality_simple"].dropna().astype(str).unique())
    return {label: color for label, color in colors.items() if label in present}

def apply_hazard_profile(
    frame: pd.DataFrame,
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> pd.DataFrame:
    result = frame.copy()
    remap, _, labels = resolve_hazard_profile(hazard_profile=hazard_profile)
    result["hazard_raw"] = result["hazard"]
    result["hazard"] = result["hazard"].map(remap).fillna(result["hazard"])
    result["hazard_label"] = result["hazard"].map(labels).fillna(
        result["hazard"].str.replace("_", " ", regex=False).str.title()
    )
    return result


def google_maps_link(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Build Google Maps query URLs for latitude/longitude series pairs."""
    gmaps_url = (
        "https://www.google.com/maps?q=" + lat.astype(str) + "," + lon.astype(str)
    )
    gmaps_link = (
        '<a href="' + gmaps_url + '" target="_blank">📍 Open in Google Maps</a>'
    )
    return gmaps_link


def prepare_segment_display_columns(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> gpd.GeoDataFrame:
    """Return a copy with presentation columns used by folium visualizations."""
    frame = apply_hazard_profile(gdf_segments, hazard_profile=hazard_profile)
    _, colors, _ = resolve_hazard_profile(
        hazard_profile=hazard_profile,
        hazard_colors=hazard_colors,
    )
    frame["Segment"] = frame["step"].astype("Int64").astype(str)
    frame["More Details"] = google_maps_link(frame['lat'], frame['lon'])
    frame["Turn"] = (
        frame["step_turn"].round(2).astype(str) + "°"
    )
    frame["Grade"] = (
        frame["step_grade"].multiply(100).round(2).astype(str) + "%"
    )
    frame["Ride Type"] = frame["hazard_label"]
    frame["_display_color"] = frame["hazard"].map(colors).fillna("#8a8a8a")
    return frame

def prepare_osm_columns(gdf_segments_enriched: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = gdf_segments_enriched.copy()
    frame["Road Name"] = (
    frame["osm_name"].fillna("Unknown Road")
    )
    frame["Road Type"] = (
    frame["osm_highway"].str.title().fillna('Unknown type') + " " +
    frame["osm_lanes"].fillna('unknown') +
    " lane road"
    )
    frame["Speed Limit"] = (
    frame["osm_maxspeed"].fillna("Unknown")
    )
    return frame


def _select_present_columns(
    frame: gpd.GeoDataFrame,
    columns: list[str],
) -> gpd.GeoDataFrame:
    """Return a frame with just the requested columns that are present."""
    keep = [column for column in columns if column in frame.columns]
    return frame.loc[:, keep].copy()


def _marker_point_and_normal(
    segment: object,
    fallback_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a segment start point and a unit normal for label offsets."""
    coords = np.asarray(segment.coords, dtype=float)
    start = coords[0]
    end = coords[-1]
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm == 0:
        return start, np.array([0.0, float(fallback_sign)])
    unit_direction = direction / norm
    normal = np.array([-unit_direction[1], unit_direction[0]])
    return start, normal


def _resolve_number_marker_locations(
    frame: gpd.GeoDataFrame,
    marker_indexes: list[int],
    min_spacing_m: float = 520.0,
    base_offset_m: float = 18.0,
    offset_step_m: float = 28.0,
    max_attempts: int = 6,
) -> list[list[float]]:
    """Place number markers near their segments while avoiding overlap."""
    projected = frame[["geometry"]].to_crs(3857)
    placed_points: list[np.ndarray] = []
    locations: list[list[float]] = []

    for marker_order, marker_index in enumerate(marker_indexes):
        segment = projected.iloc[marker_index].geometry
        base_point, normal = _marker_point_and_normal(
            segment,
            fallback_sign=1 if marker_order % 2 == 0 else -1,
        )

        candidate = base_point
        for attempt in range(max_attempts):
            offset_scale = base_offset_m + (attempt * offset_step_m)
            if marker_order % 2 == 1:
                offset_scale *= -1
            candidate = base_point + (normal * offset_scale)
            if all(np.linalg.norm(candidate - placed) >= min_spacing_m for placed in placed_points):
                break

        placed_points.append(candidate)
        point_wgs84 = (
            gpd.GeoSeries([Point(candidate[0], candidate[1])], crs=3857)
            .to_crs(4326)
            .iloc[0]
        )
        locations.append([point_wgs84.y, point_wgs84.x])

    return locations


def _number_marker_count(frame: gpd.GeoDataFrame) -> int:
    """Return the number of numbered route markers based on route length."""
    distance_m = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0).sum()
    distance_mi = distance_m / 1609.344
    return max(3, 3 + int(distance_mi // 15))


def _number_marker_indexes(frame: gpd.GeoDataFrame) -> list[int]:
    """Spread numbered markers across the route, excluding the start marker."""
    marker_count = _number_marker_count(frame)
    last_index = len(frame) - 1
    if last_index <= 0:
        return [0] * marker_count

    fractions = np.linspace(
        1 / (marker_count + 10),
        marker_count / (marker_count + 1),
        marker_count,
    )
    indexes = [min(max(1, int(round(last_index * fraction))), last_index) for fraction in fractions]
    return indexes


def _route_chevron_dimensions(frame: gpd.GeoDataFrame) -> tuple[float, float]:
    """Scale chevron geometry by overall route length, anchored to a ~30 mile route."""
    distance_m = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0.0).sum()
    distance_mi = distance_m / 1609.344
    scale = float(np.clip(distance_mi / 30.0, 0.55, 1.15))
    return 350.0 * scale, 250.0 * scale


def _chevron_paths_for_segment(
    segment: object,
    chevron_length_m: float,
    chevron_half_width_m: float,
) -> list[list[list[float]]]:
    """Return two WGS84 line paths forming a centered chevron for the segment."""
    coords = np.asarray(segment.coords, dtype=float)
    if len(coords) < 2:
        return []

    start = coords[0]
    end = coords[-1]
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm == 0:
        return []

    unit_direction = direction / norm
    unit_normal = np.array([-unit_direction[1], unit_direction[0]])
    midpoint = (start + end) / 2.0
    tip = midpoint + (unit_direction * (chevron_length_m / 2.0))
    tail_center = midpoint - (unit_direction * (chevron_length_m / 2.0))
    left = tail_center + (unit_normal * chevron_half_width_m)
    right = tail_center - (unit_normal * chevron_half_width_m)

    chevron_lines = gpd.GeoSeries(
        [
            LineString([tuple(left), tuple(tip)]),
            LineString([tuple(right), tuple(tip)]),
        ],
        crs=3857,
    ).to_crs(4326)
    return [[[lat, lon] for lon, lat in line.coords] for line in chevron_lines]


def _chevron_midpoint(segment: object) -> np.ndarray | None:
    """Return the projected midpoint of a segment for chevron overlap checks."""
    coords = np.asarray(segment.coords, dtype=float)
    if len(coords) < 2:
        return None
    start = coords[0]
    end = coords[-1]
    if np.allclose(end, start):
        return None
    return (start + end) / 2.0


def _segment_indexes_with_route_overlap(
    projected: gpd.GeoDataFrame,
    overlap_proximity_m: float = 20.0,
    min_shared_length_m: float = 25.0,
) -> set[int]:
    """Return segment indexes that truly share route geometry with non-adjacent segments."""
    if projected.empty:
        return set()

    buffered = projected[["geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(overlap_proximity_m)
    overlap_indexes: set[int] = set()
    joined = gpd.sjoin(
        projected[["geometry"]],
        buffered[["geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
    )
    joined = joined[
        (joined.index != joined["index_right"])
        & ((joined.index - joined["index_right"]).abs() > 1)
    ]
    if joined.empty:
        return overlap_indexes

    geometries = projected.geometry
    for segment_index, match_index in joined[["index_right"]].itertuples(index=True, name=None):
        left = geometries.loc[segment_index]
        right = geometries.loc[match_index]
        shared_length_m = left.intersection(right).length
        if shared_length_m >= min_shared_length_m:
            overlap_indexes.add(int(segment_index))
            overlap_indexes.add(int(match_index))
    return overlap_indexes


def _frames_share_route_overlap(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    overlap_proximity_m: float = 20.0,
    min_shared_length_m: float = 25.0,
    column: str | None = None,
    ignore_value: str | None = None,
) -> bool:
    """Return True when two route halves share meaningful geometry that merits a pass control."""
    if left.empty or right.empty:
        return False

    right_buffered = right[["geometry"]].copy()
    right_buffered["geometry"] = right_buffered.geometry.buffer(overlap_proximity_m)
    joined = gpd.sjoin(
        left[["geometry"]],
        right_buffered[["geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
    )
    if joined.empty:
        return False

    left_geometries = left.geometry
    right_geometries = right.geometry
    has_meaningful_overlap = False
    for left_index, right_index in joined[["index_right"]].itertuples(index=True, name=None):
        shared_length_m = left_geometries.loc[left_index].intersection(right_geometries.loc[right_index]).length
        if shared_length_m >= min_shared_length_m:
            has_meaningful_overlap = True
            if column is None or ignore_value is None:
                return True

            left_value = left.loc[left_index, column] if column in left.columns else None
            right_value = right.loc[right_index, column] if column in right.columns else None
            if not (left_value == ignore_value and right_value == ignore_value):
                return True

    return False if has_meaningful_overlap else False


def _chevron_marker_segments(
    frame: gpd.GeoDataFrame,
    spacing_fraction: int = 9,
    min_segment_length_m: float = 100.0,
) -> list[list[list[list[float]]]]:
    """Return route-spaced chevron paths for sufficiently long, non-overlapping segments."""
    if frame.empty:
        return []

    projected = frame[["geometry"]].to_crs(3857).copy()
    projected["segment_length_m"] = projected.geometry.length
    overlapping_indexes = _segment_indexes_with_route_overlap(projected)
    eligible = projected[projected["segment_length_m"] >= min_segment_length_m].copy()
    if overlapping_indexes:
        eligible = eligible[~eligible.index.isin(overlapping_indexes)]
    if eligible.empty:
        return []

    chevron_length_m, chevron_half_width_m = _route_chevron_dimensions(frame)
    min_chevron_spacing_m = chevron_length_m * 0.9
    target_count = max(1, spacing_fraction - 1)
    route_distances = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0.0)
    cumulative_end_m = route_distances.cumsum()
    total_distance_m = float(cumulative_end_m.iloc[-1]) if not cumulative_end_m.empty else 0.0
    if total_distance_m <= 0:
        target_indexes = list(eligible.index[:target_count])
    else:
        target_positions = np.linspace(
            total_distance_m / spacing_fraction,
            total_distance_m * (spacing_fraction - 1) / spacing_fraction,
            target_count,
        )
        eligible_end_m = cumulative_end_m.loc[eligible.index]
        used_indexes: set[int] = set()
        target_indexes: list[int] = []
        for target_position in target_positions:
            ranked_indexes = (
                (eligible_end_m - target_position)
                .abs()
                .sort_values()
                .index
            )
            for segment_index in ranked_indexes:
                if int(segment_index) not in used_indexes:
                    used_indexes.add(int(segment_index))
                    target_indexes.append(int(segment_index))
                    break

    chevrons: list[list[list[list[float]]]] = []
    accepted_midpoints: list[np.ndarray] = []
    for segment_index in target_indexes:
        segment = projected.loc[segment_index, "geometry"]
        midpoint = _chevron_midpoint(segment)
        if midpoint is None:
            continue
        if any(np.linalg.norm(midpoint - accepted) < min_chevron_spacing_m for accepted in accepted_midpoints):
            continue
        chevron_paths = _chevron_paths_for_segment(
            segment,
            chevron_length_m=chevron_length_m,
            chevron_half_width_m=chevron_half_width_m,
        )
        if chevron_paths:
            accepted_midpoints.append(midpoint)
            chevrons.append(chevron_paths)
    return chevrons


def _split_outbound_return(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split ordered route segments into outbound and return halves by cumulative distance."""
    if frame.empty:
        return frame.copy(), frame.iloc[0:0].copy()

    route_distances = pd.to_numeric(frame.get("step_dist_m"), errors="coerce").fillna(0.0)
    cumulative_end_m = route_distances.cumsum()
    total_distance_m = float(cumulative_end_m.iloc[-1]) if not cumulative_end_m.empty else 0.0
    split_distance_m = total_distance_m / 2.0
    outbound_mask = cumulative_end_m <= split_distance_m
    if outbound_mask.sum() == 0:
        outbound_mask.iloc[0] = True
    if outbound_mask.sum() == len(frame):
        outbound_mask.iloc[-1] = False
    outbound = frame.loc[outbound_mask].copy()
    returning = frame.loc[~outbound_mask].copy()
    return outbound, returning


def _remove_geojson_layers(m: folium.Map) -> None:
    """Remove existing GeoJson layers so split overlays can replace the default route layer."""
    geojson_child_keys = [
        key for key, child in m._children.items()
        if isinstance(child, folium.features.GeoJson)
    ]
    for key in geojson_child_keys:
        m._children.pop(key, None)


def _add_whole_route_backdrop(m: folium.Map, frame: gpd.GeoDataFrame) -> None:
    """Add an always-on light-gray route backdrop behind directional overlays."""
    folium.GeoJson(
        data=frame[["geometry"]].to_json(),
        name="Whole Route",
        control=False,
        style_function=lambda _: {
            "color": "#c7c7c7",
            "weight": 6,
            "opacity": 0.45,
        },
    ).add_to(m)


def _add_direction_radio_control(
    m: folium.Map,
    outbound_layer: folium.FeatureGroup,
    return_layer: folium.FeatureGroup,
) -> None:
    """Add a custom radio control so exactly one highlighted direction stays visible."""
    map_name = m.get_name()
    outbound_name = outbound_layer.get_name()
    return_name = return_layer.get_name()
    template = Template(
        f"""
        {{% macro script(this, kwargs) %}}
        (function() {{
            var map = {map_name};
            var outboundLayer = {outbound_name};
            var returnLayer = {return_name};
            var control = L.control({{position: 'topright'}});

            control.onAdd = function() {{
                var div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                div.style.background = 'white';
                div.style.padding = '8px 10px';
                div.style.fontSize = '13px';
                div.style.lineHeight = '1.4';
                div.innerHTML = `
                    <div style="font-weight:600; margin-bottom:4px;">Route Pass</div>
                    <label style="display:block; cursor:pointer;">
                        <input type="radio" name="route-pass-toggle" value="outbound" checked> Outbound
                    </label>
                    <label style="display:block; cursor:pointer;">
                        <input type="radio" name="route-pass-toggle" value="return"> Return
                    </label>
                `;
                L.DomEvent.disableClickPropagation(div);
                return div;
            }};

            control.addTo(map);

            function setPass(passName) {{
                if (passName === 'return') {{
                    if (map.hasLayer(outboundLayer)) map.removeLayer(outboundLayer);
                    if (!map.hasLayer(returnLayer)) map.addLayer(returnLayer);
                }} else {{
                    if (map.hasLayer(returnLayer)) map.removeLayer(returnLayer);
                    if (!map.hasLayer(outboundLayer)) map.addLayer(outboundLayer);
                }}
            }}

            setPass('outbound');
            var radios = document.getElementsByName('route-pass-toggle');
            for (var i = 0; i < radios.length; i++) {{
                radios[i].addEventListener('change', function(evt) {{
                    setPass(evt.target.value);
                }});
            }}
        }})();
        {{% endmacro %}}
        """
    )
    control = MacroElement()
    control._template = template
    m.add_child(control)


def _ensure_map_pane(m: folium.Map, pane_name: str, z_index: int) -> None:
    """Create a custom Leaflet pane with the requested z-index when needed."""
    map_name = m.get_name()
    template = Template(
        f"""
        {{% macro script(this, kwargs) %}}
        (function() {{
            var map = {map_name};
            if (!map.getPane('{pane_name}')) {{
                map.createPane('{pane_name}');
            }}
            map.getPane('{pane_name}').style.zIndex = {z_index};
        }})();
        {{% endmacro %}}
        """
    )
    pane = MacroElement()
    pane._template = template
    m.add_child(pane)


def _disable_tooltips_on_touch(m: folium.Map) -> None:
    """Hide Leaflet tooltips on touch devices while preserving desktop hover."""
    style = Template(
        """
        {% macro header(this, kwargs) %}
        <style>
        @media (hover: none), (pointer: coarse) {
            .leaflet-tooltip {
                display: none !important;
            }
        }
        </style> 
        {% endmacro %}
        """
    )
    style_element = MacroElement()
    style_element._template = style
    m.add_child(style_element)


def _route_is_close_ended(frame: gpd.GeoDataFrame, close_distance_m: float = 250.0) -> bool:
    """Return True when the route end is close enough to the start to treat it as a loop."""
    if frame.empty:
        return False

    start = Point(frame.iloc[0].geometry.coords[0])
    end = Point(frame.iloc[-1].geometry.coords[-1])
    endpoints = gpd.GeoSeries([start, end], crs=frame.crs)
    try:
        projected = endpoints.to_crs(3857)
    except ValueError:
        return False
    return float(projected.iloc[0].distance(projected.iloc[1])) <= close_distance_m


def _add_direction_layers(
    m: folium.Map,
    frame: gpd.GeoDataFrame,
    column: str,
    tooltip_fields: list[str] | None,
    popup_cols: list[str] | None,
    categories: list[str] | None,
    cmap: list[str] | None,
    style_kwds: dict[str, object] | None,
    escape: bool,
) -> folium.Map:
    """Replace the default route layer with outbound/return overlays when the route overlaps itself."""
    projected = frame[["geometry"]].to_crs(3857)
    outbound, returning = _split_outbound_return(frame)
    if outbound.empty or returning.empty:
        return m
    projected_outbound = outbound[["geometry"]].to_crs(3857)
    projected_returning = returning[["geometry"]].to_crs(3857)
    overlap_column = column if column in frame.columns else None
    ignore_value = "mellow" if column == "hazard" else None
    if overlap_column is not None:
        projected_outbound[overlap_column] = outbound[overlap_column]
        projected_returning[overlap_column] = returning[overlap_column]
    if not _frames_share_route_overlap(
        projected_outbound,
        projected_returning,
        column=overlap_column,
        ignore_value=ignore_value,
    ):
        return m

    _remove_geojson_layers(m)
    _add_whole_route_backdrop(m, frame)
    explore_kwargs = {
        "column": column,
        "tooltip": tooltip_fields,
        "popup": popup_cols,
        "categorical": True,
        "legend": False,
        "style_kwds": style_kwds or {"weight": 4},
        "escape": escape,
    }
    if categories is not None:
        explore_kwargs["categories"] = categories
    if cmap is not None:
        explore_kwargs["cmap"] = cmap

    outbound_layer = folium.FeatureGroup(name="Outbound", overlay=True, control=False, show=True)
    outbound_layer.add_to(m)
    outbound.explore(
        m=outbound_layer,
        **explore_kwargs,
    )
    return_layer = folium.FeatureGroup(name="Return", overlay=True, control=False, show=False)
    return_layer.add_to(m)
    returning.explore(
        m=return_layer,
        **explore_kwargs,
    )
    _add_direction_radio_control(m, outbound_layer, return_layer)
    return m


def add_map_elements(
    m: folium.Map,
    frame: gpd.GeoDataFrame,
    show_numbers: bool = True,
    show_route_pass_control: bool = False,
    layer_column: str | None = None,
    tooltip_fields: list[str] | None = None,
    popup_cols: list[str] | None = None,
    categories: list[str] | None = None,
    cmap: list[str] | None = None,
    style_kwds: dict[str, object] | None = None,
    escape: bool = False,
) -> None:
    if show_route_pass_control and layer_column:
        m = _add_direction_layers(
            m,
            frame,
            column=layer_column,
            tooltip_fields=tooltip_fields,
            popup_cols=popup_cols,
            categories=categories,
            cmap=cmap,
            style_kwds=style_kwds,
            escape=escape,
        )
    # Fullscreen control
    Fullscreen(
        position="topleft",
        title="Fullscreen",
        title_cancel="Exit fullscreen",
        force_separate_button=True,
    ).add_to(m)
    # Start / end
    start = frame.iloc[0].geometry.coords[0]
    folium.Marker(
        location=[start[1], start[0]],  # folium uses [lat, lon]
        icon=folium.Icon(color="green", icon="arrow-right", prefix="fa"),
    ).add_to(m)
    if not _route_is_close_ended(frame):
        end = frame.iloc[-1].geometry.coords[-1]
        folium.Marker(
            location=[end[1], end[0]],  # folium uses [lat, lon]
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        ).add_to(m)
    # Numbers
    if show_numbers:
        marker_indexes = _number_marker_indexes(frame)
        marker_locations = _resolve_number_marker_locations(frame, marker_indexes)
        number_style = "font-size:41px; font-weight:700; color:#C96A1B; opacity:0.30;"
        for marker_number, marker_location in enumerate(marker_locations, start=1):
            folium.Marker(
                location=marker_location,
                icon=folium.DivIcon(
                    html=(
                        f'<div style="{number_style}">'
                        f"{marker_number}"
                        '</div>'
                    )
                )
            ).add_to(m)
    # Direction chevrons
    _ensure_map_pane(m, pane_name="route-chevrons", z_index=650)
    for chevron_paths in _chevron_marker_segments(frame):
        for chevron_path in chevron_paths:
            folium.PolyLine(
                locations=chevron_path,
                color="#111111",
                weight=2,
                opacity=0.95,
                line_cap="square",
                line_join="round",
                pane="route-chevrons",
            ).add_to(m)
    _disable_tooltips_on_touch(m)
    return m

def make_route_map(
    gdf_segments: gpd.GeoDataFrame,
    hazard_colors: Mapping[str, str] | None = None,
    popup_cols: list[str] | None = ["Ride Type", "Turn", "Grade", "More Details"],
    tooltip_fields: list[str] | None = ['Segment', 'Ride Type'],
    tiles: str = "CartoDB Voyager",
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> folium.Map:
    """Build a Folium map with hazard-colored segments and route popups/tooltips."""
    frame = prepare_segment_display_columns(
        gdf_segments,
        hazard_colors=hazard_colors,
        hazard_profile=hazard_profile,
    )
    frame = _select_present_columns(
        frame,
        [
            "geometry",
            "step_dist_m",
            "Segment",
            "Ride Type",
            "Turn",
            "Grade",
            "More Details",
            "hazard",
            "_display_color",
        ],
    )
    _, colors, _ = resolve_hazard_profile(
        hazard_profile=hazard_profile,
        hazard_colors=hazard_colors,
    )
    m = frame.explore(
    column='hazard',
    tooltip=tooltip_fields,
    popup=popup_cols,
    tiles=tiles,
    categorical=True,
    cmap=list(colors.values()),
    categories=list(colors.keys()),
    legend=True,
    style_kwds={"weight": 4},
    escape=False
    )
    m = add_map_elements(
        m,
        frame,
        show_route_pass_control=True,
        layer_column="hazard",
        tooltip_fields=tooltip_fields,
        popup_cols=popup_cols,
        categories=list(colors.keys()),
        cmap=list(colors.values()),
        style_kwds={"weight": 4},
        escape=False,
    )
    return m

def make_road_quality_map(
    gdf_segments: gpd.GeoDataFrame,
    popup_cols: list[str] | None = ["mtc_road_name", 'Road Quality', 'mtc_pci_info', 'mtc_pci_date',"Ride Type", "Turn", "Grade", "More Details"],
    tooltip_fields: list[str] | None = ['Segment', 'Road Quality'],
    tiles: str = "Cartodb Positron",
    hazard_profile: HazardProfileName = DEFAULT_HAZARD_PROFILE,
) -> folium.Map:
    """Build a Folium map with hazard-colored segments and route popups/tooltips."""
    frame = prepare_segment_display_columns(
        gdf_segments,
        hazard_profile=hazard_profile,
    )
    frame["road_quality_simple"] = frame["mtc_pci_info"].apply(simplify_road_quality_category)
    frame["Road Quality"] = frame["road_quality_simple"]
    colors = resolve_simplified_road_quality_profile(frame)
    frame["_display_color"] = frame["road_quality_simple"].map(colors).fillna("#8a8a8a")
    frame = _select_present_columns(
        frame,
        [
            "geometry",
            "step_dist_m",
            "Segment",
            "Road Quality",
            "mtc_road_name",
            "mtc_pci_info",
            "mtc_pci_date",
            "Ride Type",
            "Turn",
            "Grade",
            "More Details",
            "road_quality_simple",
            "_display_color",
        ],
    )
    m = frame.explore(
    column='road_quality_simple',
    tooltip=tooltip_fields,
    popup=popup_cols,
    tiles=tiles,
    categorical=True,
    cmap=list(colors.values()),
    categories=list(colors.keys()),
    legend=True,
    style_kwds={"weight": 4},
    escape=False
    )
    m = add_map_elements(
        m,
        frame,
    )
    return m


def make_chunk_map(
    gdf_segments: gpd.GeoDataFrame,
    popup_cols: list[str] | None = None,
    tooltip_fields: list[str] | None = None,
    tiles: str = "CartoDB Voyager",
) -> folium.Map:
    """Build a Folium map with chunk-state colored segments and chunk detail popups/tooltips."""
    if "chunk_state" in gdf_segments.columns:
        frame = gdf_segments.copy()
    else:
        from .analytics import detect_chunks
        frame = detect_chunks(gdf_segments)
    frame["Segment"] = frame["step"].astype("Int64").astype(str)
    frame["More Details"] = google_maps_link(frame["lat"], frame["lon"])
    frame["Road Name"] = frame["osm_name"].fillna("Unknown Road")
    frame["Turn"] = frame["step_turn"].round(2).astype(str) + "°"
    frame["Grade"] = frame["step_grade"].multiply(100).round(2).astype(str) + "%"
    frame["Chunk Avg Grade"] = frame["chunk_avg_grade"].multiply(100).round(2).astype(str) + "%"
    frame["Chunk Distance (ft)"] = frame["chunk_dist_ft"].round(0).astype("Int64").astype(str)
    frame["Candidate Chunk Distance (ft)"] = frame["candidate_chunk_dist_ft"].round(0).astype("Int64").astype(str)
    if "section_road_name" in frame.columns:
        frame["Section Road Name"] = frame["section_road_name"].fillna(frame["Road Name"])
    if "section_distance_mi" in frame.columns:
        frame["Section Distance (mi)"] = pd.to_numeric(
            frame["section_distance_mi"],
            errors="coerce",
        ).round(1)
    if "section_time_min" in frame.columns:
        frame["Section Time (min)"] = frame["section_time_min"].fillna("")
    frame["_display_color"] = frame["chunk_state"].map(CHUNK_STATE_COLORS).fillna("#8a8a8a")

    if tooltip_fields is None:
        tooltip_fields = [
            "chunk_state",
            "Road Name",
            "Chunk Distance (ft)",
            "Chunk Avg Grade",
            "Grade",
        ]
    if popup_cols is None:
        popup_cols = ["chunk_state"]
        if "Section Road Name" in frame.columns:
            popup_cols.extend([
                "Section Road Name",
                "Section Distance (mi)",
                "Section Time (min)",
            ])
        else:
            popup_cols.extend([
                "Road Name",
                "Chunk Distance (ft)",
            ])
        popup_cols.extend([
            "Chunk Avg Grade",
            "Candidate Chunk Distance (ft)",
            "Grade",
            "More Details",
        ])

    m = frame.explore(
        column="chunk_state",
        tooltip=tooltip_fields,
        popup=popup_cols,
        tiles=tiles,
        categorical=True,
        cmap=list(CHUNK_STATE_COLORS.values()),
        categories=list(CHUNK_STATE_COLORS.keys()),
        legend=True,
        style_kwds={"weight": 4},
        escape=False,
    )
    m = add_map_elements(
        m,
        frame,
    )
    return m
