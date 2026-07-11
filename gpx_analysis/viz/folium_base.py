from __future__ import annotations

import folium
import geopandas as gpd
import pandas as pd
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen
from shapely.geometry import Point

from .geometry import (
    _chevron_marker_segments,
    _number_marker_indexes,
    _resolve_number_marker_locations,
    _route_overlap_pass_indexes,
    _overlap_ignore_value,
)
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



def _add_fullscreen_mobile_fallback(m: folium.Map) -> None:
    """Open the standalone map when fullscreen is unavailable in an iframe."""
    script = Template(
        """
        {% macro script(this, kwargs) %}
        (function() {
            var canFullscreen = document.fullscreenEnabled
                || document.webkitFullscreenEnabled
                || document.mozFullScreenEnabled
                || document.msFullscreenEnabled;
            if (canFullscreen) {
                return;
            }

            var controls = document.querySelectorAll(
                ".leaflet-control-fullscreen a, .leaflet-control-fullscreen-button, .fullscreen-icon"
            );
            controls.forEach(function(control) {
                control.setAttribute("title", "Open full map");
                control.setAttribute("aria-label", "Open full map");
                control.addEventListener("click", function(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    window.open(window.location.href, "_blank");
                }, true);
            });
        })();
        {% endmacro %}
        """
    )
    fallback = MacroElement()
    fallback._template = script
    m.add_child(fallback)



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



def _present_interaction_fields(
    frame: gpd.GeoDataFrame,
    fields: list[str] | None,
) -> list[str] | None:
    """Return popup/tooltip fields present in the frame."""
    if fields is None:
        return None
    present = [field for field in fields if field in frame.columns]
    return present or None



def _add_touch_target_layer(
    parent: folium.Map | folium.FeatureGroup,
    frame: gpd.GeoDataFrame,
    popup_cols: list[str] | None,
    tooltip_fields: list[str] | None,
) -> None:
    """Add an invisible, touch-only wide route layer for easier mobile taps."""
    popup_fields = _present_interaction_fields(frame, popup_cols)
    tooltip_fields = _present_interaction_fields(frame, tooltip_fields)
    if not popup_fields and not tooltip_fields:
        return
    display_fields = [*(popup_fields or []), *(tooltip_fields or [])]
    display_fields = list(dict.fromkeys(display_fields))
    target_frame = frame[["geometry", *display_fields]].copy()
    for column in display_fields:
        if pd.api.types.is_datetime64_any_dtype(target_frame[column]):
            target_frame[column] = target_frame[column].astype(str)
    folium.GeoJson(
        data=target_frame.to_json(),
        name="Route Touch Target",
        control=False,
        style_function=lambda _: {
            "color": "#000000",
            "weight": 18,
            "opacity": 0,
            "fillOpacity": 0,
            "className": "route-touch-target",
        },
        pane="route-touch-targets",
        tooltip=(
            folium.GeoJsonTooltip(fields=tooltip_fields, sticky=True)
            if tooltip_fields
            else None
        ),
        popup=(
            folium.GeoJsonPopup(fields=popup_fields, max_width=320)
            if popup_fields
            else None
        ),
    ).add_to(parent)


def _hide_tile_layers_from_layer_control(m: folium.Map) -> None:
    """Keep fixed basemap tile layers out of the layer control."""
    for child in m._children.values():
        if isinstance(child, folium.raster_layers.TileLayer):
            child.control = False


def _add_layer_control_once(m: folium.Map) -> None:
    """Add one overlay layer control while keeping fixed basemaps hidden."""
    if any(isinstance(child, folium.map.LayerControl) for child in m._children.values()):
        return
    _hide_tile_layers_from_layer_control(m)
    folium.LayerControl(collapsed=True).add_to(m)


def _route_arrow_parents(m: folium.Map) -> list[folium.MacroElement]:
    """Return route layer groups that should own direction arrows."""
    parents: list[folium.MacroElement] = []
    for child in m._children.values():
        if getattr(child, "layer_name", None) in {"Route", "Outbound", "Return"} and isinstance(
            child,
            (folium.features.GeoJson, folium.map.FeatureGroup),
        ):
            parents.append(child)
    return parents



def _enable_touch_target_styles(m: folium.Map) -> None:
    """Make wide route hit targets interactive only on coarse pointers."""
    _ensure_map_pane(m, pane_name="route-touch-targets", z_index=575)
    style = Template(
        """
        {% macro header(this, kwargs) %}
        <style>
        .route-touch-target {
            pointer-events: none;
        }
        @media (hover: none), (pointer: coarse) {
            .route-touch-target {
                pointer-events: stroke;
            }
        }
        </style>
        {% endmacro %}
        """
    )
    style_element = MacroElement()
    style_element._template = style
    m.add_child(style_element)



def _add_direction_layers(
    m: folium.Map,
    frame: gpd.GeoDataFrame,
    column: str,
    tooltip_fields: list[str] | None,
    popup_cols: list[str] | None,
    categories: list[str] | None,
    cmap: object | None,
    style_kwds: dict[str, object] | None,
    escape: bool,
    categorical: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[folium.Map, bool]:
    """Replace only overlapping route sections with outbound/return toggle overlays."""
    projected = frame[["geometry"]].to_crs(3857)
    overlap_column = column if column in frame.columns else None
    ignore_value = _overlap_ignore_value(column)
    if overlap_column is not None:
        projected[overlap_column] = frame[overlap_column]
    outbound_indexes, return_indexes = _route_overlap_pass_indexes(
        projected,
        column=overlap_column,
        ignore_value=ignore_value,
    )
    if not outbound_indexes or not return_indexes:
        return m, False

    overlap_indexes = outbound_indexes | return_indexes
    base = frame.loc[~frame.index.isin(overlap_indexes)].copy()
    outbound = frame.loc[frame.index.isin(outbound_indexes)].copy()
    returning = frame.loc[frame.index.isin(return_indexes)].copy()
    if outbound.empty or returning.empty:
        return m, False

    _remove_geojson_layers(m)
    _enable_touch_target_styles(m)
    explore_kwargs = {
        "column": column,
        "tooltip": tooltip_fields,
        "popup": popup_cols,
        "categorical": categorical,
        "legend": False,
        "style_kwds": style_kwds or {"weight": 4},
        "escape": escape,
    }
    if categories is not None:
        explore_kwargs["categories"] = categories
    if cmap is not None:
        explore_kwargs["cmap"] = cmap
    if vmin is not None:
        explore_kwargs["vmin"] = vmin
    if vmax is not None:
        explore_kwargs["vmax"] = vmax

    if not base.empty:
        base.explore(
            m=m,
            name="Route",
            **explore_kwargs,
        )
        _add_touch_target_layer(m, base, popup_cols, tooltip_fields)

    outbound_layer = folium.FeatureGroup(name="Outbound", overlay=True, control=False, show=True)
    outbound_layer.add_to(m)
    outbound.explore(
        m=outbound_layer,
        **explore_kwargs,
    )
    _add_touch_target_layer(outbound_layer, outbound, popup_cols, tooltip_fields)
    return_layer = folium.FeatureGroup(name="Return", overlay=True, control=False, show=False)
    return_layer.add_to(m)
    returning.explore(
        m=return_layer,
        **explore_kwargs,
    )
    _add_touch_target_layer(return_layer, returning, popup_cols, tooltip_fields)
    _add_direction_radio_control(m, outbound_layer, return_layer)
    return m, True



def add_map_elements(
    m: folium.Map,
    frame: gpd.GeoDataFrame,
    show_numbers: bool = True,
    show_route_pass_control: bool = False,
    layer_column: str | None = None,
    tooltip_fields: list[str] | None = None,
    popup_cols: list[str] | None = None,
    categories: list[str] | None = None,
    cmap: object | None = None,
    style_kwds: dict[str, object] | None = None,
    touch_target_frame: gpd.GeoDataFrame | None = None,
    escape: bool = False,
    show_gravel_overlay: bool = False,
    categorical: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    has_split_direction_layers = False
    has_gravel_overlay = False
    if show_route_pass_control and layer_column:
        m, has_split_direction_layers = _add_direction_layers(
            m,
            frame,
            column=layer_column,
            tooltip_fields=tooltip_fields,
            popup_cols=popup_cols,
            categories=categories,
            cmap=cmap,
            style_kwds=style_kwds,
            escape=escape,
            categorical=categorical,
            vmin=vmin,
            vmax=vmax,
        )
    if not has_split_direction_layers:
        _enable_touch_target_styles(m)
        _add_touch_target_layer(
            m,
            touch_target_frame if touch_target_frame is not None else frame,
            popup_cols,
            tooltip_fields,
        )
    if show_gravel_overlay:
        from .overlays import _add_gravel_overlay

        has_gravel_overlay = _add_gravel_overlay(m, frame)
    # Fullscreen control
    Fullscreen(
        position="topleft",
        title="Fullscreen",
        title_cancel="Exit fullscreen",
        force_separate_button=True,
    ).add_to(m)
    _add_fullscreen_mobile_fallback(m)
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
        number_layer = folium.FeatureGroup(
            name="Route Numbers",
            overlay=True,
            control=True,
            show=True,
        )
        number_layer.add_to(m)
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
            ).add_to(number_layer)
    if show_numbers or has_gravel_overlay:
        _add_layer_control_once(m)
    # Direction chevrons
    _ensure_map_pane(m, pane_name="route-chevrons", z_index=650)
    arrow_parents = _route_arrow_parents(m)
    if not arrow_parents:
        arrow_parents = [m]
    for chevron_paths in _chevron_marker_segments(frame):
        for chevron_path in chevron_paths:
            for arrow_parent in arrow_parents:
                folium.PolyLine(
                    locations=chevron_path,
                    color="#111111",
                    weight=2,
                    opacity=0.95,
                    line_cap="square",
                    line_join="round",
                    pane="route-chevrons",
                    interactive=False,
                ).add_to(arrow_parent)
    _disable_tooltips_on_touch(m)
    return m

