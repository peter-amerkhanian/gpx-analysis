from __future__ import annotations

import folium
import geopandas as gpd
import pandas as pd

from .folium_base import _ensure_map_pane
def _gravel_overlay_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return route segments that should be marked as gravel."""
    if frame.empty:
        return frame.iloc[0:0].copy()

    gravel_mask = pd.Series(False, index=frame.index)
    if "road_type" in frame.columns:
        gravel_mask = gravel_mask | frame["road_type"].fillna("").astype(str).str.lower().eq("gravel")
    if "mtc_pci_info" in frame.columns:
        gravel_mask = gravel_mask | frame["mtc_pci_info"].fillna("").astype(str).str.lower().eq("gravel")
    return frame.loc[gravel_mask, ["geometry"]].copy()



def _add_gravel_overlay(m: folium.Map, frame: gpd.GeoDataFrame) -> bool:
    """Add a subtle dashed overlay for gravel route segments."""
    gravel = _gravel_overlay_frame(frame)
    if gravel.empty:
        return False

    _ensure_map_pane(m, pane_name="route-gravel-overlay", z_index=390)
    folium.GeoJson(
        data=gravel.to_json(),
        name="Gravel Segments",
        control=True,
        style_function=lambda _: {
            "color": "#b37400",
            "weight": 9,
            "opacity": 1,
            "className": "route-gravel-overlay",
        },
        pane="route-gravel-overlay",
        interactive=False,
    ).add_to(m)
    return True

