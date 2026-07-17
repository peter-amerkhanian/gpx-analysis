from pathlib import Path

PROJECTED_CRS = 3857
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BART_KML_PATH = DATA_DIR / "bart_stations.kml"
BART_STATION_LAYER = "BART Station"
OSM_DATA_DIR = DATA_DIR / "osm"
VITAL_SIGNS_DATA_DIR = DATA_DIR / "vital-signs"
LOCAL_OSM_NODES_PATH = OSM_DATA_DIR / "sf_bay_area_all_public_nodes.parquet"
LOCAL_OSM_EDGES_PATH = OSM_DATA_DIR / "sf_bay_area_all_public_edges.parquet"
LOCAL_MTC_STREETS_PATH = VITAL_SIGNS_DATA_DIR / "Streets_and_Roads_2026.geojson"
LOCAL_MTC_STREETS_PARQUET_PATH = VITAL_SIGNS_DATA_DIR / "Streets_and_Roads_2026.parquet"
LOCAL_MTC_STREET_ATTRS = [
    "start_location",
    "end_location",
    "road_name",
    "pci_date",
    "pci_info",
]
LOCAL_OSM_NETWORK_TYPE = "all_public"
LOCAL_OSM_CRS = 4326
LOCAL_OSM_TILE_SIZE_DEG = 0.05
LOCAL_OSM_NODES_TILE_DIR = OSM_DATA_DIR / "sf_bay_area_all_public_nodes_tiles"
LOCAL_OSM_EDGES_TILE_DIR = OSM_DATA_DIR / "sf_bay_area_all_public_edges_tiles"
OSM_HIGHWAY_PRIORITY = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "unclassified": 5,
    "residential": 6,
    "living_street": 7,
    "road": 8,
    "service": 9,
    "track": 10,
    "cycleway": 11,
    "path": 12,
    "footway": 13,
    "pedestrian": 14,
    "steps": 15,
}

ROAD_NAME_SUFFIXES = {
    "ave",
    "avenue",
    "blvd",
    "boulevard",
    "cir",
    "circle",
    "ct",
    "court",
    "dr",
    "drive",
    "hwy",
    "highway",
    "ln",
    "lane",
    "pkwy",
    "parkway",
    "pl",
    "place",
    "rd",
    "road",
    "st",
    "street",
    "ter",
    "terrace",
    "trl",
    "trail",
    "way",
}
