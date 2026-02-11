import pandas as pd
import gpxpy


def read_simple_gpx(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)

    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for index, point in enumerate(segment.points):
                row = {
                    "name": getattr(track, "name", None),
                    "step": index,
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "elevation_m": point.elevation,
                    "time": point.time,
                }
                if isinstance(point.elevation, (float, int)):
                    row["elevation_f"] = point.elevation * 3.28084
                else:
                    row["elevation_f"] = None
                rows.append(row)

    return pd.DataFrame(rows)
