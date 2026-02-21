import pandas as pd
import gpxpy


def read_simple_gpx(path: str, reverse: bool = False) -> pd.DataFrame:
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
    df = pd.DataFrame(rows)
    if reverse:
        df = df.iloc[::-1].reset_index(drop=True)
        df["step"] = range(len(df))
    return df
