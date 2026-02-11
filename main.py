from pathlib import Path
import argparse

from gpx_analysis import analyze_steps, read_simple_gpx


def resolve_input_path(path: str | None) -> Path:
    if path:
        return Path(path)

    data_dir = Path("gpx_data")
    first_gpx = sorted(data_dir.glob("*.gpx"))
    if not first_gpx:
        raise FileNotFoundError("No .gpx files found in gpx_data/ and no --path provided.")
    return first_gpx[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic GPX step analysis.")
    parser.add_argument("--path", type=str, default=None, help="Path to a .gpx file")
    parser.add_argument("--rolling-window", type=int, default=3, help="Rolling window for hazard features")
    args = parser.parse_args()

    input_path = resolve_input_path(args.path)
    points = read_simple_gpx(str(input_path))
    analyzed = analyze_steps(points, rolling_window=args.rolling_window)

    hazard_counts = analyzed["hazard"].value_counts()
    print(f"Input: {input_path}")
    print(f"Points: {len(analyzed)}")
    print("Hazard counts:")
    print(hazard_counts.to_string())


if __name__ == "__main__":
    main()
