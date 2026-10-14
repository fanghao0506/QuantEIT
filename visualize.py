"""Visualize a saved reconstruction without rerunning the optimizer."""
import argparse
import json
from pathlib import Path

from quanteit_lung.data import DEFAULT_DATA, ROOT
from quanteit_lung.visualization import DEFAULT_THRESHOLD, visualize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "examples" / "reconstruction.npz")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "visualization")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Display-only normalized isosurface level")
    args = parser.parse_args()
    output = visualize(args.result, args.output, args.data_dir, args.threshold)
    print(f"Open {output / 'viewer_3d.html'} in any modern browser (offline).")
    settings = json.loads((output / "display_settings.json").read_text(encoding="utf-8"))
    print(f"Projections: {output / 'projections.png'}" if settings["frame_count"] == 1 else f"Animation: {output / 'sequence.gif'}")


if __name__ == "__main__":
    main()

