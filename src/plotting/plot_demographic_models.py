import matplotlib
matplotlib.use("Agg")

import demes
import demesdraw
from pathlib import Path

def plot_demes_folder(input_folder: str, output_folder: str = None):
    """
    Load all .yaml/.yml files in input_folder, draw their demes graph,
    and save as PNGs with the same name.
    """
    input_path = Path(input_folder)
    if output_folder is None:
        output_path = input_path
    else:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

    yaml_files = list(input_path.glob("*.yaml")) + list(input_path.glob("*.yml"))

    if not yaml_files:
        print(f"No YAML files found in {input_folder}")
        return

    for yaml_file in yaml_files:
        print(f"Processing {yaml_file.name}...")
        # Load the demes graph
        graph = demes.load(str(yaml_file))
        # Compute layout separation heuristic (optional)
        demesdraw.utils.separation_heuristic(graph)
        # Draw the graph
        ax = demesdraw.tubes(graph, log_time=True)
        # Save the figure with the same name as the YAML file
        out_file = output_path / (yaml_file.stem + ".png")
        ax.figure.savefig(out_file, dpi=300, bbox_inches="tight")
        ax.figure.clf()  # Clear figure to avoid overlap
        print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_demes.py <input_folder> [output_folder]")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    plot_demes_folder(input_folder, output_folder)
