import argparse
import sys
import yaml
from pathlib import Path
from simulations.run_all import run_all_simulations  # anpassen, wo die Funktion liegt


def main():
    script_dir = Path(__file__).resolve().parent
    default_config_dir = script_dir.parent / "configs"
    default_raw_dir = script_dir.parent / "data" / "raw"

    parser = argparse.ArgumentParser(
        description="Run simulations with one or more config files."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir,
        help="Path to directory containing config files (default: configs/)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=str,
        help="One or more config filenames to run (default: all *.yaml in config-dir)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_raw_dir,
        help="Base output directory (default: data/raw/)",
    )

    args = parser.parse_args()

    config_dir = args.config_dir
    outdir = args.outdir

    if not config_dir.exists():
        sys.exit(f"Error: Config directory not found: {config_dir}")

    # Wenn keine configs angegeben -> alle yaml im Ordner
    if args.configs:
        config_files = [config_dir / name for name in args.configs]
    else:
        config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        sys.exit(f"No config files found in {config_dir}")

    for config_path in config_files:
        if not config_path.exists():
            sys.exit(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        config_name = config_path.stem
        config_outdir = outdir / config_name
        config_outdir.mkdir(parents=True, exist_ok=True)

        print(f"Running simulations for {config_name} -> {config_outdir}")
        run_all_simulations(config_path=str(config_path), config=cfg)


if __name__ == "__main__":
    main()

