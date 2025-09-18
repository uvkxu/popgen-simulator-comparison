import os
import sys
import yaml
from pathlib import Path
from simulations.run_all import run_all_simulations  # anpassen, wo die Funktion liegt

def main():
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir.parent / "configs" / "sample_config_2.yaml"

    # CLI-Arg oder Default
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        config_path = arg_path if arg_path.is_absolute() else script_dir.parent / arg_path
    else:
        config_path = default_config

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Laden der Config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Übergabe an run_all_simulations
    run_all_simulations(config_path=str(config_path), config=cfg)

if __name__ == "__main__":
    main()
