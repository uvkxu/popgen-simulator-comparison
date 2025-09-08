import sys
import yaml
import os
from ms_runner import run_ms_simulation
from msprime_runner import run_msprime_simulation
import msprime

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "..", "configs", "sample_config_2.yaml")

    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if not os.path.isabs(arg_path):
            config_path = os.path.join(script_dir, "..", arg_path)
        else:
            config_path = arg_path
    else:
        config_path = default_config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_ms_simulation(cfg)
    run_msprime_simulation(cfg)

if __name__ == "__main__":
    main()
