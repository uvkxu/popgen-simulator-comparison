import os
from datetime import datetime

def make_run_output_dir(config_path, base_dir="data/raw"):
    """
    Create a directory for simulation results based on config file name and current timestamp.
    Example:
        configs/sim_split.yaml → data/raw/sim_split/2025-09-18_22-45-12/
    """
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = os.path.join(base_dir, config_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
