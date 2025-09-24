from pathlib import Path
from datetime import datetime

def make_run_output_dir(config_path, base_dir="data/raw", timestamp=False):
    """
    Create a directory for simulation results based on config file name and current timestamp.
    """
    config_name = Path(config_path).stem
    run_dir = Path(base_dir) / config_name

    if timestamp:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = run_dir / ts

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
