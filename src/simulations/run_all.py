from pathlib import Path
from simulations.ms_runner import run_ms_simulation
from simulations.msms_runner import run_msms_simulation
from simulations.msprime_runner import run_msprime_simulation
from utils.io_helpers import make_run_output_dir


def run_all_simulations(config_path, config, output_dir=None):
    """
    Run all simulations (ms, msms, msprime) for a given config.

    Parameters
    ----------
    config_path : str or Path
        Path to the config file.
    config : dict
        Parsed config dictionary.
    output_dir : str or Path, optional
        Directory where results will be saved. If None, falls back to make_run_output_dir().
    """

    if output_dir is None:
        run_dir = make_run_output_dir(config_path)
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    print(run_dir)
    ms_file = run_dir / "ms.txt"
    msms_file = run_dir / "msms.txt"
    msprime_file = run_dir / "msprime.txt"

    run_ms_simulation(config, output_file=ms_file)
    run_msms_simulation(config, output_file=msms_file)
    run_msprime_simulation(config, output_file=msprime_file)

    print(f"✅ Results saved in {run_dir}")
