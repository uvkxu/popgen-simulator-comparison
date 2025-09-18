from utils.io_helpers import make_run_output_dir
from simulations.ms_runner import run_ms_simulation
from simulations.msms_runner import run_msms_simulation
from simulations.msprime_runner import run_msprime_simulation

def run_all_simulations(config_path, config):
    run_dir = make_run_output_dir(config_path)

    ms_file = f"{run_dir}/ms.txt"
    msms_file = f"{run_dir}/msms.txt"
    msprime_file = f"{run_dir}/msprime.txt"

    run_ms_simulation(config, output_file=ms_file)
    run_msms_simulation(config, output_file=msms_file)
    run_msprime_simulation(config, output_file=msprime_file)

    print(f"✅ Results saved in {run_dir}")
