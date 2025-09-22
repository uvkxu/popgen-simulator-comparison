import os
import subprocess
from simulations.ms_runner import build_ms_command

def run_msms_simulation(config, output_file=None):
    default_msms_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bin/msms.exe"))
    msms_path = config.get("msms_path", default_msms_path)
    if not msms_path or not os.path.exists(msms_path):
        raise FileNotFoundError(f"msms executable not found at {msms_path}. Please check the path in config.")

    if "gene_conversion_rate" in config or "gene_conversion_tract_length" in config:
        raise ValueError("msms cannot be run with gene conversion")

    command = "msms" + build_ms_command(config).removeprefix("ms")

    if output_file is None:
        output_file = config.get("ms_output_file", "results/msms/output.txt")

    print(f"🚀 Running msms: {command}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cmd_parts = command.split()
    cmd_parts[0] = msms_path

    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ms failed:\n{result.stderr}")

    with open(output_file, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)