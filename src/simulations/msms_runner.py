import os
import subprocess
from ms_runner import build_ms_command, clean_output, get_rate_multiplier


def build_msms_command(config, run_msms_ms=False):
    ms_cmd = build_ms_command(config)
    parts = ms_cmd.removeprefix("ms").split()

    if "-p" in parts:
        index = parts.index("-p")
        # remove "-p" and the following argument
        parts.pop(index)
        parts.pop(index)  # same index because the list has shifted

    msms_ms_cmd = "-ms " if run_msms_ms else ""
    msms_cmd = "msms " + msms_ms_cmd + " ".join(parts)

    return msms_cmd

def run_msms_simulation(config, output_file=None, run_msms_ms=False):
    default_msms_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bin/msms.exe"))
    msms_path = config.get("msms_path", default_msms_path)
    if not msms_path or not os.path.exists(msms_path):
        raise FileNotFoundError(f"msms executable not found at {msms_path}. Please check the path in config.")

    if "gene_conversion_rate" in config or "gene_conversion_tract_length" in config:
        raise ValueError("msms cannot be run with gene conversion")

    command = build_msms_command(config, run_msms_ms)

    if output_file is None:
        output_file = config.get("msms_output_file", "results/msms/output.txt")

    if run_msms_ms:
        print(f"🚀 Running msms_ms: {command}")
    else:
        print(f"🚀 Running msms: {command}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cmd_parts = command.split()
    cmd_parts[0] = msms_path

    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"msms failed:\n{result.stderr}")

    with open(output_file, "w") as f:
        cleaned_result = clean_output(
            ms_output = result.stdout, 
            sequence_length = config.get("sequence_length", None),
            factor = get_rate_multiplier(config)
            )
        f.write(cleaned_result)
        f.write(result.stderr)

def run_msms_ms_simulation(config, output_file=None):
    run_msms_simulation(config, output_file, True)