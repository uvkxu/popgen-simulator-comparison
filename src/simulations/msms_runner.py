import os
import subprocess
from ms_runner import build_ms_command, get_rate_multiplier


def build_msms_command(config, run_msms_ms=False):
    ms_cmd = build_ms_command(config)
    parts = ms_cmd.removeprefix("ms-mod").split()

    if "-p" in parts:
        index = parts.index("-p")
        # remove "-p" and the following argument
        parts.pop(index)
        parts.pop(index)  # same index because the list has shifted

    msms_ms_cmd = "-ms " if run_msms_ms else ""
    msms_cmd = "msms " + msms_ms_cmd + " ".join(parts)

    return msms_cmd

def run_msms_simulation(config=None, command=None, output_file=None, run_msms_ms=False):
    msms_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bin/msms"))

    if config is not None:
        msms_path = config.get("msms_path", msms_path)

    if not msms_path or not os.path.exists(msms_path):
        raise FileNotFoundError(f"msms executable not found at {msms_path}. Please check the path in config.")

    if config is not None and (
        "gene_conversion_rate" in config or "gene_conversion_tract_length" in config
    ):
        raise ValueError("msms cannot be run with gene conversion parameters.")

    # Build command
    if config is not None:
        command = build_msms_command(config, run_msms_ms)
        cmd_parts = command.split()
        cmd_parts[0] = msms_path
    elif command is not None:
        # Ensure command starts with msms path
        cmd_parts = command.split()
        first_token = os.path.basename(cmd_parts[0])
        if first_token != "msms" and first_token != "msms.exe":
            cmd_parts.insert(0, msms_path)
    else:
        raise ValueError("Either 'config' or 'command' must be provided.")

    if output_file is None:
        output_file = config.get("msms_output_file", "results/msms/output.txt") if config else "results/msms/output.txt"

    if run_msms_ms:
        print(f"🚀 Running msms_ms: {' '.join(cmd_parts)}")
    else:
        print(f"🚀 Running msms: {' '.join(cmd_parts)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"msms failed:\n{result.stderr}")

    with open(output_file, "w") as f:
        f.write(result.stdout)
        if result.stderr.strip():
            f.write("\n\n# STDERR OUTPUT:\n")
            f.write(result.stderr)

    return output_file


def run_msms_ms_simulation(config, output_file=None):
    run_msms_simulation(config, output_file, True)