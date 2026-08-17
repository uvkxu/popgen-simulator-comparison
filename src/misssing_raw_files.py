from pathlib import Path

RAW_INPUTS = [
    f"{RAW_DATA_DIR}/{config}/{combo}/rep{i+1}/{FILE_FORMATS['raw'].format(SIM=sim)}"
    for config in all_configs
    for combo in combinations_by_config[config]
    for sim in simulator_interest
    for i in range(REPS)
]

missing = [p for p in RAW_INPUTS if not Path(p).exists()]

with open("missing_raw_inputs.txt", "w") as f:
    f.write("\n".join(missing))

print(f"{len(missing)} files missing.")