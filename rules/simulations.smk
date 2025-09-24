import os
from glob import glob

CONFIG_DIR = "configs"
RAW_DATA_DIR = "data/raw"

# Alle Config-Dateien ohne .yaml-Endung und ohne Pfad
all_configs = [os.path.splitext(os.path.basename(f))[0] for f in glob(f"{CONFIG_DIR}/*.yaml")]

rule run_one_simulation:
    input:
        config_file=lambda wildcards: f"{CONFIG_DIR}/{wildcards.config}.yaml"
    output:
        msprime_out=f"{RAW_DATA_DIR}/{{config}}/msprime.txt",
        ms_out=f"{RAW_DATA_DIR}/{{config}}/ms.txt",
        msms_out=f"{RAW_DATA_DIR}/{{config}}/msms.txt"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        """
        # Übergabe nur des Dateinamens an main.py
        python src/main.py --configs {wildcards.config}.yaml
        """

rule run_all_simulations:
    input:
        expand(f"{RAW_DATA_DIR}/{{config}}/msprime.txt", config=all_configs),
        expand(f"{RAW_DATA_DIR}/{{config}}/ms.txt", config=all_configs),
        expand(f"{RAW_DATA_DIR}/{{config}}/msms.txt", config=all_configs)

