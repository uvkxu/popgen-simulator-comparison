include: "../snakemake_vars.py"

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
        python src/main.py --configs {wildcards.config}.yaml
        """

rule run_all_simulations:
    input:
        expand(f"{RAW_DATA_DIR}/{{config}}/msprime.txt", config=all_configs),
        expand(f"{RAW_DATA_DIR}/{{config}}/ms.txt", config=all_configs),
        expand(f"{RAW_DATA_DIR}/{{config}}/msms.txt", config=all_configs)

