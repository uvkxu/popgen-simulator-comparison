include: "../snakemake_vars.py"

rule run_one_config_all_simulations:
    input:
        config_file=lambda wildcards: f"{CONFIG_DIR}/{wildcards.config}.yaml"
    output:
        directory(f"{RAW_DATA_DIR}/{{config}}")
    conda:
        "../envs/simulations-env.yaml"
    shell:
        """
        python src/simulations/run_all.py --configs {wildcards.config}.yaml --reps {REPS} --rec_reps {REC_REPS}
        """

