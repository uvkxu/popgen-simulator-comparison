include: "../snakemake_vars.py"

rule run_one_config_all_simulations_demes:
    input:
       lambda wildcards: f"{CONFIG_DIR}/{wildcards.config}.yaml"
    output:
        dummy=f"{RAW_DATA_DIR}/{{config}}/.done"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        """
        python src/simulations/demes_run_all.py --config-dir {CONFIG_DIR}/ --configs {wildcards.config}.yaml --reps {REPS} --rec_reps {REC_REPS} --cmddir {CMD_DIR} --cmd_variables variables.txt --ms_mod_path {DEFAULT_MS_MOD_PATH}
        touch {output.dummy}
        """

rule run_one_config_all_simulations_own:
    input:
       lambda wildcards: f"{OWN_CONFIG_DIR}/{wildcards.config}.yaml"
    output:
        dummy=f"{RAW_DATA_DIR}/{{config}}/.done"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        """
        python src/simulations/run_all.py --config-dir {OWN_CONFIG_DIR}/ --configs {wildcards.config}.yaml --reps {REPS} --rec_reps {REC_REPS}
        touch {output.dummy}
        """