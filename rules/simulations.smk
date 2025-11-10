include: "../snakemake_vars.py"

rule run_one_config_all_simulations:
    input:
       lambda wildcards: f"{CONFIG_DIR}/{wildcards.config}.yaml"
    output:
        dummy=f"{RAW_DATA_DIR}/{{config}}/.done"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        """
        python src/simulations/run_all.py --configs {wildcards.config}.yaml --reps {REPS} --rec_reps {REC_REPS}
        touch {output.dummy}
        """


rule run_commands_variables:
    input:
        variables = f"{CMD_DIR}/variables.txt"
    output:
        dummy = f"{RAW_DATA_DIR}/{CMD_SUBDIR}/.done"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        """
        python src/simulations/run_all.py --cmddir {CMD_DIR} --cmd_variables variables.txt --outdir {RAW_DATA_DIR}/{CMD_SUBDIR}
        touch {output.dummy}
        """