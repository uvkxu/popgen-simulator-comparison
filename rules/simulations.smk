include: "../snakemake_vars.py"

rule run_config_combo_rep_simulation_demes:
    input:
        config_file = f"{CONFIG_DIR}/{{config}}.yaml"
    output:
        output_file = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt"
    params:
        combo_json=lambda w: combo_folder_to_json[w.combo],
        file_name=lambda w: FILE_FORMATS["raw"].format(SIM=w.simulator),
        rep_dir=lambda w: FILE_FORMATS["reps"].format(REP=w.rep)
    conda:
        "../envs/simulations-env.yaml"
    shell:
        r"""
        python src/simulations/demes_run.py \
            --config_path {input.config_file}\
            --simulator {wildcards.simulator} \
            --combo '{params.combo_json}' \
            --nreps {NUM_REPLICATES} \
            --output_file {output.output_file}
        """


rule create_combo_yaml_file:
    input:
        config_file = f"{CONFIG_DIR}/{{config}}.yaml"
    output:
        output_file = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    params:
        combo_json=lambda w: combo_folder_to_json[w.combo]
    conda:
        "../envs/simulations-env.yaml"
    shell:
        r"""
        python src/simulations/combo_to_yaml.py \
        --config_path {input.config_file} \
        --combo '{params.combo_json}' \
        --output_file {output.output_file} \
        --nreps {NUM_REPLICATES}
        """



rule run_one_config_all_simulations_demes:
    input:
        lambda wildcards: f"{CONFIG_DIR}/{wildcards.config}.yaml"
    output:
        dummy=f"{RAW_DATA_DIR}/{{config}}/.done"
    log:
        out=f"{RAW_DATA_DIR}/{{config}}/run.out"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        r"""
        mkdir -p $(dirname {log.out})
        tmp_err=$(mktemp)
        python src/simulations/demes_run_all.py \
            --config-dir {CONFIG_DIR}/ \
            --configs {wildcards.config}.yaml \
            --reps {REPS} \
            --rec_reps {REC_REPS} \
            --cmddir {CMD_DIR} \
            --cmd_variables variables.txt \
            --ms_mod_path {DEFAULT_MS_MOD_PATH} \
            > {log.out} 2> $tmp_err || {{ 
                mv $tmp_err {RAW_DATA_DIR}/{wildcards.config}/run.err; 
                exit 1; 
            }}
        rm -f $tmp_err
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