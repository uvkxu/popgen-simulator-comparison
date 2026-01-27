include: "../snakemake_vars.py"

rule run_config_combo_rep_simulation_demes:
    input:
        config_file = f"{CONFIG_DIR}/{{config}}.yaml"
    output:
        output_file = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt"
    params:
        combo_json=lambda w: combo_folder_to_json[w.combo],
        err_file=lambda w: f"{RAW_DATA_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
    resources:
        mem_mb = lambda wildcards, attempt: estimate_mem_mb(
            config=wildcards.config,
            combo=wildcards.combo,
            simulator=wildcards.simulator,
            func="simulate"
        )
    conda:
        "../envs/simulations-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
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
        combo_json=lambda w: combo_folder_to_json[w.combo],
        err_file = lambda w: f"{RAW_DATA_DIR}/{w.config}/{w.combo}/errorlog.err"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/simulations/combo_to_yaml.py \
        --config_path {input.config_file} \
        --combo '{params.combo_json}' \
        --output_file {output.output_file} \
        --nreps {NUM_REPLICATES}
        """