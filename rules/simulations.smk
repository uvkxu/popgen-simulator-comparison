include: "../project_paths.py"
include: "../snakemake_vars.py"

rule run_config_combo_rep_simulation_demes:
    input:
        config_file = f"{DEM_MODELS_DIR}/{{config}}.yaml"
    output:
        output_file = f"{DATA_DIR}/{{scenario}}/raw/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt"
    params:
        combo_json=lambda w: combo_folder_to_json[w.combo],
        err_file=lambda w: f"{DATA_DIR}/{w.scenario}/raw/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
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
            --nreps {INTERNAL_REPS} \
            --output_file {output.output_file}
        """


rule create_combo_yaml_file:
    input:
        config_file = f"{DEM_MODELS_DIR}/{{config}}.yaml"
    output:
        output_file = f"{DATA_DIR}/{{scenario}}/raw/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    params:
        combo_json=lambda w: combo_folder_to_json[w.combo],
        err_file = lambda w: f"{DATA_DIR}/{w.scenario}/raw/{w.config}/{w.combo}/errorlog.err"
    conda:
        "../envs/simulations-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/simulations/combo_to_yaml.py \
        --config_path {input.config_file} \
        --scenario {wildcards.scenario}\
        --combo '{params.combo_json}' \
        --output_file {output.output_file} \
        --nreps {INTERNAL_REPS}
        """