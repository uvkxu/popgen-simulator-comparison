include: "../project_paths.py"
include: "../snakemake_vars.py"


rule compute_tree_hyperparameter_optuna:
    input:
        lambda wildcards: get_evaluation_input(
            config=wildcards.config,
            mode=wildcards.mode,
            feature_type="tree",
            group=wildcards.group_string,
        )
    output:
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/xgb_params_esgrid_{{group_string}}.yaml"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(
            input_folder=f"{TREE_FEATURES_DIR}/{wildcards.config}/",
            config=wildcards.config,
            group=wildcards.group_string,
            mode=wildcards.mode,
        )
    params:
        args_group=lambda wildcards: ",".join(wildcards.group_string.split("_")),
        input_dir=lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        output_dir=lambda wildcards: f"{TREE_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file=lambda wildcards: f"{TREE_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}/errorlog_optuna.err",
        # optional: adjust these defaults if you want
        es_grid="10,30,50",
        optuna_trials=50,
    conda:
        "../envs/evaluation-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/evaluation/get_top_features.py \
        --use_optuna \
        --es_grid {params.es_grid} \
        --optuna_trials {params.optuna_trials} \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """


rule compute_allel_hyperparameter_optuna:
    input:
        lambda wildcards: get_evaluation_input(
            config=wildcards.config,
            mode=wildcards.mode,
            feature_type="allel",
            group=wildcards.group_string,
        )
    output:
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/xgb_params_esgrid_{{group_string}}.yaml"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(
            input_folder=f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
            config=wildcards.config,
            group=wildcards.group_string,
            mode=wildcards.mode,
        )
    params:
        args_group=lambda wildcards: ",".join(wildcards.group_string.split("_")),
        input_dir=lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        output_dir=lambda wildcards: f"{ALLEL_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file=lambda wildcards: f"{ALLEL_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}/errorlog_optuna.err",
        # optional: adjust these defaults if you want
        es_grid="10,30,50",
        optuna_trials=50,
    conda:
        "../envs/evaluation-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/evaluation/get_top_features.py \
        --use_optuna \
        --es_grid {params.es_grid} \
        --optuna_trials {params.optuna_trials} \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """