include: "../snakemake_vars.py"


rule copy_combo_yaml_tree_features:
    output:
        copy_file = f"{TREE_FEATURES_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    params:
        input = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    run:
        from pathlib import Path
        import shutil

        Path(output.copy_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(params.input, output.copy_file)


rule copy_combo_yaml_allel_features:
    output:
        copy_file = f"{ALLEL_FEATURES_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    params:
        input = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    run:
        from pathlib import Path
        import shutil

        Path(output.copy_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(params.input, output.copy_file)


rule copy_combo_yaml_cleaned:
    output:
        copy_file = f"{CLEAN_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    params:
        input = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    run:
        from pathlib import Path
        import shutil

        Path(output.copy_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(params.input, output.copy_file)


rule copy_combo_yaml_tree_block:
    output:
        copy_file = f"{TREEBLOCK_FEATURES_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    params:
        input = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    run:
        from pathlib import Path
        import shutil

        Path(output.copy_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(params.input, output.copy_file)

    
rule compute_tree_topfeatures_grouped_by_mode:
    output:
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_barplot.png",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{XGB_STR}.png"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(input_folder=f"{TREE_FEATURES_DIR}/{wildcards.config}/", config=wildcards.config, group=wildcards.group_string, mode=wildcards.mode)
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        input_dir = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        output_dir = lambda wildcards: f"{TREE_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{TREE_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """


rule compute_allel_topfeatures_grouped_by_mode:
    output:
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_barplot.png",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{XGB_STR}.png"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(input_folder=f"{ALLEL_FEATURES_DIR}/{wildcards.config}/", config=wildcards.config, group=wildcards.group_string, mode=wildcards.mode)
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        input_dir = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        output_dir = lambda wildcards: f"{ALLEL_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{ALLEL_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """

rule compute_tree_hyperparameter_optuna:
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
        "../envs/models-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --use_optuna \
        --es_grid {params.es_grid} \
        --optuna_trials {params.optuna_trials} \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """


rule compute_allel_hyperparameter_optuna:
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
        "../envs/models-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --use_optuna \
        --es_grid {params.es_grid} \
        --optuna_trials {params.optuna_trials} \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """


rule compute_together_topfeatures_grouped_by_mode:
    output:
        f"{TOGETHER_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv",
        f"{TOGETHER_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_barplot.png",
        f"{TOGETHER_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{XGB_STR}.png"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(input_folder=f"{ALLEL_FEATURES_DIR}/{wildcards.config}/", config=wildcards.config, group=wildcards.group_string, mode=wildcards.mode)
    params:
        input_dir_allel=lambda wc: f"{ALLEL_FEATURES_DIR}/{wc.config}/",
        input_dir_tree=lambda wc: f"{TREE_FEATURES_DIR}/{wc.config}/",
        input_dir_cleaned=lambda wc: f"{CLEAN_DIR}/{wc.config}/",
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        output_dir = lambda wildcards: f"{TOGETHER_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{TOGETHER_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features_together.py \
        --input_folder_tree {params.input_dir_tree} \
        --input_folder_allel {params.input_dir_allel} \
        --input_folder_cleaned {params.input_dir_cleaned} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """

rule compute_actual_topfeatures_grouped_by_mode:
    output:
        f"{TIPTOP_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(
            input_folder=f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
            config=wildcards.config,
            group=wildcards.group_string,
            mode=wildcards.mode
        )
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
         input_dir_allel=lambda wc: [
            f"{ALLEL_FEATURES_DIR}/{wc.config}/",
            f"{MSPMS_ALLEL_FEATURES_DIR}/{wc.config}/",
        ],
        input_dir_tree=lambda wc: [
            f"{TREE_FEATURES_DIR}/{wc.config}/",
            f"{MSPMS_TREE_FEATURES_DIR}/{wc.config}/",
        ],
        output_dir=lambda wildcards: f"{TIPTOP_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file=lambda wildcards: f"{TIPTOP_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/only_top_features.py \
        --input_folder_tree {params.input_dir_tree} \
        --input_folder_allel {params.input_dir_allel} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """



#---------------Statistical Tests-----------------

rule pairwise_statistical_test:
    output:
        output_file = f"{STATISTICS_RESULTS_DIR}/{{config}}/{{mode}}/{{group_string}}_feature_comparison.csv"
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        input_dir_allel=lambda wc: f"{ALLEL_FEATURES_DIR}/{wc.config}/",
        input_dir_tree=lambda wc: f"{TREEBLOCK_FEATURES_DIR}/{wc.config}/",
        err_file=lambda wildcards: f"{STATISTICS_RESULTS_DIR}/{wildcards.config}/{wildcards.mode}/{wildcards.group_string}.err"
    conda:
        "../envs/stats-test-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/statistical_tests.py \
        --input_folder_tree {params.input_dir_tree} \
        --input_folder_allel {params.input_dir_allel} \
        --output_file {output.output_file} \
        --grouped_by {wildcards.mode} \
        --simulators {params.args_group}
        """