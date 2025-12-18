include: "../snakemake_vars.py"


rule copy_combo_yaml_tree_features:
    input:
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        copy_file = f"{TREE_FEATURES_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    run:
        from pathlib import Path
        import shutil

        Path(output.copy_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input.combo_yaml, output.copy_file)


rule copy_combo_yaml_allel_features:
    input:
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        copy_file = f"{ALLEL_FEATURES_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    run:
        from pathlib import Path
        import shutil

        Path(output.copy_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input.combo_yaml, output.copy_file)

    
rule compute_tree_topfeatures_grouped_by_mode:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, wildcards.group_string, wildcards.mode, "tree")
    output:
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_barplot.png",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{XGB_STR}.png",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{RF_STR}.png",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{LR_STR}.png",
    wildcard_constraints:
        mode = "|".join(VALID_GROUPING_MODES)
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


rule compute_tree_topfeatures_single:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, wildcards.group_string, "single", "tree", wildcards.combo)
    output:
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_ranking_{{group_string}}_with_summary.csv",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_ranking_{{group_string}}_barplot.png",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_importance_{{group_string}}_{XGB_STR}.png",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_importance_{{group_string}}_{RF_STR}.png",
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_importance_{{group_string}}_{LR_STR}.png",
    wildcard_constraints:
        combo = lambda wc: wc in all_combo_folders
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        input_dir = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/{wildcards.combo}",
        output_dir = lambda wildcards: f"{TREE_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.combo}",
        err_file = lambda wildcards: f"{TREE_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.combo}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by single
        """


rule compute_allel_topfeatures_grouped_by_mode:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, wildcards.group_string, wildcards.mode, "allel")
    output:
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_barplot.png",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{XGB_STR}.png",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{RF_STR}.png",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_importance_{{group_string}}_{LR_STR}.png",
    wildcard_constraints:
        mode = "|".join(VALID_GROUPING_MODES)
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

rule compute_allel_topfeatures_single:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, wildcards.group_string, "single", "allel", wildcards.combo)
    output:
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_ranking_{{group_string}}_with_summary.csv",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_ranking_{{group_string}}_barplot.png",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_importance_{{group_string}}_{XGB_STR}.png",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_importance_{{group_string}}_{RF_STR}.png",
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{combo}}/feature_importance_{{group_string}}_{LR_STR}.png",
    wildcard_constraints:
        combo = lambda wc: wc in all_combo_folders
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        input_dir = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/{wildcards.combo}",
        output_dir = lambda wildcards: f"{ALLEL_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.combo}",
        err_file = lambda wildcards: f"{ALLEL_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.combo}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by single
        """