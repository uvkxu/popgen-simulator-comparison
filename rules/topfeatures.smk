include: "../project_paths.py"
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
        shutil.copyfile(input.combo_yaml, output.copy_file)


rule copy_combo_yaml_allel_features:
    input:
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        copy_file = f"{ALLEL_FEATURES_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    run:
        from pathlib import Path
        import shutil

        Path(output.copy_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input.combo_yaml, output.copy_file)

    
rule compute_tree_topfeatures_grouped_by_mode:
    input:
        lambda wildcards: get_evaluation_input(wildcards.config, wildcards.mode, "tree", wildcards.group_string)
    output:
        f"{TREE_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv"
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
        "../envs/evaluation-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/evaluation/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """


rule compute_allel_topfeatures_grouped_by_mode:
    input:
        lambda wildcards: get_evaluation_input(wildcards.config, wildcards.mode, "allel", wildcards.group_string)
    output:
        f"{ALLEL_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv"
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
        "../envs/evaluation-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/evaluation/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """

rule compute_together_topfeatures_grouped_by_mode:
    input:
        lambda wildcards: get_evaluation_input(
            wildcards.config, wildcards.mode, "tree", wildcards.group_string
        ),
        lambda wildcards: get_evaluation_input(
            wildcards.config, wildcards.mode, "allel", wildcards.group_string
        )
    output:
        f"{TOGETHER_TOPFEATURES_DIR}/{{config}}/{{mode}}/feature_ranking_{{group_string}}_with_summary.csv"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(input_folder=f"{ALLEL_FEATURES_DIR}/{wildcards.config}/", config=wildcards.config, group=wildcards.group_string, mode=wildcards.mode)
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        input_dir_allel = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        input_dir_tree = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        output_dir = lambda wildcards: f"{TOGETHER_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{TOGETHER_TOPFEATURES_DIR}/{wildcards.config}/{wildcards.mode}/errorlog.err"
    conda:
        "../envs/evaluation-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/evaluation/get_top_features_together.py \
        --input_folder_tree {params.input_dir_tree} \
        --input_folder_allel {params.input_dir_allel} \
        --output_folder {params.output_dir} \
        --simulators {params.args_group} \
        --grouped_by {wildcards.mode}
        """


#---------------Statistical Tests-----------------

rule pairwise_statistical_test:
    input:
        lambda wildcards: get_evaluation_input(
            wildcards.config, wildcards.mode, "tree", wildcards.group_string
        ),
        lambda wildcards: get_evaluation_input(
            wildcards.config, wildcards.mode, "allel", wildcards.group_string
        )
    output:
        output_file = f"{STATISTICS_RESULTS_DIR}/{{config}}/{{mode}}/{{group_string}}_feature_comparison.csv"
    params:
        args_group=lambda wildcards: ",".join(
            wildcards.group_string.split("_")
        ),
        input_dir_allel=lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        input_dir_tree=lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        err_file=lambda wildcards: f"{STATISTICS_RESULTS_DIR}/{wildcards.config}/{wildcards.mode}/{wildcards.group_string}.err"
    conda:
        "../envs/stats-test-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/evaluation/statistical_tests.py \
        --input_folder_tree {params.input_dir_tree} \
        --input_folder_allel {params.input_dir_allel} \
        --output_file {output.output_file} \
        --grouped_by {wildcards.mode} \
        --simulators {params.args_group}
        """
    