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
        lambda wildcards: combo_yamls_tree(wildcards.config)
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
    input:
        llambda wildcards: combo_yamls_allel(wildcards.config)
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


rule aggregate_tree_topfeatures_over_sample_sizes:
    input:
        lambda wildcards: get_topfeatures_by_config(wildcards.config, wildcards.group_string, "tree")
    output:
        mean_csv = f"{TREE_TOPFEATURES_DIR}/{{config}}/aggregated/mean_aggregated_{{group_string}}_with_summary.csv",
        by_n_csv = f"{TREE_TOPFEATURES_DIR}/{{config}}/aggregated/all_features_{{group_string}}_by_sample_size.csv",
        plot = f"{TREE_TOPFEATURES_DIR}/{{config}}/aggregated/all_features_{{group_string}}_vs_sample_size.png",
    params:
        base_dir = lambda wc: f"{TREE_TOPFEATURES_DIR}/{wc.config}",
        out_dir = lambda wc: f"{TREE_TOPFEATURES_DIR}/{wc.config}/aggregated",
        err_file = lambda wc: f"{TREE_TOPFEATURES_DIR}/{wc.config}/aggregated/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/aggregate_all_features_by_sample_size.py \
            --base_dir {params.base_dir} \
            --group_string {wildcards.group_string} \
            --output_dir {params.out_dir}
        """


rule aggregate_allel_topfeatures_over_seq_len:
    input:
        lambda wildcards: get_topfeatures_by_config(wildcards.config, wildcards.group_string, "allel")
    output:
        mean_csv = f"{ALLEL_TOPFEATURES_DIR}/{{config}}/aggregated/mean_aggregated_{{group_string}}_with_summary.csv",
        by_n_csv = f"{ALLEL_TOPFEATURES_DIR}/{{config}}/aggregated/all_features_{{group_string}}_by_seq_len.csv",
        plot = f"{ALLEL_TOPFEATURES_DIR}/{{config}}/aggregated/all_features_{{group_string}}_vs_seq_len.png"
    params:
        base_dir = lambda wc: f"{ALLEL_TOPFEATURES_DIR}/{wc.config}",
        out_dir = lambda wc: f"{ALLEL_TOPFEATURES_DIR}/{wc.config}/aggregated",
        err_file = lambda wc: f"{ALLEL_TOPFEATURES_DIR}/{wc.config}/aggregated/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/aggregate_by_seq_len.py \
            --base_dir {params.base_dir} \
            --group {wildcards.group_string} \
            --output_dir {params.out_dir}
        """