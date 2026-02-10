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
        lambda wildcards: get_topfeatures_input(config = wildcards.config, mode = wildcards.mode, feature_type = "tree", group = wildcards.group_string)
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
        lambda wildcards: get_topfeatures_input(config = wildcards.config, mode = wildcards.mode, feature_type = "allel", group = wildcards.group_string)
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



rule aggregate_topfeatures_global:
    """
    Aggregates *all* topfeature summary CSVs (tree+allele) into a few global CSVs.
    No config/mode mixing: keys keep (feature_type, model=config, config=mode, sample_size, seq_len).
    """
    input:
        list_all_topfeature_summary_csv
    output:
        overall = str(AGG_OVERALL),
        feat    = str(AGG_FEAT),
        passed  = str(AGG_PASSED),
        index   = str(AGG_INDEX)
    params:
        input_root = str(TOPFEATURES_RESULTS_DIR),
        outdir     = str(TOPFEATURES_AGG_GLOBAL_DIR),
        default_chance = 1.0/3.0
    conda:
        "envs/analysis.yaml"
    shell:
        r"""
        python scripts/aggregate_topfeatures.py \
          --input-root "{params.input_root}" \
          --outdir "{params.outdir}" \
          --default-chance "{params.default_chance}"
        """


rule plot_topfeatures_global:
    """
    Produces the plots from the global aggregated CSVs.
    """
    input:
        overall = str(AGG_OVERALL),
        feat    = str(AGG_FEAT),
        passed  = str(AGG_PASSED),
        index   = str(AGG_INDEX)
    output:
        # Use a directory output so we don't have to list every file.
        outdir = directory(str(TOPFEATURES_AGG_GLOBAL_PLOTS_DIR))
    params:
        agg_dir = str(TOPFEATURES_AGG_GLOBAL_DIR),
        outdir  = str(TOPFEATURES_AGG_GLOBAL_PLOTS_DIR),
        demes_root = str(CONFIG_DIR),     # <— wichtig: deine demes-specs liegen in stdpopsim-config
        topk = 30
    conda:
        "envs/analysis.yaml"
    run:
        cmd = [
            "python", "scripts/plot_topfeatures.py",
            "--agg-dir", params.agg_dir,
            "--outdir", params.outdir,
            "--topk-features", str(params.topk),
        ]

        # If you want demes-based plots, pass the demes root. In your case: CONFIG_DIR
        # (because your demes YAMLs live in stdpopsim-config/*.yaml)
        if params.demes_root not in [None, "", "null"]:
            cmd += ["--demes-root", params.demes_root]

        shell(" ".join(cmd))


rule topfeatures_global:
    """
    Convenience meta-target
    """
    input:
        str(AGG_OVERALL),
        directory(str(TOPFEATURES_AGG_GLOBAL_PLOTS_DIR))
