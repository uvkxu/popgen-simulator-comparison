include: "../snakemake_vars.py"


rule compute_plots_tree_by_mode:
    output:
        dummy=f"{TREEFEATURES_PLOTS_DIR}/{{config}}/{{mode}}/.done"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(input_folder=f"{TREE_FEATURES_DIR}/{wildcards.config}/", config=wildcards.config, group=SIMULATORS, mode=wildcards.mode)
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        output_dir = lambda wildcards: f"{TREEFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{TREEFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.mode}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/plot_boxplots.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --grouped_by {wildcards.mode}
        touch {output.dummy}
        """

rule compute_plots_allel_by_mode:
    output:
        dummy=f"{ALLELFEATURES_PLOTS_DIR}/{{config}}/{{mode}}/.done"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_classification(input_folder=f"{ALLEL_FEATURES_DIR}/{wildcards.config}/", config=wildcards.config, group=SIMULATORS, mode=wildcards.mode)
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        output_dir = lambda wildcards: f"{ALLELFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{ALLELFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.mode}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/plot_boxplots.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --grouped_by {wildcards.mode}
        touch {output.dummy}
        """

rule collect_delta_simulators_tree:
    output:
        output_file=f"{TREEFEATURES_COLLECT_DIR}/{{config}}/simulator_feature_values.csv"
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        err_file = lambda wildcards: f"{TREEFEATURES_COLLECT_DIR}/{wildcards.config}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/collect_simulator_feature_values.py \
        --input {params.input_dir} \
        --out {output.output_file}
        """

rule collect_delta_simulators_allel:
    output:
        output_file=f"{ALLELFEATURES_COLLECT_DIR}/{{config}}/simulator_feature_values.csv"
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        err_file = lambda wildcards: f"{ALLELFEATURES_COLLECT_DIR}/{wildcards.config}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/collect_simulator_feature_values.py \
        --input {params.input_dir} \
        --out {output.output_file}
        """

rule get_merged_positions_stats:
    output:
        output_file=f"{MERGES_COLLECT_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_merge_stats.csv"
    conda:
        "../envs/plot-env.yaml"
    params:
        input = lambda w: f"{RAW_DATA_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_cleaned_output.txt",
        err_file = lambda w: f"{MERGES_COLLECT_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err",
        combo_yaml = lambda w: f"{RAW_DATA_DIR}/{w.config}/{w.combo}/{COMBO_PARAMETERS_YAML}"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/merged_positions_stats.py \
        --input {params.input} \
        --output {output.output_file} \
        --combo_yaml {params.combo_yaml} \
        --precision {PRECISION}
        """

rule collect_merged_positions_stats:
    output:
        output_overall=f"{ALLELFEATURES_COLLECT_DIR}/{{config}}/{{simulator}}_merged_positions_overall.csv",
        output_by_sample_size=f"{ALLELFEATURES_COLLECT_DIR}/{{config}}/{{simulator}}_merged_positions_sample_size.csv"
    conda:
        "../envs/plot-env.yaml"
    threads: 4
    params:
        input_dir = lambda wildcards: f"{MERGES_COLLECT_DIR}/{wildcards.config}/",
        err_file = lambda wildcards: f"{ALLELFEATURES_COLLECT_DIR}/{wildcards.config}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/aggregate_merge_stats.py \
        --input {params.input_dir} \
        --output_overall {output.output_overall} \
        --output_by_sample_size {output.output_by_sample_size} \
        --simulator {wildcards.simulator}
        """