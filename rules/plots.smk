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
        --out {params.output_file}
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
        --out {params.output_file}
        """