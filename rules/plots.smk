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

rule compute_forward_backward_delta_per_config_tree:
    output:
        output_file=f"{TREEFEATURES_PLOTS_DIR}/{{config}}/deltas.csv"
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        err_file = lambda wildcards: f"{TREEFEATURES_PLOTS_DIR}/{wildcards.config}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/compute_forward_backward_deltas.py \
        --input_dir {params.input_dir} \
        --output_file {output.output_file} \
        --row_level
        """

rule compute_forward_backward_delta_per_config_allele:
    output:
        output_file=f"{ALLELFEATURES_PLOTS_DIR}/{{config}}/deltas.csv"
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        err_file = lambda wildcards: f"{ALLELFEATURES_PLOTS_DIR}/{wildcards.config}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/compute_forward_backward_deltas.py \
        --input_dir {params.input_dir} \
        --output_file {output.output_file} \
        --row_level
        """

rule collect_non_binary_features_per_config:
    input:
        input_dir = lambda wc: f"{TREE_FEATURES_DIR}/{wc.config}/"
    output:
        out_csv = f"{TREEFEATURES_PLOTS_DIR}/{{config}}/non_binary_features.csv"
    conda:
        "../envs/plot-env.yaml"
    params:
        err_file = lambda wc: f"{TREEFEATURES_PLOTS_DIR}/{wc.config}/non_binary_features_errorlog.err",
        # falls du lieber ALLE Tree-Feature-CSVs willst statt nur SLiM:
        pattern = "slim_tree_features.csv"
        # pattern = "*tree_features*.csv"
    shell:
        r"""
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/collect_non_binary_features.py \
            {input.input_dir} \
            {output.out_csv} \
            "{params.pattern}"
        """