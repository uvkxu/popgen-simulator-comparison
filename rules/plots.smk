include: "../snakemake_vars.py"


rule compute_plots_tree_by_mode:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, SIMULATORS, wildcards.mode, "tree")
    output:
        dummy=f"{TREEFEATURES_PLOTS_DIR}/{{config}}/{{mode}}/.done"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_plot_features(f"{TREE_FEATURES_DIR}/{wildcards.config}/")
    wildcard_constraints:
        mode = "|".join(VALID_GROUPING_MODES)
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
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, SIMULATORS, wildcards.mode, "allel")
    output:
        dummy=f"{ALLELFEATURES_PLOTS_DIR}/{{config}}/{{mode}}/.done"
    resources:
        mem_mb=lambda wildcards, input: estimate_mem_mb_plot_features(f"{ALLEL_FEATURES_DIR}/{wildcards.config}/")
    wildcard_constraints:
        mode = "|".join(VALID_GROUPING_MODES)
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


rule compute_plots_pigot_rho:
    input:
        lambda wildcards: get_pigotrho_input(wildcards.simulator)
    output:
        dummy=f"{PLOTS_RESULTS_DIR}/pigotrho/{{simulator}}/.done"
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{PIGOT_RHO_DIR}/",
        output_dir = lambda wildcards: f"{PLOTS_RESULTS_DIR}/pigotrho/{wildcards.simulator}",
        err_file = lambda wildcards: f"{PIGOT_RHO_DIR}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/plot_pigot_rho.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulators {wildcards.simulator} 
        touch {output.dummy}
        """