include: "../snakemake_vars.py"


rule compute_plots_tree_by_mode:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, SIMULATORS, wildcards.mode, "tree")
    output:
        dummy=f"{TREEFEATURES_PLOTS_DIR}/{{config}}/{{mode}}/.done"
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


rule compute_plots_tree_single:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, SIMULATORS, "single", "tree", wildcards.combo)
    output:
        dummy=f"{TREEFEATURES_PLOTS_DIR}/{{config}}/{{combo}}/.done"
    wildcard_constraints:
        combo = lambda wc: wc in all_combo_folders
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/{wildcards.combo}",
        output_dir = lambda wildcards: f"{TREEFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.combo}",
        err_file = lambda wildcards: f"{TREEFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.combo}/errorlog.err"
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


rule compute_plots_allel_single:
    input:
        lambda wildcards: get_topfeatures_input(wildcards.config, SIMULATORS, "single", "allel", wildcards.combo)
    output:
        dummy=f"{ALLELFEATURES_PLOTS_DIR}/{{config}}/{{combo}}/.done"
    wildcard_constraints:
        combo = lambda wc: wc in all_combo_folders
    conda:
        "../envs/plot-env.yaml"
    params:
        input_dir = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/{wildcards.combo}",
        output_dir = lambda wildcards: f"{ALLELFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.combo}",
        err_file = lambda wildcards: f"{ALLELFEATURES_PLOTS_DIR}/{wildcards.config}/{wildcards.combo}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/plot_boxplots.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --grouped_by {wildcards.mode} 
        touch {output.dummy}
        """