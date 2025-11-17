include: "../snakemake_vars.py"

rule compute_boxplots_single_config_tree:
    input:
        lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREEFEATURES_BOXPLOTS_DIR}/{{config}}/.done"
    conda:
        "../envs/plot-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREE_FEATURES_DIR}/{wildcards.config} {TREEFEATURES_BOXPLOTS_DIR}/{wildcards.config}
        touch {output.dummy}
        """


rule compute_boxplots_single_config_allel:
    input:
        lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{ALLELFEATURES_BOXPLOTS_DIR}/{{config}}/.done"
    conda:
        "../envs/plot-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREE_FEATURES_DIR}/{wildcards.config} {ALLELFEATURES_BOXPLOTS_DIR}/{wildcards.config}
        touch {output.dummy}
        """


rule compute_boxplots_single_config_no_zero_branches_tree:
    input:
        lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREEFEATURES_ZEROB_BOXPLOTS_DIR}/{{config}}/.done"
    conda:
        "../envs/plot-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREE_FEATURES_DIR}/{wildcards.config} {TREEFEATURES_ZEROB_BOXPLOTS_DIR}/{wildcards.config} --filter_zero_branches
        touch {output.dummy}
        """


rule compute_histograms_single_config_no_zero_branches:
    input:
        lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"results/histograms/treestats_zero/{{config}}/.done"
    conda:
        "../envs/plot-env.yaml"
    shell:
        """
        python src/models/plot_histograms.py {TREE_FEATURES_DIR}/{wildcards.config} results/histograms/treestats_zero/{wildcards.config} --filter_zero_branches
        touch {output.dummy}
        """


rule compute_boxplots_altogether:
    input:
        expand(f"{TREE_FEATURES_DIR}/{{config}}/.done", config=all_configs)
    output:
        dummy=f"{TREEFEATURES_BOXPLOTS_DIR}/all/.done"
    conda:
        "../envs/plot-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREE_FEATURES_DIR} {TREEFEATURES_BOXPLOTS_DIR}/all
        touch {output.dummy}
        """