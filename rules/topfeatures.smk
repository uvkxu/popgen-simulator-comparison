include: "../snakemake_vars.py"

rule compute_top_features_treestats_single_config:
    input:
        lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREE_TOPFEATURES_DIR}/{{config}}/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREE_FEATURES_DIR}/{wildcards.config} {TREE_TOPFEATURES_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule compute_top_features_treestats_single_config_no_zero_branches:
    input:
        lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"results/top_features/treestats_zero/{{config}}/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREE_FEATURES_DIR}/{wildcards.config} results/top_features/treestats_zero/{wildcards.config} --filter_zero_branches --resample=under
        touch {output.dummy}
        """

rule compute_top_features_treestats_altogether:
    input:
        expand(f"{TREE_FEATURES_DIR}/{{config}}/.done", config=all_configs)
    output:
        dummy = f"{TREE_TOPFEATURES_DIR}/all/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREE_FEATURES_DIR} {TREE_TOPFEATURES_DIR}/all
        touch {output.dummy}
        """

rule compute_top_features_allelstats_single_config:
    input:
        lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{ALLEL_TOPFEATURES_DIR}/{{config}}/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {ALLEL_FEATURES_DIR}/{wildcards.config} {ALLEL_TOPFEATURES_DIR}/{wildcards.config}
        touch {output.dummy}
        """
