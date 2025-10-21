include: "../snakemake_vars.py"

rule compute_top_features_treestats_single_config:
    input:
        lambda wildcards: f"{TREESTATS_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREESTATS_TOPFEATURES_DIR}/{{config}}/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREESTATS_DATA_DIR}/{wildcards.config} {TREESTATS_TOPFEATURES_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule compute_top_features_treestats_altogether:
    input:
        expand(f"{TREESTATS_DATA_DIR}/{{config}}/.done", config=all_configs)
    output:
        dummy = f"{TREESTATS_TOPFEATURES_DIR}/all/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREESTATS_DATA_DIR} {TREESTATS_TOPFEATURES_DIR}/all
        touch {output.dummy}
        """

rule compute_top_features_allelstats_single_config:
    input:
        lambda wildcards: f"{ALLELSTATS_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{ALLELSTATS_TOPFEATURES_DIR}/{{config}}/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {ALLELSTATS_DATA_DIR}/{wildcards.config} {ALLELSTATS_TOPFEATURES_DIR}/{wildcards.config}
        touch {output.dummy}
        """
