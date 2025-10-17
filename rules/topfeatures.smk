include: "../snakemake_vars.py"

rule compute_top_features_single_config:
    input:
        lambda wildcards: f"{TREESTATS_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TOPFEATURES_RESULTS_DIR}/{{config}}/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREESTATS_DATA_DIR}/{wildcards.config} {TOPFEATURES_RESULTS_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule compute_top_features_altogether:
    input:
        expand(f"{TREESTATS_DATA_DIR}/{{config}}/.done", config=all_configs)
    output:
        dummy = f"{TOPFEATURES_RESULTS_DIR}/all/.done"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREESTATS_DATA_DIR} {TOPFEATURES_RESULTS_DIR}/all
        touch {output.dummy}
        """