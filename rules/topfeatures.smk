include: "../snakemake_vars.py"

rule compute_top_features_single_config:
    input:
        directory(f"{TREESTATS_DATA_DIR}/{{config}}")
    output:
        directory(f"{TOPFEATURES_RESULTS_DIR}/{{config}}")
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREESTATS_DATA_DIR}/{wildcards.config} {TOPFEATURES_RESULTS_DIR}/{wildcards.config}
        """

rule compute_top_features_every_config:
    input:
        expand(f"{TOPFEATURES_RESULTS_DIR}/{{config}}", config=all_configs)

rule compute_top_features_altogether:
    input:
        directory(f"{TREESTATS_DATA_DIR}")
    output:
        directory(f"{TOPFEATURES_RESULTS_DIR}/all")
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python src/models/get_top_features.py {TREESTATS_DATA_DIR} {TOPFEATURES_RESULTS_DIR}/all
        """