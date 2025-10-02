include: "../snakemake_vars.py"

rule compute_boxplots_single_config:
    input:
        directory(f"{TREESTATS_DATA_DIR}/{{config}}")
    output:
        directory(f"{BOXPLOTS_RESULTS_DIR}/{{config}}")
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREESTATS_DATA_DIR}/{wildcards.config} {BOXPLOTS_RESULTS_DIR}/{wildcards.config}
        """

rule compute_boxplots_every_config:
    input:
        expand(f"{BOXPLOTS_RESULTS_DIR}/{{config}}", config=all_configs)

rule compute_boxplots_altogether:
    input:
        directory(f"{TREESTATS_DATA_DIR}")
    output:
        directory(f"{BOXPLOTS_RESULTS_DIR}/all")
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREESTATS_DATA_DIR} {BOXPLOTS_RESULTS_DIR}/all
        """