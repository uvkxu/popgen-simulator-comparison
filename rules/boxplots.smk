include: "../snakemake_vars.py"

rule compute_boxplots_single_config:
    input:
        lambda wildcards: f"{TREESTATS_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{BOXPLOTS_RESULTS_DIR}/{{config}}/.done"
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREESTATS_DATA_DIR}/{wildcards.config} {BOXPLOTS_RESULTS_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule compute_boxplots_every_config:
    input:
        expand(f"{BOXPLOTS_RESULTS_DIR}/{{config}}/.done", config=all_configs)

rule compute_boxplots_altogether:
    input:
        expand(f"{TREESTATS_DATA_DIR}/{{config}}/.done", config=all_configs)
    output:
        dummy=f"{BOXPLOTS_RESULTS_DIR}/all/.done"
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREESTATS_DATA_DIR} {BOXPLOTS_RESULTS_DIR}/all
        touch {output.dummy}
        """