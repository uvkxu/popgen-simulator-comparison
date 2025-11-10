include: "../snakemake_vars.py"

rule compute_boxplots_single_config:
    input:
        lambda wildcards: f"{TREESTATS_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREESTATS_BOXPLOTS_DIR}/{{config}}/.done"
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREESTATS_DATA_DIR}/{wildcards.config} {TREESTATS_BOXPLOTS_DIR}/{wildcards.config}
        touch {output.dummy}
        """


rule compute_boxplots_single_config_no_zero_branches:
    input:
        lambda wildcards: f"{TREESTATS_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"results/boxplots/treestats_zero/{{config}}/.done"
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREESTATS_DATA_DIR}/{wildcards.config} results/boxplots/treestats_zero/{wildcards.config} --filter_zero_branches
        touch {output.dummy}
        """


rule compute_histograms_single_config_no_zero_branches:
    input:
        lambda wildcards: f"{TREESTATS_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"results/histograms/treestats_zero/{{config}}/.done"
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_histograms.py {TREESTATS_DATA_DIR}/{wildcards.config} results/histograms/treestats_zero/{wildcards.config} --filter_zero_branches
        touch {output.dummy}
        """


rule compute_boxplots_altogether:
    input:
        expand(f"{TREESTATS_DATA_DIR}/{{config}}/.done", config=all_configs)
    output:
        dummy=f"{TREESTATS_BOXPLOTS_DIR}/all/.done"
    conda:
        "../envs/treestats-boxplots-env.yaml"
    shell:
        """
        python src/models/plot_boxplots.py {TREESTATS_DATA_DIR} {TREESTATS_BOXPLOTS_DIR}/all
        touch {output.dummy}
        """