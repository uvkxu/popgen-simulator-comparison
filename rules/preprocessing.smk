include: "../snakemake_vars.py"

rule compute_treestats_config:
    input:
        lambda wildcards: f"{RAW_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREESTATS_DATA_DIR}/{{config}}/.done"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        Rscript -e 'if(!requireNamespace("treestats", quietly=TRUE)) devtools::install_github("thijsjanzen/treestats")'
        Rscript src/preprocessing/calc_all_treestatistics.R {RAW_DATA_DIR}/{wildcards.config} {TREESTATS_DATA_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule compute_treestats_cmds:
    output:
        dummy=f"{TREESTATS_DATA_DIR}/commands/.done"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        Rscript -e 'if(!requireNamespace("treestats", quietly=TRUE)) devtools::install_github("thijsjanzen/treestats")' \
        && Rscript src/preprocessing/calc_all_treestatistics.R {RAW_DATA_DIR}/commands {TREESTATS_DATA_DIR}/commands \
        && touch {output.dummy}
        """

rule compute_allelstats_config:
    input:
        lambda wildcards: f"{RAW_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{ALLELSTATS_DATA_DIR}/{{config}}/.done"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        python src/preprocessing/calc_all_allelstatistics.py {RAW_DATA_DIR}/{wildcards.config} {ALLELSTATS_DATA_DIR}/{wildcards.config} --input_config {CONFIG_DIR}/{wildcards.config}.yaml
        touch {output.dummy}
        """

rule compute_allelstats_commands:
    output:
        dummy=f"{ALLELSTATS_DATA_DIR}/commands/.done"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        python src/preprocessing/calc_all_allelstatistics.py {RAW_DATA_DIR}/commands {ALLELSTATS_DATA_DIR}/commands
        touch {output.dummy}
        """
