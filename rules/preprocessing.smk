include: "../snakemake_vars.py"

rule compute_treestats:
    input:
        lambda wildcards: f"{RAW_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREESTATS_DATA_DIR}/{{config}}/.done"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        Rscript -e 'if(!requireNamespace("treestats", quietly=TRUE)) devtools::install_github("thijsjanzen/treestats")'
        Rscript src/preprocessing/run_treestats_clean.R {RAW_DATA_DIR}/{wildcards.config} {TREESTATS_DATA_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule compute_allelstats:
    input:
        lambda wildcards: f"{RAW_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{ALLELSTATS_DATA_DIR}/{{config}}/.done"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        python src/preprocessing/calc_all_allelstatistics.py {RAW_DATA_DIR}/{wildcards.config} {ALLELSTATS_DATA_DIR}/{wildcards.config}
        touch {output.dummy}
        """
