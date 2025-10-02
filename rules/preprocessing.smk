include: "../snakemake_vars.py"

rule compute_treestats:
    input:
        directory(f"{RAW_DATA_DIR}/{{config}}")
    output:
        directory(f"{TREESTATS_DATA_DIR}/{{config}}")
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        Rscript -e 'if(!requireNamespace("treestats", quietly=TRUE)) devtools::install_github("thijsjanzen/treestats")'
        Rscript src/preprocessing/run_treestats_clean.R {RAW_DATA_DIR}/{wildcards.config} {TREESTATS_DATA_DIR}/{wildcards.config}
        """

