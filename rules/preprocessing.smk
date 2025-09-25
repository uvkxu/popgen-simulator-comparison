include: "../snakemake_vars.py"

rule compute_stats:
    input:
        msprime_in=f"{RAW_DATA_DIR}/{{config}}/msprime.txt",
        ms_in=f"{RAW_DATA_DIR}/{{config}}/ms.txt",
        msms_in=f"{RAW_DATA_DIR}/{{config}}/msms.txt"
    output:
        msprime_out=f"{TREESTATS_DATA_DIR}/{{config}}/msprime_stats.csv",
        ms_out=f"{TREESTATS_DATA_DIR}/{{config}}/ms_stats.csv",
        msms_out=f"{TREESTATS_DATA_DIR}/{{config}}/msms_stats.csv"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        Rscript -e 'if(!requireNamespace("treestats", quietly=TRUE)) devtools::install_github("thijsjanzen/treestats")'
        Rscript src/preprocessing/run_treestats_clean.R {RAW_DATA_DIR}/{wildcards.config} {TREESTATS_DATA_DIR}/{wildcards.config}
        """
