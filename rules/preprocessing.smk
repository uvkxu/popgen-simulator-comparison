include: "../snakemake_vars.py"

rule compute_treestats_config:
    input:
        lambda wildcards: f"{CLEAN_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREE_FEATURES_DIR}/{{config}}/.done"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        Rscript -e 'if(!requireNamespace("treestats", quietly=TRUE)) devtools::install_github("thijsjanzen/treestats")'
        Rscript src/preprocessing/calc_all_treestatistics.R {CLEAN_DIR}/{wildcards.config} {TREE_FEATURES_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule compute_allelstats_config:
    input:
        lambda wildcards: f"{CLEAN_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{ALLEL_FEATURES_DIR}/{{config}}/.done"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        python src/preprocessing/calc_all_allelstatistics.py {CLEAN_DIR}/{wildcards.config} {ALLEL_FEATURES_DIR}/{wildcards.config}
        touch {output.dummy}
        """

rule clean_raw_data:
    input:
        lambda wildcards: f"{RAW_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{CLEAN_DIR}/{{config}}/.done"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        python src/preprocessing/clean_all.py --input {RAW_DATA_DIR}/{wildcards.config} --output {CLEAN_DIR}/{wildcards.config}
        touch {output.dummy}
        """