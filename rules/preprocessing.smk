include: "../snakemake_vars.py"

rule compute_treestats_config:
    input:
        lambda wildcards: f"{CLEAN_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{TREE_FEATURES_DIR}/{{config}}/.done"
    log:
        out=f"{TREE_FEATURES_DIR}/{{config}}/run.out"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        mkdir -p $(dirname {log.out})
        tmp_err=$(mktemp)

        # Install treestats if not installed (stdout+stderr in run.out)
        Rscript -e 'if(!requireNamespace("treestats", quietly=TRUE)) devtools::install_github("thijsjanzen/treestats")' \
            > {log.out} 2>> $tmp_err || {{ 
                mv $tmp_err {TREE_FEATURES_DIR}/{wildcards.config}/run.err
                exit 1
            }}

        # Run the actual calculation
        Rscript src/preprocessing/calc_all_treestatistics.R {CLEAN_DIR}/{wildcards.config} {TREE_FEATURES_DIR}/{wildcards.config} \
            >> {log.out} 2>> $tmp_err || {{ 
                mv $tmp_err {TREE_FEATURES_DIR}/{wildcards.config}/run.err
                exit 1
            }}

        rm -f $tmp_err
        touch {output.dummy}
        """

rule compute_allelstats_config:
    input:
        lambda wildcards: f"{CLEAN_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{ALLEL_FEATURES_DIR}/{{config}}/.done"
    log:
        out=f"{ALLEL_FEATURES_DIR}/{{config}}/run.out"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        mkdir -p $(dirname {log.out})
        tmp_err=$(mktemp)
        python src/preprocessing/calc_all_allelstatistics.py {CLEAN_DIR}/{wildcards.config} {ALLEL_FEATURES_DIR}/{wildcards.config} \
            > {log.out} 2> $tmp_err || {{ 
                mv $tmp_err {ALLEL_FEATURES_DIR}/{wildcards.config}/run.err; 
                exit 1; 
            }}
        rm -f $tmp_err
        touch {output.dummy}
        """

rule clean_raw_data:
    input:
        lambda wildcards: f"{RAW_DATA_DIR}/{wildcards.config}/.done"
    output:
        dummy=f"{CLEAN_DIR}/{{config}}/.done"
    log:
        out=f"{CLEAN_DIR}/{{config}}/run.out"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        mkdir -p $(dirname {log.out})
        tmp_err=$(mktemp)
        python src/preprocessing/clean_all.py --input {RAW_DATA_DIR}/{wildcards.config} --output {CLEAN_DIR}/{wildcards.config} \
            > {log.out} 2> $tmp_err || {{ 
                mv $tmp_err {CLEAN_DIR}/{wildcards.config}/run.err; 
                exit 1; 
            }}
        rm -f $tmp_err
        touch {output.dummy}
        """