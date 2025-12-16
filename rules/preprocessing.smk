include: "../snakemake_vars.py"


rule clean_single_file:
    input:
        raw = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        python src/preprocessing/clean.py \
            --input {input.raw} \
            --output {output.cleaned} \
            --combo_yaml {input.combo_yaml} \
            --sim {wildcards.simulator}
        """
    
rule compute_allel_features_single_file:
    input:
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        allelstats = f"{ALLEL_FEATURES_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_allel_features.csv"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        python src/preprocessing/calc_allelstatistics.py \
            --input {input.cleaned} \
            --output {output.allelstats} \
            --combo_yaml {input.combo_yaml}
        """





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