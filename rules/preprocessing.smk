include: "../snakemake_vars.py"


rule clean_single_file:
    input:
        raw = ancient(f"{RAW_DATA_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt"),
        combo_yaml = ancient(f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}")
    output:
        cleaned = protected(f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt")
    params:
        err_file= lambda w: f"{CLEAN_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/preprocessing/clean.py \
            --input {input.raw} \
            --output {output.cleaned} \
            --combo_yaml {input.combo_yaml} \
            --sim {wildcards.simulator}
        """
    
rule compute_allel_features_single_file:
    input:
        cleaned = ancient(f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt"),
        combo_yaml = ancient(f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}")
    output:
        allelstats = protected(f"{ALLEL_FEATURES_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_allel_features.csv")
    params:
        err_file= lambda w: f"{ALLEL_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
    resources:
        mem_mb = lambda wildcards, attempt: estimate_mem_mb(
            config=wildcards.config,
            combo=wildcards.combo,
            func="calc_allelstatistics"
        )
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/preprocessing/calc_allelstatistics.py \
            --input {input.cleaned} \
            --output {output.allelstats} \
            --combo_yaml {input.combo_yaml}
        """


rule compute_tree_features_single_file:
    input:
        cleaned = ancient(f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt"),
        combo_yaml = ancient(f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}")
    output:
        treestats = protected(f"{TREE_FEATURES_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_tree_features.csv")
    params:
        err_file = lambda w: f"{TREE_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        Rscript src/preprocessing/calc_treestatistics.R \
            {input.cleaned} {output.treestats} {input.combo_yaml}
        """


rule compute_pigot_rho_stats_ms:
    output:
        pigotrhostats = f"{PIGOT_RHO_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_pigotrho_features.csv"
    params:
        err_file = lambda w: f"{PIGOT_RHO_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err",
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        Rscript src/preprocessing/check_ultrametric.R \
            {params.cleaned} {output.pigotrhostats} {params.combo_yaml}
        """
    


