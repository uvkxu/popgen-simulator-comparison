include: "../snakemake_vars.py"


rule clean_single_file:
    input:
        raw = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt"
    params:
        err_file= lambda w: f"{CLEAN_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err",
        copy_file= lambda w: f"{COPY_CLEAN_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_cleaned_output.txt"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/preprocessing/clean.py \
            --input {input.raw} \
            --output {output.cleaned} \
            --combo_yaml {input.combo_yaml} \
            --sim {wildcards.simulator} \
            --copy_file {params.copy_file}
        """
    
rule compute_allel_features_single_file:
    input:
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        allelstats = f"{ALLEL_FEATURES_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_allel_features.csv"
    params:
        err_file= lambda w: f"{ALLEL_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err",
        copy_file= lambda w: f"{COPY_ALLEL_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_allel_features.csv"
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
            --combo_yaml {input.combo_yaml} \
            --copy_file {params.copy_file}
        """


rule compute_tree_features_single_file:
    input:
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        treestats = f"{TREE_FEATURES_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_tree_features.csv"
    params:
        err_file = lambda w: f"{TREE_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err",
        copy_tree_file= lambda w: f"{COPY_TREE_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_tree_features.csv",
        copy_clean_file= lambda w: f"{COPY_CLEAN_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_cleaned_output.txt"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        Rscript src/preprocessing/calc_treestatistics.R \
            {input.cleaned} {output.treestats} {input.combo_yaml} {params.copy_tree_file} {params.copy_clean_file}
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



rule check_features_tree:
    output:
        report = f"{RESULTS_DIR}/check_tree_features/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_report.txt"
    params:
        err_file = lambda w: f"{RESULTS_DIR}/check_tree_features/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err",
        feature_file= lambda w: f"{TREE_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_tree_features.csv",
        clean_file= lambda w: f"{CLEAN_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_cleaned_output.txt"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/preprocessing/check_tree_file.py \
            --tree_file {params.feature_file} \
            --clean_file {params.clean_file} \
            --output_file {output.report} \
        """


rule fix_blum_phylo_tree:
    output:
        patched_file = f"{PATCHED_TREE_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_tree_features.csv"
    params:
        err_file = lambda w: f"{PATCHED_TREE_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err",
        feature_file= lambda w: f"{TREE_FEATURES_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_tree_features.csv",
        clean_file= lambda w: f"{CLEAN_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_cleaned_output.txt"
    conda:
        "../envs/allelstats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/preprocessing/replace_blum_phylogenetic_div.py \
            --tree_file {params.feature_file} \
            --clean_file {params.clean_file} \
            --output_file {output.patched_file} \
        """
