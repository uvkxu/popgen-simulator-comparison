include: "../project_paths.py"
include: "../snakemake_vars.py"


rule clean_single_file:
    input:
        raw = f"{DATA_DIR}/{{scenario}}/raw/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt",
        combo_yaml = f"{DATA_DIR}/{{scenario}}/raw/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        cleaned = f"{DATA_DIR}/{{scenario}}/cleaned/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt"
    params:
        err_file= lambda w: f"{DATA_DIR}/{w.scenario}/cleaned/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
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
        cleaned = f"{DATA_DIR}/{{scenario}}/cleaned/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{DATA_DIR}/{{scenario}}/raw/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        allelstats = f"{DATA_DIR}/{{scenario}}/features/allele/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_allel_features.csv"
    params:
        err_file= lambda w: f"{DATA_DIR}/{w.scenario}/features/allele/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
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
        cleaned = f"{DATA_DIR}/{{scenario}}/cleaned/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{DATA_DIR}/{{scenario}}/raw/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        treestats = f"{DATA_DIR}/{{scenario}}/features/tree/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_tree_features.csv"
    params:
        err_file = lambda w: f"{DATA_DIR}/{w.scenario}/features/tree/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
    conda:
        "../envs/treestats-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        Rscript src/preprocessing/calc_treestatistics.R \
            {input.cleaned} {output.treestats} {input.combo_yaml}
        """