include: "../snakemake_vars.py"


    
rule compute_tree_check_Ne:
    output:
        f"{RESULTS_DIR}/{{simulator}}/tree/{{config}}/{{mode}}/check_eff_pop_size_{{simulator}}_summary.csv",
        f"{RESULTS_DIR}/{{simulator}}/tree/{{config}}/{{mode}}/check_eff_pop_size_{{simulator}}_plot.png",
    params:
        input_dir = lambda wildcards: f"{TREE_FEATURES_DIR}/{wildcards.config}/",
        output_dir = lambda wildcards: f"{RESULTS_DIR}/{wildcards.simulator}/tree/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{RESULTS_DIR}/{wildcards.simulator}/tree/{wildcards.config}/{wildcards.mode}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulator {wildcards.simulator} \
        --grouped_by {wildcards.mode}
        """


rule compute_allele_topfeatures_check_Ne:
    output:
        f"{RESULTS_DIR}/{{simulator}}/allele/{{config}}/{{mode}}/check_eff_pop_size_{{simulator}}_summary.csv",
        f"{RESULTS_DIR}/{{simulator}}/allele/{{config}}/{{mode}}/check_eff_pop_size_{{simulator}}_plot.png",
    params:
        input_dir = lambda wildcards: f"{ALLEL_FEATURES_DIR}/{wildcards.config}/",
        output_dir = lambda wildcards: f"{RESULTS_DIR}/{wildcards.simulator}/allele/{wildcards.config}/{wildcards.mode}",
        err_file = lambda wildcards: f"{RESULTS_DIR}/{wildcards.simulator}/allele/{wildcards.config}/{wildcards.mode}/errorlog.err"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/models/get_top_features.py \
        --input_folder {params.input_dir} \
        --output_folder {params.output_dir} \
        --simulator {wildcards.simulator} \
        --grouped_by {wildcards.mode}
        """