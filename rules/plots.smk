include: "../project_paths.py"
include: "../snakemake_vars.py"

rule get_merged_positions_stats:
    input:
        input_file = lambda w: f"{RAW_DATA_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}_output.txt",
        combo_yaml = lambda w: f"{RAW_DATA_DIR}/{w.config}/{w.combo}/{COMBO_PARAMETERS_YAML}"
    output:
        output_file=f"{MERGES_COLLECT_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_merge_stats.csv"
    conda:
        "../envs/plot-env.yaml"
    params:
        err_file = lambda w: f"{MERGES_COLLECT_DIR}/{w.config}/{w.combo}/rep{w.rep}/{w.simulator}.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/merged_positions_stats.py \
        --input {input.input_file} \
        --output {output.output_file} \
        --combo_yaml {input.combo_yaml} \
        --precision {PRECISION}
        """

rule collect_merged_positions_stats:
    input:
        lambda wildcards: get_merged_stats_overall_input(wildcards.config, wildcards.simulator)
    output:
        output_overall=f"{MERGES_COLLECT_DIR}/{{config}}/{{simulator}}_merged_positions_overall.csv",
        output_by_sample_size=f"{MERGES_COLLECT_DIR}/{{config}}/{{simulator}}_merged_positions_sample_size.csv"
    conda:
        "../envs/plot-env.yaml"
    threads: 4
    params:
        input_dir = lambda wildcards: f"{MERGES_COLLECT_DIR}/{wildcards.config}/",
        err_file = lambda wildcards: f"{MERGES_COLLECT_DIR}/{wildcards.config}/errorlog.err"
    shell:
        """
        scripts/run_with_errorlog.sh {params.err_file} \
        python src/plotting/aggregate_merge_stats.py \
        --input {params.input_dir} \
        --output_overall {output.output_overall} \
        --output_by_sample_size {output.output_by_sample_size} \
        --simulator {wildcards.simulator}
        """