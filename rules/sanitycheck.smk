include: "../snakemake_vars.py"

rule sanity_check_one_config:
    input:
        directory(f"{TREESTATS_DATA_DIR}/{{config}}")
    output:
        f"{SANITY_RESULTS_DIR}/{{config}}/{FILE_FORMATS['sanity_check_file']}"
    conda:
        "../envs/models-env.yaml"
    shell:
        """
        python tests/sanity_check.py {TREESTATS_DATA_DIR}/{wildcards.config} {SANITY_RESULTS_DIR}
        """

rule sanity_check_all_configs:
    input:
        expand(
            f"{SANITY_RESULTS_DIR}/{{config}}/{FILE_FORMATS['sanity_check_file']}",
            config=all_configs
        )
