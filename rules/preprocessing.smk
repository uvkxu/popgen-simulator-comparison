include: "../snakemake_vars.py"


rule clean_single_file:
    input:
        raw = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt"
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
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        allelstats = f"{ALLEL_FEATURES_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_allel_features.csv"
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
        cleaned = f"{CLEAN_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_cleaned_output.txt",
        combo_yaml = f"{RAW_DATA_DIR}/{{config}}/{{combo}}/{COMBO_PARAMETERS_YAML}"
    output:
        treestats = f"{TREE_FEATURES_DIR}/{{config}}/{{combo}}/rep{{rep}}/{{simulator}}_tree_features.csv"
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


rule find_inconsistent_csv:
    input:
        csv=lambda wc: CSV_MAP[wc.id]
    output:
        log="logs/{id}.log"
    params:
        delimiter=","
    shell:
        """
        python src/find_inconsistent_csv.py \
            --input_csv {input.csv} \
            --log {output.log} \
            --delimiter {params.delimiter}
        """


rule aggregate_csv_logs:
    input:
        expand("logs/{id}.log", id=CSV_MAP.keys())
    output:
        "logs/summary.tsv"
    run:
        import pathlib

        with open(output[0], "w") as out:
            out.write("id\tpath\tstatus\terror\tdetails\n")

            for log_file in input:
                csv_id = pathlib.Path(log_file).stem
                csv_path = CSV_MAP[csv_id]

                lines = pathlib.Path(log_file).read_text().splitlines()

                if not any(l == "OK" for l in lines):
                    error = next(l for l in lines if l.startswith("ERROR"))
                    details = ";".join(l for l in lines if l.startswith("DETAIL"))
                    out.write(
                        f"{csv_id}\t{csv_path}\tBROKEN\t{error}\t{details}\n"
                    )
