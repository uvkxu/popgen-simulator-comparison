from pathlib import Path
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing


def process_file(file):
    try:
        df = pd.read_csv(file)

        if len(df) == 0:
            return None

        return {
            "file": str(file),
            "simulator": file.name.replace("_tree_check.csv", ""),

            "n_trees": len(df),

            "n_wrong_leaves":
                int((~df["leaves_ok"]).sum()),

            "n_branch_mismatch":
                int((df["branch_difference"] != 0).sum()),

            "n_unary_trees":
                int((df["n_unary_nodes"] > 0).sum()),

            "n_polytomies":
                int((df["n_multifurcations"] > 0).sum()),

            # -------- neue Checks --------

            "n_pd_branch_mismatch":
                int((df["observed_branches"] !=
                    df["observed_branches_pd"]).sum()),

            "n_height_mismatch":
                int((df["tree_height_uw"] !=
                    df["max_depth_ape"]).sum()),

            "max_unary_nodes":
                int(df["n_unary_nodes"].max()),

            "max_branch_difference":
                int(df["branch_difference"].max()),

            "min_branch_difference":
                int(df["branch_difference"].min()),

            "max_height_difference":
                float(df["height_difference"].max()),

            "min_height_difference":
                float(df["height_difference"].min()),

            "max_pd_branch_difference":
                float(df["branch_pd_difference"].max()),

            "min_pd_branch_difference":
                float(df["branch_pd_difference"].min()),
        }

    except Exception as e:
        return {
            "file": str(file),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of worker processes.",
    )

    args = parser.parse_args()

    files = list(
        args.input_dir.rglob("*_tree_check.csv")
    )

    print(f"Found {len(files)} tree_check files.")
    print(f"Using {args.n_jobs} workers.")

    rows = []

    with ProcessPoolExecutor(
        max_workers=args.n_jobs
    ) as executor:

        for res in tqdm(
            executor.map(process_file, files),
            total=len(files),
            desc="Processing files",
        ):
            if res is not None:
                rows.append(res)

    summary = pd.DataFrame(rows)

    if len(summary) == 0:
        raise ValueError("No valid tree_check files found.")

    sort_cols = [
        c for c in [
            "n_branch_mismatch",
            "n_unary_trees",
            "n_wrong_leaves",
        ]
        if c in summary.columns
    ]

    if sort_cols:
        summary = summary.sort_values(
            sort_cols,
            ascending=False,
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.output,
        index=False,
    )

    print()
    print("Summary:")
    print(
        "Files with branch mismatch:",
        (summary["n_branch_mismatch"] > 0).sum()
        if "n_branch_mismatch" in summary.columns
        else 0,
    )
    print(
        "Files with unary nodes:",
        (summary["n_unary_trees"] > 0).sum()
        if "n_unary_trees" in summary.columns
        else 0,
    )
    print(
        "Files with wrong leaf count:",
        (summary["n_wrong_leaves"] > 0).sum()
        if "n_wrong_leaves" in summary.columns
        else 0,
    )
    print(
        "Files with polytomies:",
        (summary["n_polytomies"] > 0).sum()
        if "n_polytomies" in summary.columns
        else 0,
    )
    print(
    "Files with PD branch mismatch:",
    (summary["n_pd_branch_mismatch"] > 0).sum()
    )

    print(
        "Files with height mismatch:",
        (summary["n_height_mismatch"] > 0).sum()
    )


if __name__ == "__main__":
    main()