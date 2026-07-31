#!/usr/bin/env python3

import argparse
import io
import math
from pathlib import Path

import pandas as pd
from Bio import Phylo


def is_tree_line(line: str) ->bool:
    """Return True if a line contains a Newick tree."""
    line = line.strip()

    return (
        "(" in line
        and ";" in line
        and not line.startswith("//")
        and not line.lower().startswith("segsites")
        and not line.lower().startswith("positions")
    )


def load_trees(clean_file: str | Path):
    """Extract all Newick trees from a clean simulation output file."""
    trees = []

    with open(clean_file, "r") as f:
        for line in f:
            if is_tree_line(line):
                trees.append(Phylo.read(io.StringIO(line.strip()), "newick"))

    return trees


def phylogenetic_diversity(tree):
    """Phylogenetic diversity = sum of all branch lengths."""
    return sum(
        clade.branch_length or 0.0
        for clade in tree.find_clades()
    )


def blum_index(tree):
    """
    Blum index of imbalance.

    Sum of log(N - 1) over all internal nodes,
    where N is the number of descendant tips.
    """
    score = 0.0

    def count_tips(clade):
        nonlocal score

        if clade.is_terminal():
            return 1

        n = sum(count_tips(child) for child in clade.clades)

        if n > 1:
            score += math.log(n - 1)

        return n

    count_tips(tree.root)

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Recompute phylogenetic diversity and Blum index."
    )
    parser.add_argument(
        "--tree_file",
        type=Path,
        required=True,
        help="Input CSV containing tree features.",
    )
    parser.add_argument(
        "--clean_file",
        type=Path,
        required=True,
        help="Clean simulation output containing Newick trees.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output CSV with updated tree features.",
    )

    args = parser.parse_args()

    features = pd.read_csv(args.tree_file)
    trees = load_trees(args.clean_file)

    if len(features) != len(trees):
        raise ValueError(
            f"Number of feature rows ({len(features)}) "
            f"does not match number of trees ({len(trees)})."
        )

    features["phylogenetic_div"] = [
        phylogenetic_diversity(tree)
        for tree in trees
    ]

    features["blum"] = [
        blum_index(tree)
        for tree in trees
    ]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()