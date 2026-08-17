#!/usr/bin/env python3

import argparse
import io
import math
from pathlib import Path

import pandas as pd
from Bio import Phylo


def is_tree_line(line: str) -> bool:
    line = line.strip()

    return (
        "(" in line
        and ";" in line
        and not line.startswith("//")
        and not line.lower().startswith("segsites")
        and not line.lower().startswith("positions")
    )


def load_trees(clean_file: str | Path):
    trees = []

    with open(clean_file, "r") as f:
        for line in f:
            if is_tree_line(line):
                tree = Phylo.read(
                    io.StringIO(line.strip()),
                    "newick"
                )
                trees.append(tree)

    return trees


# ---------------------------------------------------------
# Features aus dem bisherigen Python-Skript
# ---------------------------------------------------------

def phylogenetic_diversity(tree):
    return sum(
        clade.branch_length or 0.0
        for clade in tree.find_clades()
    )


def blum_index(tree):
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


# ---------------------------------------------------------
# Features aus dem R-Skript
# ---------------------------------------------------------

def num_branches(tree):
    """
    Entspricht ungefähr:
        nrow(tr$edge)
    in ape.
    """

    return sum(
        1
        for clade in tree.find_clades(order="level")
        if clade is not tree.root
    )


def tree_height_unweighted(tree):
    """
    Entspricht:
        max(node.depth(tr_unweighted))

    wobei jede Kante Länge 1 hat.
    """

    max_depth = 0

    stack = [(tree.root, 1)]

    while stack:
        node, depth = stack.pop()

        max_depth = max(max_depth, depth)

        for child in node.clades:
            stack.append((child, depth + 1))

    return max_depth


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute tree structure features, "
            "phylogenetic diversity and Blum index."
        )
    )

    parser.add_argument(
        "--tree_file",
        type=Path,
        required=True,
        help="CSV containing tree features."
    )

    parser.add_argument(
        "--clean_file",
        type=Path,
        required=True,
        help="Simulation output containing Newick trees."
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output CSV."
    )

    args = parser.parse_args()

    features = pd.read_csv(args.tree_file)
    trees = load_trees(args.clean_file)

    if len(features) != len(trees):
        raise ValueError(
            f"Number of feature rows ({len(features)}) "
            f"does not match number of trees ({len(trees)})."
        )

    # R-Skript Features
    features["num_branches"] = [
        num_branches(tree)
        for tree in trees
    ]

    features["tree_height_uw"] = [
        tree_height_unweighted(tree)
        for tree in trees
    ]

    # Python-Skript Features
    features["phylogenetic_div"] = [
        phylogenetic_diversity(tree)
        for tree in trees
    ]

    features["blum"] = [
        blum_index(tree)
        for tree in trees
    ]

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        args.output_file,
        index=False
    )

    print(
        f"Successfully updated {len(trees)} trees.\n"
        f"Output written to:\n{args.output_file}"
    )


if __name__ == "__main__":
    main()