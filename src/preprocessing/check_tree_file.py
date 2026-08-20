#!/usr/bin/env python3

import argparse
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo


def is_tree_line(line: str) -> bool:
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
                tree = Phylo.read(io.StringIO(line.strip()), "newick")
                trees.append(tree)

    return trees


def phylogenetic_diversity(tree):
    """
    Phylogenetic diversity = sum of all branch lengths.
    """
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


def eigen_centrality(tree):
    """
    Compute the maximum eigenvector centrality of a Bio.Phylo tree.

    Corresponds to:
    max(treestats::eigen_centrality(tree, weight = FALSE)$eigenvector)
    """

    nodes = list(tree.find_clades(order="level"))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    A = np.zeros((n, n), dtype=float)

    # unweighted adjacency matrix
    for parent in tree.find_clades(order="level"):
        i = node_to_idx[parent]

        for child in parent.clades:
            j = node_to_idx[child]

            A[i, j] = 1.0
            A[j, i] = 1.0

    eigvals, eigvecs = np.linalg.eigh(A)

    # largest eigenvalue
    idx = np.argmax(eigvals)

    eigenvector = np.abs(eigvecs[:, idx])

    return float(np.max(eigenvector))

def stairs_index(tree):
    """
    Stairs index.

    Fraction of internal nodes whose two daughter
    subtrees contain a different number of tips.
    """

    imbalanced = 0
    internal = 0

    def tip_count(clade):
        if clade.is_terminal():
            return 1

        counts = [tip_count(c) for c in clade.clades]

        nonlocal imbalanced, internal

        if len(counts) == 2:
            internal += 1

            if counts[0] != counts[1]:
                imbalanced += 1

        return sum(counts)

    tip_count(tree.root)

    if internal == 0:
        return 0.0

    return imbalanced / internal

def symmetry_nodes(tree):
    """
    Count internal nodes that are NOT symmetric.
    """

    nonsymmetric = 0

    def depth_sum(clade, depth=0):
        total = depth

        for child in clade.clades:
            total += depth_sum(child, depth + 1)

        return total

    def traverse(clade):
        nonlocal nonsymmetric

        if clade.is_terminal():
            return

        if len(clade.clades) == 2:
            left, right = clade.clades

            s_left = depth_sum(left, 0)
            s_right = depth_sum(right, 0)

            if s_left != s_right:
                nonsymmetric += 1

        for child in clade.clades:
            traverse(child)

    traverse(tree.root)

    return nonsymmetric

def max_betweenness(tree):
    """
    treestats::max_betweenness equivalent
    """

    subtree_nodes = {}

    def count_nodes(clade):
        n = 1

        for child in clade.clades:
            n += count_nodes(child)

        subtree_nodes[clade] = n
        return n

    N = count_nodes(tree.root)

    best = -1.0

    for node in tree.find_clades():

        child_sizes = [
            subtree_nodes[child]
            for child in node.clades
        ]

        q = sum(child_sizes)

        sum_sq = sum(s * s for s in child_sizes)

        child_pairs = (q * q - sum_sq) / 2.0

        local_b = child_pairs + q * (N - 1 - q)

        best = max(best, local_b)

    return float(best)





def min_adj(tree):
    """
    Smallest positive eigenvalue of weighted adjacency matrix.
    Equivalent to treestats::minmax_adj(... )$min
    """

    nodes = list(tree.find_clades(order="level"))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    A = np.zeros((n, n), dtype=float)

    for parent in tree.find_clades(order="level"):
        i = node_to_idx[parent]

        for child in parent.clades:
            j = node_to_idx[child]

            w = child.branch_length or 0.0

            A[i, j] = w
            A[j, i] = w

    eigvals = np.linalg.eigvalsh(A)

    eigvals = np.round(eigvals, 10)

    positive = eigvals[eigvals > 0]

    if len(positive) == 0:
        return 0.0

    return float(np.min(positive))

def main():
    parser = argparse.ArgumentParser(
        description="Validate phylogenetic diversity and Blum index."
    )
    parser.add_argument(
        "--tree_file",
        type=Path,
        required=True,
        help="CSV containing computed tree features.",
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
        help="Output report."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for floating-point comparisons.",
    )

    args = parser.parse_args()

    features = pd.read_csv(args.tree_file)
    trees = load_trees(args.clean_file)

    if len(features) != len(trees):
        raise ValueError(
            f"Number of feature rows ({len(features)}) "
            f"does not match number of trees ({len(trees)})."
        )

    output_created = False
    out = None
    mismatches = 0

    for i, tree in enumerate(trees):
        expected_pd = features.loc[i, "phylogenetic_div"]
        expected_blum = features.loc[i, "blum"]
        expected_eigen = features.loc[i, "eigen_centrality"]
        expected_stairs = features.loc[i, "stairs"]
        expected_sym = features.loc[i, "symmetry_nodes"]
        expected_bet = features.loc[i, "max_betweenness"]
        expected_min_adj = features.loc[i, "min_adj"]

        calculated_stairs = stairs_index(tree)
        # calculated_sym = symmetry_nodes(tree)
        calculated_bet = max_betweenness(tree)
        calculated_min_adj = min_adj(tree)
        calculated_pd = phylogenetic_diversity(tree)
        calculated_blum = blum_index(tree)
        calculated_eigen = eigen_centrality(tree)

        pd_ok = np.isclose(calculated_pd, expected_pd, atol=args.atol, rtol=0)
        blum_ok = np.isclose(calculated_blum, expected_blum, atol=args.atol, rtol=0)
        eigen_ok = np.isclose(calculated_eigen, expected_eigen, atol=args.atol, rtol=0)
        # sym_ok = np.isclose(calculated_sym, expected_sym, atol=args.atol, rtol=0)
        bet_ok = np.isclose(calculated_bet, expected_bet, atol=args.atol, rtol=0)
        stairs_ok = np.isclose(calculated_stairs, expected_stairs, atol=args.atol, rtol=0)
        min_adj_ok = np.isclose(calculated_min_adj, expected_min_adj, atol=args.atol, rtol=0)

        if pd_ok and blum_ok and eigen_ok and bet_ok and stairs_ok and min_adj_ok:
            continue

        mismatches += 1

        if not output_created:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            out = open(args.output_file, "w")
            out.write(
                "Validation report for tree features\n"
                "===================================\n\n"
            )
            output_created = True

        out.write(f"Tree {i + 1}\n")

        if not pd_ok:
            out.write(
                "  Phylogenetic diversity mismatch\n"
                f"    expected   : {expected_pd:.12f}\n"
                f"    calculated : {calculated_pd:.12f}\n"
            )

        if not blum_ok:
            out.write(
                "  Blum index mismatch\n"
                f"    expected   : {expected_blum:.12f}\n"
                f"    calculated : {calculated_blum:.12f}\n"
            )

        '''
        if not sym_ok:
            out.write(
                "  Symmetry nodes mismatch\n"
                f"    expected   : {expected_sym:.12f}\n"
                f"    calculated : {calculated_sym:.12f}\n"
            )
        '''

        if not bet_ok:
            out.write(
                "  Maximum betweenness mismatch\n"
                f"    expected   : {expected_bet:.12f}\n"
                f"    calculated : {calculated_bet:.12f}\n"
            )

        if not stairs_ok:
            out.write(
                "  Stairs index mismatch\n"
                f"    expected   : {expected_stairs:.12f}\n"
                f"    calculated : {calculated_stairs:.12f}\n"
            )

        if not min_adj_ok:
            out.write(
                "  Minimum adjacency eigenvalue mismatch\n"
                f"    expected   : {expected_min_adj:.12f}\n"
                f"    calculated : {calculated_min_adj:.12f}\n"
            )

        out.write("\n")

    if output_created:
        out.write(
            "===================================\n"
            f"Summary: {mismatches} of {len(trees)} trees contained mismatching values.\n"
        )
        out.close()
    else:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.touch()


if __name__ == "__main__":
    main()