import os
from Bio import Phylo
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from scipy.linalg import eigh

def read_first_tree(file_path):
    """Liest die erste Baumdefinition aus einer ms/msms/msprime txt-Datei."""
    with open(file_path) as f:
        # parse() liefert einen Iterator über alle Bäume
        trees = list(Phylo.parse(f, "newick"))
    if not trees:
        raise ValueError(f"No tree found in {file_path}")
    return trees[0]  # erster Baum

def compute_unweighted_height(tree):
    """Ersetzt alle Branch-Lengths durch 1, um unweighted height zu berechnen."""
    for clade in tree.find_clades():
        clade.branch_length = 1.0 if clade.branch_length is not None else 1.0
    return tree

def rescale_tree(tree, target_height):
    """Skaliert den Baum so, dass die maximale Tiefe target_height entspricht."""
    def max_depth(clade):
        if clade.is_terminal():
            return 0
        return max(max_depth(c) + (c.branch_length or 1) for c in clade.clades)
    
    current_height = max_depth(tree.root)
    scale = target_height / current_height if current_height > 0 else 1.0
    
    for clade in tree.find_clades():
        if clade.branch_length is not None:
            clade.branch_length *= scale
    return tree

def characteristic_vertices(tree):
    """Berechnet die characteristic vertex/vertices des Baums nach Laplacian."""
    G = nx.Graph()
    for clade in tree.find_clades(order="level"):
        for child in clade.clades:
            # Knoten hinzufügen, auch wenn kein Name vorhanden (z.B. interne Knoten)
            if clade.name is None:
                clade_name = id(clade)  # interne Knoten bekommen eindeutige ID
            else:
                clade_name = clade.name

            if child.name is None:
                child_name = id(child)
            else:
                child_name = child.name

            G.add_node(clade_name)
            G.add_node(child_name)

            # Gewicht = branch_length, falls vorhanden, sonst 1
            weight = child.branch_length if child.branch_length is not None else 1.0
            G.add_edge(clade_name, child_name, weight=weight)

    # Laplacian
    L = nx.laplacian_matrix(G, weight='weight').toarray()
    eigenvals_nx = nx.laplacian_spectrum(G, weight='weight')
    # Eigenwerte/Eigenvectors
    eigvals, eigvecs = eigh(L)

    # Auf 12 Nachkommastellen runden
    eigvals = np.round(eigvals, 10)
    eigvecs = np.round(eigvecs, 10)
    eigenvals_nx = np.round(eigenvals_nx, 10)
    eigenvals_nx = eigenvals_nx[eigenvals_nx > 0]

    # Alle Eigenwerte, die quasi 0 sind, rausnehmen
    nonzero_indices = np.where(eigvals > 0)[0]

    # Minimaler Eigenwert > 0 → der "zweite" Eigenwert
    fiedler_index = nonzero_indices[0]

    print(f"Algeb. conn: {min(eigvals[eigvals > 0])}")
    print(f"Algeb. conn nx: {min(eigenvals_nx)}")

    # Entsprechender Fiedler-Vektor
    fiedler_vector = eigvecs[:, fiedler_index]
    # Characteristic vertices:
    zero_indices = np.where(np.isclose(fiedler_vector, 0, atol=1e-10))[0]
    result = []

    if len(zero_indices) == 1:
        result = [list(G.nodes())[zero_indices[0]]]
    elif len(zero_indices) == 0:
        # Find edge with sign change
        signs = np.sign(fiedler_vector)
        for i, u in enumerate(G.nodes()):
            for j, v in enumerate(G.nodes()):
                if G.has_edge(u, v) and signs[i]*signs[j] < 0:
                    result.append(u)
                    result.append(v)
    else:
        for i, u in enumerate(G.nodes()):
            if i not in zero_indices:
                continue
            for j, v in enumerate(G.nodes()):
                if j in zero_indices:
                    continue
                if G.has_edge(u, v):
                    result.append(u)
    if not result:
        print("⚠️ No characteristic vertices found")
        # print(f"Fiedler vector: {fiedler_vector}")
    elif len(result) > 2:
        print("⚠️ Too many characteristic vertices found")
    else:
        print("✅ Computing char vertices worked")

    return result

def plot_trees(trees, char_vertices_list, labels, output_file):
    fig, axes = plt.subplots(1, len(trees), figsize=(6*len(trees),6))
    if len(trees) == 1:
        axes = [axes]
    for ax, tree, char_vertices, label in zip(axes, trees, char_vertices_list, labels):
        Phylo.draw(tree, do_show=False, axes=ax, show_confidence=False)
        ax.set_title(label)
        # Characteristic vertices markieren
        for clade in tree.find_clades():
            if clade.name in char_vertices:
                ax.scatter(0,0, color="red", s=50)  # Platzhalter für Punktmarker
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = ["ms.txt", "msms.txt", "msprime.txt"]
    trees = []
    char_vertices_list = []
    max_unweighted_height = 0
    
    # 1. Bäume einlesen
    for file in files:
        print(f"File: {file}")
        tree = read_first_tree(os.path.join(input_dir, file))
        unweighted_tree = compute_unweighted_height(tree)
        height = max([clade.branch_length for clade in unweighted_tree.find_clades()])
        max_unweighted_height = max(max_unweighted_height, height)
        trees.append(tree)
        print("-----------------------------------------")
    
    # 2. Rescale auf max_unweighted_height
    trees = [rescale_tree(tree, max_unweighted_height) for tree in trees]
    
    # 3. Characteristic vertices berechnen
    char_vertices_list = [characteristic_vertices(tree) for tree in trees]
    
    # 4. Plotten
    plot_trees(trees, char_vertices_list, labels=["ms", "msms", "msprime"],
               output_file=os.path.join(output_dir, "trees.png"))

'''
def main(input_dir, output_dir):
    files = ["ms.txt", "msms.txt", "msprime.txt"]
    trees = []
    char_vertices_list = []

    for file in files:
        tree = read_first_tree(os.path.join(input_dir, file))
        trees.append(tree)

    char_vertices_list = [characteristic_vertices(tree) for tree in trees]
'''

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)
