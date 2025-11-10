#!/usr/bin/env python3
import os
import re
import argparse
import json
import numpy as np
import allel
import random
import pandas as pd
from scipy.stats import skew, kurtosis
from itertools import combinations
import sys
import yaml
import tempfile
from pypythia.prediction import predict_difficulty
from pypythia.msa import *
from pathlib import Path

UTILS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(UTILS_ROOT)

from utils.io_helpers import clean_segsites
    
def generate_msa(positions, genotypes, seq_length, sample_size):
    """Generate an MSA from ms-like genotypes and random ancestral sequence."""
    bases = ['A', 'C', 'G', 'T']
    n_haps = genotypes.shape[0] if genotypes.size > 0 else 0

    assert((sample_size == n_haps) or n_haps == 0)

    # 1. Generate random ancestral sequence
    ancestral_seq = [random.choice(bases) for _ in range(seq_length)]

    # 2. Copy ancestral sequence for all haplotypes
    haplotypes = [ancestral_seq.copy() for _ in range(sample_size)]

    # 3. For each segregating site: introduce derived alleles
    if n_haps != 0:
        for j, idx in enumerate(positions):
            for i in range(n_haps):
                if genotypes[i, j] == 1:
                    # Derived allele: different base from ancestral
                    try:
                        anc = haplotypes[i][idx]
                    except IndexError:
                        print("⚠️ IndexError encountered while accessing haplotypes.")
                        print(f"  Sequence length: {seq_length}")
                        print(f"  Index attempted: {idx}")
                        print(f"  Positions: {positions}")
                        print(f"  Haplotype index: {i}")
                        # Optionally, re-raise the error after printing debug info
                        raise

                    possible = [b for b in bases if b != anc]
                    haplotypes[i][idx] = random.choice(possible)

    return ["".join(seq) for seq in haplotypes]

# ============================================================
# --- Parsing ms-like output ---
# ============================================================
def parse_ms_like_output(output):
    lines = output.strip().splitlines()
    command_line = lines[0]
    sample_sizes = []

    match = re.search(r'-I\s+\d+((?:\s+\d+)+)', command_line)
    if match:
        sample_sizes = [int(x) for x in match.group(1).split()]
    else:
        parts = command_line.strip().split()
        if len(parts) >= 2 and re.match(r'\d+', parts[1]):
            # erste Zahl nach dem Simulator
            sample_sizes = [int(parts[1])]
        else:
            sample_sizes = []

    blocks = output.strip().split('//')
    all_positions, all_genotypes = [], []

    for block in blocks:
        # Skip empty or invalid blocks but preserve alignment in output lists
        if not block:
            continue

        # Extract number of segregating sites (segsites)
        segsites_match = re.search(r'segsites:\s*(\d+)', block)
        segsites = int(segsites_match.group(1)) if segsites_match else -1

        if segsites == -1:
            continue

        # If there are no segregating sites, append empty placeholders and skip
        if segsites == 0:
            all_positions.append(np.empty(0, dtype=np.int8))
            all_genotypes.append(np.empty((0, 0), dtype=np.int8))
            continue

        allele_lines = block.splitlines()
        pos_line_idx = next((i for i, l in enumerate(allele_lines) if l.startswith('positions:')), None)
        if pos_line_idx is None:
            continue

        positions = np.array([int(x) for x in allele_lines[pos_line_idx].split()[1:]])
        all_positions.append(positions)

        genotypes = []
        for line in allele_lines[pos_line_idx + 1:]:
            line = line.strip()
            if re.fullmatch(r'[01]+', line):
                genotypes.append([int(c) for c in line])
            else:
                break

        if len(genotypes) > 0:
            all_genotypes.append(np.array(genotypes, dtype=np.int64))

    return sample_sizes, all_positions, all_genotypes

# ============================================================
# --- Compute stats ---
# ============================================================
from itertools import combinations

def compute_stats(haps: allel.HaplotypeArray, positions, pops={}):
    def choose_block_length(n_variants, target_blocks=10, min_blen=5):
        """
        Choose block length for block-jackknife.

        Parameters
        ----------
        n_variants : int
            Number of variants in the current region.
        target_blocks : int, optional
            Desired number of blocks for jackknife. Default is 10.
        min_blen : int, optional
            Minimum block length in number of variants. Default is 5.

        Returns
        -------
        blen : int
            Block length in number of variants.
        """
        if n_variants == 0:
            return 1  # avoid division by zero
        blen = max(min_blen, n_variants // target_blocks)
        return blen

    def ac_for(indices):
        indices = [i for i in indices if i < haps.shape[1]]  # limit to valid range
        hsub = haps.take(indices, axis=1)

        # Handle case: no segregating sites (empty haplotype matrix)
        if hsub.shape[0] == 0 or hsub.shape[1] == 0:
            ac = np.zeros((0, 2), dtype=np.int64)
            return ac, hsub

        return hsub.count_alleles(), hsub

    def stats_from(ac, hsub):
        if hsub.size == 0:
            return {
                'pi': 0.0,
                'theta_w': 0.0,
                'tajimas_d': 0.0,
                'haplotype_homozygosity': 1.0,
                'n_unique_haplotypes': 1,
                'sfs_skewness': 0.0,
                'sfs_kurtosis': 0.0,
                'mean_ld': 0.0,
                'var_ld': 0.0,
                'mean_pairwise_distance': 0.0
            }

        n_hap = hsub.shape[1]

        # Diversity
        pi = allel.sequence_diversity(positions, ac)
        theta_w = allel.watterson_theta(positions, ac)
        try:
            tajd = allel.tajima_d(ac)
            if np.isnan(tajd):
                tajd = 0.0
        except Exception:
            tajd = 0.0

        # Haplotype matrix
        hmat = hsub
        if hmat.size == 0:
            hap_homo = 1.0
            n_unique = 1
        else:
            cols = np.ascontiguousarray(hmat.T)
            uniq, counts = np.unique(cols, axis=0, return_counts=True)
            freqs = counts / counts.sum()
            hap_homo = np.sum(freqs ** 2)
            n_unique = len(counts)

        # Derived counts
        if ac.shape[1] > 1:
            derived_counts = ac[:, 1]
        else:
            derived_counts = np.zeros(ac.shape[0], dtype=int)

        max_k = n_hap
        if derived_counts.size == 0:
            sfs = np.zeros(max_k + 1, dtype=int)
        else:
            minor_counts = np.minimum(derived_counts, max_k - derived_counts)
            sfs = np.bincount(minor_counts.astype(int), minlength=max_k + 1)

        # SFS moments
        try:
            sfs_skew = skew(sfs)
            if np.isnan(sfs_skew):
                sfs_skew = 0.0
        except Exception:
            sfs_skew = 0.0

        try:
            sfs_kurt = kurtosis(sfs)
            if np.isnan(sfs_kurt):
                sfs_kurt = 0.0
        except Exception:
            sfs_kurt = 0.0

        # LD
        try:
            r = allel.rogers_huff_r(hsub)
            r2 = r ** 2
            if r2.size == 0 or np.all(np.isnan(r2)):
                mean_ld = 0.0
                var_ld = 0.0
            else:
                mean_ld = float(np.nanmean(r2))
                var_ld = float(np.nanvar(r2))
        except Exception:
            mean_ld = 0.0
            var_ld = 0.0

        # Mean pairwise distance (Hamming)
        try:
            d = allel.pairwise_distance(hsub, metric='hamming')
            mean_dist = float(np.mean(d)) if d.size > 0 else 0.0
        except Exception:
            mean_dist = 0.0

        # Garud’s H statistics
        try:
            h1, h12, h123, h2_h1 = allel.garud_h(hsub)
        except Exception:
            h1, h12, h123, h2_h1 = 0, 0, 0, 0 

        # Haplotype diversity
        try:
            hap_div = allel.haplotype_diversity(hsub)
        except Exception:
            hap_div = 0.0

        return {
            'pi': float(pi),
            'theta_w': float(theta_w),
            'tajimas_d': float(tajd),
            'haplotype_homozygosity': float(hap_homo),
            'n_unique_haplotypes': int(n_unique),
            'sfs_skewness': float(sfs_skew),
            'sfs_kurtosis': float(sfs_kurt),
            'mean_ld': float(mean_ld),
            'var_ld': float(var_ld),
            'mean_pairwise_distance': float(mean_dist),
            'h1': float(h1),
            'h12': float(h12),
            'h123': float(h123),
            'h2_h1': float(h2_h1),
            'hap_div': float(hap_div)
        }

    results = {}
    all_indices = [i for i in range(0, haps.shape[1])]
    
    ac_all, hsub_all = ac_for(all_indices)

    results['pooled'] = stats_from(ac_all, hsub_all)
    results['num_segsites'] = len(positions)

    if len(pops) > 1:
        # Compute allele counts for each population for Patterson stats
        ac_dict = {}
        for pname, indices in pops.items():
            ac, _ = ac_for(indices)
            ac_dict[pname] = ac

        # --- Compute per-population stats ---
        results['per_population'] = {}
        for pname, indices in pops.items():
            ac, hsub = ac_for(indices)
            results['per_population'][pname] = stats_from(ac, hsub)

        # --- Compute Patterson statistics ---
        pop_names = list(pops.keys())

        # F2 for all pairs
        for a, b in combinations(pop_names, 2):
            ac_a, ac_b = ac_dict[a], ac_dict[b]
            nvar = min(ac_a.shape[0], ac_b.shape[0])
            blen = min(choose_block_length(nvar), nvar // 2)

            if all(ac.shape[0] > 0 and ac.shape[1] == 2 for ac in [ac_a, ac_b]):
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fst, se, vb, vj = allel.average_hudson_fst(ac_a, ac_b, blen=blen)
                    fst = np.nan_to_num(fst, nan=np.nan)
                except Exception:
                    fst = 0.0

                f2 = allel.patterson_f2(ac_a, ac_b)
                results[f"f2_{a}_{b}"] = float(np.mean(f2))
                results[f"fst_{a}_{b}"] = float(fst)
            else:
                results[f"f2_{a}_{b}"] = 0.0
                results[f"fst_{a}_{b}"] = 0.0

        # F3* for all triplets
        for c, a, b in combinations(pop_names, 3):
            ac_c, ac_a, ac_b = ac_dict[c], ac_dict[a], ac_dict[b]
            if all(ac.shape[0] > 0 and ac.shape[1] == 2 for ac in [ac_a, ac_b, ac_c]):
                T, B = allel.patterson_f3(ac_c, ac_a, ac_b)
                results[f"f3star_{c}_{a}_{b}"] = float(np.sum(T) / np.sum(B)) if np.sum(B) > 0 else 0.0
            else:
                results[f"f3star_{c}_{a}_{b}"] = 0.0

    return results


# ============================================================
# --- Helper to compute per-block stats and flatten ---
# ============================================================
def calculate_block_stats(positions, genotypes, sample_sizes):
    n_haplotypes = sum(sample_sizes)
    haps = allel.HaplotypeArray(genotypes.T)
    n_haps = haps.shape[1]
    no_segsites = False

    if n_haps == 0:
        assert genotypes.size == 0, "Genotypes muss leer sein"
        assert positions.size == 0, "Positions muss leer sein, wenn es keine mutation gab"

    # Flatten dictionary for CSV
    flat = {}
    if len(sample_sizes) > 1:
        pops = {}
        start = 0

        for i, s in enumerate(sample_sizes):
            pops[f"pop{i+1}"] = list(range(start, start + s))
            start += s
        
        for pname in list(pops.keys()):
            pops[pname] = [i for i in pops[pname] if i < n_haps]

        stats = compute_stats(haps, positions, pops)

        for pname, vals in stats['per_population'].items():
            for k, v in vals.items():
                if isinstance(v, (list, dict)):
                    v = json.dumps(v)
                flat[f"{pname}_{k}"] = v
    else:
        stats = compute_stats(haps, positions)

    for k, v in stats.items():
            if k not in ['per_population', 'pooled']:
                flat[k] = v

    for k, v in stats['pooled'].items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v)
        flat[f"pooled_{k}"] = v

    return flat


def parse_sample_size_and_seq_length_from_dir(dirname):
    #print(f"Das ist der dirname: {dirname}")
    """
    Extract sample_size and sequence_length from directory name, assuming 
    naming scheme like 'popsize-50_seq-1000_...' 
    """
    sample_size = None
    seq_length = None

    m = re.search(r'sample_sizes-(\d+)', dirname)
    if m:
        sample_size = int(m.group(1))

    m = re.search(r'sequence_lengths-(\d+)', dirname)
    if m:
        seq_length = int(m.group(1))

    if sample_size is None or seq_length is None:
        raise ValueError(f"Could not parse sample_size or seq_length from {dirname}")

    return sample_size, seq_length


# ============================================================
# --- Main function ---
# ============================================================
def main(input_dir, output_dir, input_config):
    # Ensure the main output directory exists
    os.makedirs(output_dir, exist_ok=True)

    sample_size = 0
    seq_length = 0

    if input_config:
        with open(input_config, "r") as f:
            cfg = yaml.safe_load(f)

        sample_size = sum(p.get("sample_size", 0) for p in cfg.get("populations", [])) \
                            if "populations" in cfg else int(cfg.get("sample_size"))

        seq_length = int(cfg.get("sequence_length"))

    # Walk recursively through all subdirectories in the input folder
    for root, _, files in os.walk(input_dir):
        # Determine relative path from input_dir to current folder (e.g. "rep1", "rep2")
        rel_path = os.path.relpath(root, input_dir)

        # Create corresponding subfolder structure inside output_dir
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        # Process all ms-like files in the current directory
        for fname in files:
            if not fname.endswith('.txt') and not fname.endswith('.ms'):
                continue

            input_path = Path(root)/ fname
            with open(input_path) as f:
                content = f.read()

            if not input_config:
                parent_folder = input_path.parent.name
                sample_size, seq_length = parse_sample_size_and_seq_length_from_dir(parent_folder)

                if not sample_size or not seq_length:
                    raise ValueError(
                        f"Could not extract sample_size or sequence_length from folder '{parent_folder}'. "
                        "Please provide a valid input_config or ensure the folder name encodes these values."
                    )
            
            # TODO: clean content
            content = clean_segsites(
                sim_output = content, 
                sequence_length = seq_length
                )

            # Parse ms/msms/msprime-like simulator output
            sample_sizes, all_positions, all_genotypes = parse_ms_like_output(content)
            rows = []
                
            # Compute statistics for each block in the file
            for block_idx, (positions, genotypes) in enumerate(zip(all_positions, all_genotypes)):
                allel_stats = calculate_block_stats(positions, genotypes, sample_sizes)
                rows.append(allel_stats)

                # --- Create an MSA object ---
                seqs = generate_msa(positions, genotypes, seq_length, sample_size)

                # Create names for the samples (taxa)
                taxa = np.array([f"sample_{i+1}" for i in range(sample_size)])

                # Convert list of strings into a 2D NumPy array with dtype='S1'
                seq_matrix = np.array([list(s) for s in seqs], dtype='S1')

                # Initialize the MSA object
                msa = MSA(
                    taxa=taxa,
                    sequences=seq_matrix,
                    data_type=DataType.DNA,   # or DataType("DNA"), depending on the implementation
                    name=f"{fname}_block{block_idx}"
                )

                # Create a temporary file to store the MSA in PHYLIP format
                with tempfile.NamedTemporaryFile(mode="w", suffix=".phy", delete=False) as tmp_msa_file:
                    # Write the MSA object to a file (make sure your MSA class supports this)
                    msa.write(tmp_msa_file.name)  # or msa.write_phylip(tmp_msa_file.name)
                    tmp_msa_path = pathlib.Path(tmp_msa_file.name)

                try:
                    # Call predict_difficulty using the file path instead of the MSA object
                    difficulty = predict_difficulty(
                        msa_file=tmp_msa_path,
                        raxmlng=pathlib.Path("/home/kuchvict/miniforge3/envs/allelstats/bin/raxml-ng"),       # set this if RAxML-NG is not in your PATH
                        deduplicate=False,
                        store_results=False # avoid saving extra result files
                    )
                    allel_stats["difficulty"] = difficulty
                finally:
                    # Clean up the temporary file after prediction
                    os.remove(tmp_msa_path)

                allel_stats["bollback_multinomial"] = msa.bollback_multinomial()
                allel_stats["pattern_entropy"] = msa.pattern_entropy()
                allel_stats["entropy"] = msa.entropy()

            # Save per-block statistics as CSV if any data was generated
            if rows:
                df = pd.DataFrame(rows)
                out_name = os.path.splitext(fname)[0] + '_stats.csv'
                output_path = os.path.join(target_dir, out_name)

                # Save with fixed float precision for reproducibility
                df.to_csv(output_path, index=False, float_format='%.10f')
                print(f"✅ Saved {len(rows)} blocks to {output_path}")


# ============================================================
# --- CLI entry point ---
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute population genetic statistics from ms-like output and save as CSV.'
    )
    parser.add_argument('input_dir', help='Directory containing ms/msms/msprime output files')
    parser.add_argument('output_dir', help='Directory to write CSV files')
    parser.add_argument('--input_config', help='Optional config file for simulator parameters', default=None)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.input_config)
