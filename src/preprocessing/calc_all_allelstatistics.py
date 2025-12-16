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
import shutil
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from snakemake_vars import FILE_FORMATS

# ============================================================
# --- Parsing ms-like output ---
# ============================================================
def parse_ms_like_output(output):
    lines = output.strip().splitlines()
    command_line = lines[0]

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

    return all_positions, all_genotypes

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
    if positions.size > 1:
        dup = positions[np.where(np.diff(positions) == 0)]
        if len(dup) > 0:
            print("⚠️ WARNING: Duplicate positions found:", dup)
            print("  All positions:", positions)
        
        if not np.all(np.diff(positions) > 0):
            print("⚠️ WARNING: positions are NOT strictly increasing!")
            print("  All positions:", positions)

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


# ============================================================
# --- Main function ---
# ============================================================
def main(input_dir, output_dir):
    # Ensure the main output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Walk recursively through all subdirectories in the input folder
    for root, _, files in os.walk(input_dir):
        root_path = Path(root)

        # --- metadata.yaml laden ---
        metadata_files = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
        if not metadata_files:
            continue
        
        metadata_path = root_path / metadata_files[0]
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        if "sequence_length" not in metadata:
            raise ValueError(f"sequence_length missing in metadata file {metadata_path}")
        if "overall_samples" not in metadata:
            raise ValueError(f"sequence_length missing in metadata file {metadata_path}")
        if "sample_structure" not in metadata:
            raise ValueError(f"sample_structure missing in metadata file {metadata_path}")
        if "cleaned_outputs" not in metadata and "simulator_outputs" not in metadata:
            raise ValueError(f"simulator_outputs and cleaned_outputs missing in metadata file {metadata_path}")
        
        if "cleaned_outputs" in metadata:
            cleaned_outputs = metadata.get("cleaned_outputs")
        else:
            cleaned_outputs = metadata.get("simulator_outputs")

        seq_length = metadata.get("sequence_length")
        overall_samples = metadata.get("overall_samples")
        sample_structure = metadata.get("sample_structure")
        cleaned_outputs = metadata.get("cleaned_outputs")
        # --- Unterordner (z.B. Replikate) durchgehen ---
        for rep_root, rep_dirs, rep_files in os.walk(root_path):
            rep_root_path = Path(rep_root)
            if not rep_files:
                continue

            for sim, filename in cleaned_outputs.items():
                file_path = Path(filename)
                input_file = file_path if file_path.is_absolute() else rep_root_path / file_path

                if not input_file.exists():
                    continue

                # --- Ausgabe-Datei ---
                target_dir = output_dir / rep_root_path.relative_to(input_dir)
                os.makedirs(target_dir, exist_ok=True)

                # --- Dateiinhalt lesen ---
                with open(input_file) as f:
                    content = f.read()

                # Parse ms/msms/msprime-like simulator output
                all_positions, all_genotypes = parse_ms_like_output(content)
                rows = []
                    
                # Compute statistics for each block in the file
                for block_idx, (positions, genotypes) in enumerate(zip(all_positions, all_genotypes)):
                    allel_features = calculate_block_stats(positions, genotypes, sample_structure)
                    rows.append(allel_features)

                    # === NEW: Direct MSA from binary genotypes ===

                    # We assume genotypes is (n_haplotypes × n_segsites)
                    # If segsites = 0, skip difficulty
                    if genotypes.size == 0:
                        allel_features["difficulty"] = -1
                        allel_features["bollback_multinomial"] = 0.0
                        allel_features["pattern_entropy"] = 0.0
                        allel_features["entropy"] = 0.0
                        rows.append(allel_features)
                        continue

                    # Convert genotype matrix (0/1 ints) into sequences of "0"/"1"
                    seqs = ["".join(str(x) for x in row) for row in genotypes]

                    # Create taxa names
                    taxa = np.array([f"sample_{i+1}" for i in range(genotypes.shape[0])])

                    # Convert list → matrix of bytes
                    seq_matrix = np.array([list(s) for s in seqs], dtype="S1")

                    # Initialize MSA with MORPH type
                    msa = MSA(
                        taxa=taxa,
                        sequences=seq_matrix,
                        data_type=DataType.MORPH,
                        name=f"{sim}_block{block_idx}"
                    )

                    # Write temporary PHYLIP file
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".phy", delete=False) as tmp_msa_file:
                        msa.write(tmp_msa_file.name)
                        tmp_msa_path = Path(tmp_msa_file.name)

                    raxml_path = Path(shutil.which("raxml-ng"))
                    if raxml_path is None:
                        raise FileNotFoundError("RAxML-NG not found in PATH")

                    try:
                        unique_seqs = np.unique(seqs, axis=0)
                        if len(unique_seqs) < 4:
                            print(f"⚠️ Skipping difficulty: only {len(unique_seqs)} sequences (need ≥4).")
                            allel_features["difficulty"] = -1
                        else:
                            try:
                                difficulty = predict_difficulty(
                                    msa_file=tmp_msa_path,
                                    raxmlng=raxml_path,
                                    file_format=FileFormat.PHYLIP,
                                    data_type=DataType.MORPH,
                                    deduplicate=False,
                                    store_results=False
                                )
                                allel_features["difficulty"] = difficulty
                            except Exception as e:
                                print(f"⚠️ PyPythia failed for block {block_idx}: {e}")
                                allel_features["difficulty"] = -1

                    finally:
                        os.remove(tmp_msa_path)

                    # MSA internal statistics
                    allel_features["bollback_multinomial"] = msa.bollback_multinomial()
                    allel_features["pattern_entropy"] = msa.pattern_entropy()
                    allel_features["entropy"] = msa.entropy()


                # Save per-block statistics as CSV if any data was generated
                if rows:
                    df = pd.DataFrame(rows)
                    out_name = FILE_FORMATS["allel_features"].format(SIM=sim)
                    output_path = os.path.join(target_dir, out_name)

                    # Save with fixed float precision for reproducibility
                    df.to_csv(output_path, index=False, float_format='%.10f')
                    print(f"✅ Saved {len(rows)} blocks to {output_path}")

                    # --- Pfad in metadata ergänzen ---
                    metadata.setdefault("allel_features", {})[sim] = str(out_name)
                    
        out_metadata_path = output_dir / root_path.relative_to(input_dir) / metadata_files[0]
        out_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_metadata_path, "w") as f_out:
            yaml.dump(metadata, f_out, sort_keys=False)

# ============================================================
# --- CLI entry point ---
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute population genetic statistics from ms-like output and save as CSV.'
    )
    parser.add_argument('input_dir', help='Directory containing ms/msms/msprime output files')
    parser.add_argument('output_dir', help='Directory to write CSV files')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
