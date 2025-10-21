#!/usr/bin/env Rscript
source("src/preprocessing/calc_all_treestatistics.R")

# --- Argument parsing ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_treestats_clean.R <input_txt|input_dir> <output_dir>")

input_path <- args[1]
output_dir <- args[2]

if (!file.exists(input_path)) stop("Input file/directory does not exist: ", input_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

library(ape)
library(treestats)
library(igraph)

# --- Collect input files recursively ---
if (dir.exists(input_path)) {
  txt_files <- list.files(input_path, pattern = "\\.txt$", full.names = TRUE, recursive = TRUE)
  if (length(txt_files) == 0) stop("No .txt files found in directory: ", input_path)
} else {
  txt_files <- input_path
}

# --- Main file processor ---
process_file <- function(infile, input_base, output_base) {
  rel_path <- sub(paste0("^", normalizePath(input_base, winslash = "/"), "/?"), "", normalizePath(infile, winslash = "/"))
  outfile <- file.path(output_base, paste0(tools::file_path_sans_ext(rel_path), "_stats.csv"))
  dir.create(dirname(outfile), recursive = TRUE, showWarnings = FALSE)

  lines <- readLines(infile, warn = FALSE)

  # --- filter only valid Newick lines ---
  is_tree_line <- grepl("\\(", lines) & grepl(";+", lines) &
    !grepl("^\\s*//", lines) &
    !grepl("^\\s*segsites", lines, ignore.case = TRUE) &
    !grepl("^\\s*positions", lines, ignore.case = TRUE)

  tree_lines <- trimws(sub(";+$", ";", lines[is_tree_line]))
  results_list <- list()
  skipped <- 0L

  for (i in seq_along(tree_lines)) {
    txt <- tree_lines[i]
    tr <- tryCatch(ape::read.tree(text = txt), error = function(e) e)
    if (inherits(tr, "error")) { 
      skipped <- skipped + 1L
      next 
    }

    # --- Create unweighted tree ---
    tr_unweighted <- tr
    tr_unweighted$edge.length <- rep(1, nrow(tr_unweighted$edge))

    # --- Weighted-Stats ---
    tree_stats <- tryCatch(
      calculate_all_treestats(tr, normalize = FALSE, weighted = TRUE),
      error = function(e) NULL
    )

    # --- Unweighted-Stats ---
    tree_stats_unweighted <- tryCatch(
      calculate_all_treestats(tr_unweighted, normalize = FALSE, weighted = FALSE),
      error = function(e) NULL
    )

    # Skip this tree if computation failed
    if (is.null(tree_stats) || is.null(tree_stats_unweighted)) {
      skipped <- skipped + 1L
      next
    }

    # --- ensure both results are data frames ---
    if (!is.data.frame(tree_stats)) tree_stats <- as.data.frame(as.list(tree_stats))
    if (!is.data.frame(tree_stats_unweighted)) tree_stats_unweighted <- as.data.frame(as.list(tree_stats_unweighted))

    # --- combine both results into one row ---
    combined <- cbind(tree_stats, tree_stats_unweighted)

    # --- save ---
    results_list[[length(results_list) + 1L]] <- combined
  }

  if (length(results_list) == 0L) {
    warning("⚠️ No trees processed successfully in ", infile)
  } else {
    stats_df <- dplyr::bind_rows(results_list)  # rbind.fill aus plyr, falls Spalten mal fehlen
    write.csv(stats_df, file = outfile, row.names = FALSE)
    cat(sprintf("✅ Done: %s -> %s (processed %d trees, skipped %d)\n", infile, outfile, length(results_list), skipped))
  }
}


# --- Process all files ---
for (f in txt_files) {
  process_file(f, input_path, output_dir)
}