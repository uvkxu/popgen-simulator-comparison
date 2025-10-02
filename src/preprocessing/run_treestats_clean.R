#!/usr/bin/env Rscript

# --- Argument parsing ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_treestats_clean.R <input_txt|input_dir> <output_dir>")

input_path <- args[1]
output_dir <- args[2]

if (!file.exists(input_path)) stop("Input file/directory does not exist: ", input_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

library(ape)
library(treestats)

# --- Collect input files recursively ---
if (dir.exists(input_path)) {
  txt_files <- list.files(input_path, pattern = "\\.txt$", full.names = TRUE, recursive = TRUE)
  if (length(txt_files) == 0) stop("No .txt files found in directory: ", input_path)
} else {
  txt_files <- input_path
}

# --- Function to process a single input file ---
process_file <- function(infile, input_base, output_base) {
  # Compute relative path to preserve folder structure
  rel_path <- sub(paste0("^", normalizePath(input_base, winslash = "/"), "/?"), "", normalizePath(infile, winslash = "/"))
  outfile <- file.path(output_base, paste0(tools::file_path_sans_ext(rel_path), "_stats.csv"))
  
  # Ensure output directory exists
  dir.create(dirname(outfile), recursive = TRUE, showWarnings = FALSE)
  
  # Read all lines of the input text file
  lines <- readLines(infile, warn = FALSE)
  
  # Identify tree lines
  is_tree_line <- grepl("\\(", lines) & grepl(";+", lines) &
                  !grepl("^\\s*//", lines) &
                  !grepl("^\\s*segsites", lines, ignore.case = TRUE) &
                  !grepl("^\\s*positions", lines, ignore.case = TRUE)
  
  tree_lines_raw <- lines[is_tree_line]
  clean_line <- function(line) { line <- trimws(line); sub(";+$", ";", line) }
  tree_lines <- vapply(tree_lines_raw, clean_line, FUN.VALUE = character(1))
  
  results_list <- list()
  skipped <- 0L
  
  for (i in seq_along(tree_lines)) {
    txt <- tree_lines[i]
    tr <- tryCatch(ape::read.tree(text = txt), error = function(e) e)
    if (inherits(tr, "error")) { skipped <- skipped + 1L; next }
    
    stats <- tryCatch(treestats::calc_all_stats(tr), error = function(e) { skipped <- skipped + 1L; NULL })
    if (is.null(stats)) next
    results_list[[length(results_list) + 1L]] <- stats
  }
  
  if (length(results_list) == 0L) {
    warning("No trees processed successfully in ", infile)
  } else {
    stats_df <- do.call(rbind, results_list)
    stats_df <- stats_df[, colSums(!is.na(stats_df)) > 0, drop = FALSE]
    write.csv(stats_df, file = outfile, row.names = FALSE)
    cat(sprintf("Done: %s -> %s (processed %d trees, skipped %d)\n",
                infile, outfile, length(results_list), skipped))
  }
}

# --- Process all files ---
for (f in txt_files) {
  process_file(f, input_path, output_dir)
}


