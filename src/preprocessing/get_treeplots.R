#!/usr/bin/env Rscript

# --- Argument parsing ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript save_trees_as_png.R <input_txt|input_dir> <output_dir>")

input_path <- args[1]
output_dir <- args[2]

if (!file.exists(input_path)) stop("Input file/directory does not exist: ", input_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

library(ape)

# --- Collect input files recursively ---
if (dir.exists(input_path)) {
  txt_files <- list.files(input_path, pattern = "\\.txt$", full.names = TRUE, recursive = TRUE)
  if (length(txt_files) == 0) stop("No .txt files found in directory: ", input_path)
} else {
  txt_files <- input_path
}

# --- Function to visualize all trees in one file ---
process_file <- function(infile, input_base, output_base) {
  rel_path <- sub(paste0("^", normalizePath(input_base, winslash = "/"), "/?"), "", normalizePath(infile, winslash = "/"))
  out_subdir <- file.path(output_base, tools::file_path_sans_ext(rel_path))
  dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)
  
  lines <- readLines(infile, warn = FALSE)
  
  # Identify tree lines (Newick format)
  is_tree_line <- grepl("\\(", lines) & grepl(";+", lines) &
    !grepl("^\\s*//", lines) & 
    !grepl("^\\s*segsites", lines, ignore.case = TRUE) &
    !grepl("^\\s*positions", lines, ignore.case = TRUE)
  
  tree_lines_raw <- lines[is_tree_line]
  clean_line <- function(line) trimws(sub(";+$", ";", line))
  tree_lines <- vapply(tree_lines_raw, clean_line, FUN.VALUE = character(1))
  
  skipped <- 0L
  
  for (i in seq_along(tree_lines)) {
    txt <- tree_lines[i]
    tr <- tryCatch(read.tree(text = txt), error = function(e) e)
    if (inherits(tr, "error")) { skipped <- skipped + 1L; next }
    
    outfile <- file.path(out_subdir, sprintf("tree_%03d.png", i))
    png(outfile, width = 600, height = 600)
    plot(tr, main = sprintf("Tree %d from %s", i, basename(infile)), edge.width = 2, cex = 0.8)
    dev.off()
  }
  
  cat(sprintf("Saved %d tree plots to %s (skipped %d)\n", length(tree_lines) - skipped, out_subdir, skipped))
}

# --- Process all files ---
for (f in txt_files) {
  process_file(f, input_path, output_dir)
}
