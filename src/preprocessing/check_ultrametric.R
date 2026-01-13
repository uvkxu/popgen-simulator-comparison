#!/usr/bin/env Rscript

library(ape)
library(treestats)
library(yaml)

# --- Argument parsing ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_treestats_clean.R <input_file> <output_file> <combo_yaml>")
}

input_file  <- args[1]
output_file <- args[2]
combo_yaml  <- args[3]

# --- Checks ---
if (!file.exists(input_file)) {
  stop("Input file does not exist: ", input_file)
}

if (!file.exists(combo_yaml)) {
  stop("YAML config file does not exist: ", combo_yaml)
}

# --- Ensure output directory exists ---
output_dir <- dirname(output_file)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

# --- Read YAML config ---
combo_params <- yaml::read_yaml(combo_yaml)
required <- c("sequence_length")
missing <- setdiff(required, names(combo_params))
if (length(missing) > 0) {
  stop("Missing fields in ", combo_yaml, ": ", paste(missing, collapse = ", "))
}

seq_length <- combo_params$sequence_length

# Check for recombination (optional)
has_recomb <- "recombination_rate" %in% names(combo_params) &&
               !is.null(combo_params$recombination_rate) &&
               combo_params$recombination_rate != 0

# --- Helper functions ---
is_tree_line <- function(line) {
  grepl("\\(", line) & grepl(";+", line) &
  !grepl("^\\s*//", line) &
  !grepl("^\\s*segsites", line, ignore.case = TRUE) &
  !grepl("^\\s*positions", line, ignore.case = TRUE)
}

clean_tree_line <- function(line) {
  trimws(sub(";+$", ";", line))
}

# --- Main loop ---
results <- list()
skipped <- 0L

con <- file(input_file, open = "r")
while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
  if (!is_tree_line(line)) next
  line <- clean_tree_line(line)
  
  tr <- tryCatch(ape::read.tree(text = line), error = function(e) NULL)
  tr_unweighted <- tr
  tr_unweighted$edge.length <- rep(1, nrow(tr_unweighted$edge))

  if (is.null(tr)) {
    skipped <- skipped + 1L
    next
  }

  pigot_rho <- tryCatch(treestats::pigot_rho(tr), error = function(e) NA)
  num_branches <- tryCatch(treestats::phylogenetic_diversity(tr_unweighted), error = function(e) NA)
  is_ultrametric <- ape::is.ultrametric(tr, tol = 1e-7)
  effective_pop_size <- combo_params$population_size
  sample_size <- combo_params$sample_structure[1]
  overall_sample_size <- combo_params$overall_samples
  tree_height <- tryCatch(treestats::tree_height(tr), error = function(e) NA)
  mean_branch_length <- tryCatch(treestats::mean_branch_length(tr), error = function(e) NA)
  crown_age <- tryCatch(treestats::crown_age(tr), error = function(e) NA)
  # brts: Branching Times, absteigend sortiert
  brts <- sort(branching.times(tr), decreasing = TRUE)

  n3 <- 1 + length(brts)
  mid_point <- brts[1] / 2

  # index_mid_point = Anzahl Werte >= mid_point
  index_mid_point <- sum(brts >= mid_point)

  n2 <- 1 + index_mid_point




  results[[length(results) + 1]] <- data.frame(
    pigot_rho = pigot_rho,
    is_ultrametric = is_ultrametric,
    effective_pop_size = effective_pop_size,
    num_branches = num_branches,
    sample_size = sample_size,
    overall_sample_size = overall_sample_size,
    tree_height = tree_height,
    mean_branch_length = mean_branch_length,
    crown_age = crown_age,
    stringsAsFactors = FALSE,
    n2 = n2,
    n3 = n3
  )
}
close(con)

# --- Combine and write results ---
df <- do.call(rbind, results)
write.csv(df, file = output_file, row.names = FALSE)

cat(sprintf("✅ Done: %s (%d trees skipped)\n", output_file, skipped))
