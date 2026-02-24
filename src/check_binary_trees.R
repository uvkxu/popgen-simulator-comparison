#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ape)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript check_binary_trees_parallel.R <input_folder> [output_csv] [n_cores]")
}

input_dir <- args[1]
output_csv <- if (length(args) >= 2) args[2] else "tree_binary_report.csv"
n_cores <- if (length(args) >= 3) as.integer(args[3]) else max(1L, parallel::detectCores() - 1L)

if (!dir.exists(input_dir)) stop("Input directory does not exist: ", input_dir)

# Strict-ish newick line: optional [123] prefix + (...) + ; (maybe multiple ;)
is_tree_line <- function(line) {
  grepl("^\\s*(\\[\\d+\\])?\\s*\\(.*\\);+\\s*$", line)
}

clean_tree_line <- function(line) {
  # keep optional [123] prefix if present; normalize trailing ;;; -> ;
  line <- trimws(line)
  sub(";+$", ";", line)
}

files <- list.files(input_dir, pattern = "\\.txt$", recursive = TRUE, full.names = TRUE)
if (length(files) == 0) stop("No .txt files found in: ", input_dir)

cat(sprintf("Found %d txt files\n", length(files)))
cat(sprintf("Using %d cores\n", n_cores))

# Worker: process one file, always return a data.frame
process_one_file <- function(file) {
  # Wrap whole worker in tryCatch so mclapply doesn't die
  tryCatch({
    lines <- readLines(file, warn = FALSE)
    if (length(lines) == 0) {
      return(data.frame(
        file = character(0),
        tree_id = integer(0),
        readable = logical(0),
        binary = logical(0),
        ultrametric = logical(0),
        zero_branches = integer(0),
        stringsAsFactors = FALSE
      ))
    }

    idx <- which(vapply(lines, is_tree_line, logical(1)))
    if (length(idx) == 0) {
      return(data.frame(
        file = file,
        tree_id = integer(0),
        readable = logical(0),
        binary = logical(0),
        ultrametric = logical(0),
        zero_branches = integer(0),
        stringsAsFactors = FALSE
      ))
    }

    out <- vector("list", length(idx))

    for (j in seq_along(idx)) {
      i <- idx[j]
      line_clean <- clean_tree_line(lines[i])

      tr <- tryCatch(
        suppressWarnings(ape::read.tree(text = line_clean)),
        error = function(e) NULL
      )

      if (is.null(tr)) {
        out[[j]] <- data.frame(
          file = file,
          tree_id = j,
          readable = FALSE,
          binary = NA,
          ultrametric = NA,
          zero_branches = NA,
          stringsAsFactors = FALSE
        )
        next
      }

      zb <- NA_integer_
      if (!is.null(tr$edge.length)) {
        zb <- sum(tr$edge.length == 0, na.rm = TRUE)
      }

      out[[j]] <- data.frame(
        file = file,
        tree_id = j,
        readable = TRUE,
        binary = suppressWarnings(ape::is.binary(tr)),       # <- replaces is.binary.tree
        ultrametric = suppressWarnings(ape::is.ultrametric(tr)),
        zero_branches = zb,
        stringsAsFactors = FALSE
      )
    }

    do.call(rbind, out)
  }, error = function(e) {
    # Return one-row error marker so you can see which file failed
    data.frame(
      file = file,
      tree_id = NA_integer_,
      readable = FALSE,
      binary = NA,
      ultrametric = NA,
      zero_branches = NA_integer_,
      stringsAsFactors = FALSE
    )
  })
}

df_list <- parallel::mclapply(files, process_one_file, mc.cores = n_cores)

# Make sure we only rbind data.frames (just in case)
df_list <- Filter(is.data.frame, df_list)
if (length(df_list) == 0) stop("No results produced (all workers failed).")

df <- do.call(rbind, df_list)

write.csv(df, output_csv, row.names = FALSE)

cat("--------------------------------------------------\n")
cat("Summary:\n")
cat(sprintf("Total trees: %d\n", nrow(df)))
cat(sprintf("Unreadable: %d\n", sum(!df$readable, na.rm = TRUE)))
cat(sprintf("Non-binary: %d\n", sum(df$readable & !df$binary, na.rm = TRUE)))
cat(sprintf("Non-ultrametric: %d\n", sum(df$readable & !df$ultrametric, na.rm = TRUE)))
cat(sprintf("Trees with zero-length branches: %d\n",
            sum(df$readable & df$zero_branches > 0, na.rm = TRUE)))
cat("--------------------------------------------------\n")
cat("Detailed report written to:", output_csv, "\n")