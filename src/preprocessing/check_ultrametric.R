#!/usr/bin/env Rscript

library(ape)
library(treestats)
library(yaml)

create_branch_set <- function(tr) {
  node_ages <- node.depth.edgelength(tr)
  max_age   <- max(node_ages)

  branchset <- vector("list", nrow(tr$edge))

  for (i in seq_len(nrow(tr$edge))) {
    parent <- tr$edge[i, 1]
    child  <- tr$edge[i, 2]
    bl     <- tr$edge.length[i]

    end_date   <- max_age - node_ages[parent]
    start_date <- end_date - bl

    branchset[[i]] <- list(
      start_date = start_date,
      end_date   = end_date,
      parent     = parent,
      child      = child
    )
  }

  branchset
}


calc_pigot_rho_diagnose <- function(tr) {

  # -------------------------
  # 1. EDGE + EDGE LENGTH
  # -------------------------
  edge <- tr$edge
  el   <- tr$edge.length

  if (is.null(edge) || is.null(el))
    stop("Tree has no edge or edge.length")

  # -------------------------
  # 2. CROWN AGE (wie C++)
  # -------------------------
  crown_age <- treestats::crown_age(tr)
  mid_point <- crown_age * 0.5

  # -------------------------
  # 3. C++ ROOT ASSUMPTION
  # -------------------------
  crown_node <- edge[1, 1]   # phy.edge[0][0]

  # -------------------------
  # 4. n3 = count_extant_lineages
  # -------------------------
  n3 <- 0L
  for (i in seq_len(nrow(edge))) {
    if (edge[i, 2] < crown_node) {
      n3 <- n3 + 1L
    }
  }

  # -------------------------
  # 5. branchset ≈ branching.times
  # -------------------------
  # create_branch_set(...) is effectively encoding lineage survival
  # branching.times gives node heights from present backwards
  branchset <- create_branch_set(tr)

  n2 <- 0L
  for (i in seq_along(branchset)) {
    if (branchset[[i]]$start_date < mid_point &&
        branchset[[i]]$end_date   >= mid_point) {
      n2 <- n2 + 1L
    }
  }

  # -------------------------
  # 7. FIXED n1
  # -------------------------
  n1 <- 2L

  # -------------------------
  # 8. r1, r2 (NO SAFETY)
  # -------------------------
  r1 <- (log(n2) - log(n1)) / mid_point
  r2 <- (log(n3) - log(n2)) / mid_point

  rho <- (r2 - r1) / (r1 + r2)

  # -------------------------
  # 9. FULL DIAGNOSTICS
  # -------------------------
  list(
    crown_age  = crown_age,
    mid_point  = mid_point,
    crown_node = crown_node,
    diag_n2 = n2,
    diag_n3 = n3,
    diag_r1 = r1,
    diag_r2 = r2,
    diag_rho = rho
  )
}


# --- Argument parsing ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript run_treestats_clean.R <input_file> <output_file> <combo_yaml>")

input_file  <- args[1]
output_file <- args[2]
combo_yaml  <- args[3]

# --- Checks ---
if (!file.exists(input_file)) stop("Input file does not exist: ", input_file)
if (!file.exists(combo_yaml)) stop("YAML config file does not exist: ", combo_yaml)

# --- Ensure output directory exists ---
output_dir <- dirname(output_file)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Read YAML config ---
combo_params <- yaml::read_yaml(combo_yaml)
required <- c("sequence_length")
missing <- setdiff(required, names(combo_params))
if (length(missing) > 0) stop("Missing fields in ", combo_yaml, ": ", paste(missing, collapse = ", "))
seq_length <- combo_params$sequence_length

# --- Helper functions ---
is_tree_line <- function(line) {
  grepl("\\(", line) & grepl(";+", line) &
    !grepl("^\\s*//", line) &
    !grepl("^\\s*segsites", line, ignore.case = TRUE) &
    !grepl("^\\s*positions", line, ignore.case = TRUE)
}

clean_tree_line <- function(line) trimws(sub(";+$", ";", line))

# --- Load existing results ---
existing_df <- if (file.exists(output_file)) read.csv(output_file, stringsAsFactors = FALSE) else data.frame()
results <- list()
skipped <- 0L

con <- file(input_file, open = "r")
while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
  if (!is_tree_line(line)) next
  line <- clean_tree_line(line)

  tr <- tryCatch(ape::read.tree(text = line), error = function(e) NULL)
  if (is.null(tr)) {
    skipped <- skipped + 1L
    next
  }

  tr_unweighted <- tr
  tr_unweighted$edge.length <- rep(1, nrow(tr_unweighted$edge))

  # --- Check if tree already exists in output ---
  existing_row <- if (nrow(existing_df) > 0) existing_df[1, FALSE] else NULL  # nur Struktur

  new_row <- list()

  # --- Compute only missing stats ---
  new_row$pigot_rho <- if (is.null(existing_row) || !"pigot_rho" %in% names(existing_df)) {
    tryCatch(treestats::pigot_rho(tr), error = function(e) NA)
  } else existing_row$pigot_rho

  # --- Diagnose Pigot rho (C++-nah) ---
  diag <- calc_pigot_rho_diagnose(tr)

  new_row$diag_n2  <- diag$diag_n2
  new_row$diag_n3  <- diag$diag_n3
  new_row$diag_r1  <- diag$diag_r1
  new_row$diag_r2  <- diag$diag_r2
  new_row$diag_rho <- diag$diag_rho
  new_row$mid_point <- diag$mid_point
  new_row$crown_node <- diag$crown_node


  new_row$num_nodes <- if (is.null(existing_row) || !"num_nodes" %in% names(existing_df)) {
    tr$Nnode
  } else existing_row$num_nodes

  new_row$is_ultrametric <- if (is.null(existing_row) || !"is_ultrametric" %in% names(existing_df)) {
    ape::is.ultrametric(tr)
  } else existing_row$is_ultrametric
   
  new_row$tree_height <- if (is.null(existing_row) || !"tree_height" %in% names(existing_df)) {
    tryCatch(treestats::tree_height(tr), error = function(e) NA)
  } else existing_row$tree_height

  new_row$mean_branch_length <- if (is.null(existing_row) || !"mean_branch_length" %in% names(existing_df)) {
    tryCatch(treestats::mean_branch_length(tr), error = function(e) NA)
  } else existing_row$mean_branch_length

  new_row$crown_age <- if (is.null(existing_row) || !"crown_age" %in% names(existing_df)) {
    diag$crown_age
  } else existing_row$crown_age

  # --- Root node check ---
  actual_root_node <- setdiff(tr$edge[,1], tr$edge[,2])
  if (length(actual_root_node) != 1) {
    warning("Root node could not be uniquely determined, setting right_root_node = FALSE")
    new_row$right_root_node <- FALSE
  } else {
    actual_root_node <- actual_root_node[1]
    root_node <- tr$edge[1,1]
    # Vergleich absichern gegen Vektor/NA
    new_row$right_root_node <- if (length(root_node) == 1 && !is.na(root_node)) {
      actual_root_node == root_node
    } else {
      FALSE
    }
  }

  # --- Right tip nodes check ---
  crown <- root_node
  tip_nodes <- seq_along(tr$tip.label)
  right_tip_nodes <- TRUE

  if (length(tip_nodes) == 0 || length(crown) != 1 || is.na(crown)) {
    right_tip_nodes <- FALSE
  } else {
    for (i in seq_len(nrow(tr$edge))) {
      child <- tr$edge[i,2]
      if (child < crown) {
        if (!(child %in% tip_nodes)) {
          right_tip_nodes <- FALSE
          break
        }
      } else {
        if (child %in% tip_nodes) {
          right_tip_nodes <- FALSE
          break
        }
      }
    }
  }

  new_row$right_tip_nodes <- right_tip_nodes



  # --- Branching times ---
  brts <- sort(branching.times(tr), decreasing = TRUE)
  new_row$n3 <- if (is.null(existing_row) || !"n3" %in% names(existing_df)) 1 + length(brts) else existing_row$n3
  mid_point <- brts[1] / 2
  new_row$n2 <- if (is.null(existing_row) || !"n2" %in% names(existing_df)) 1 + sum(brts >= mid_point) else existing_row$n2

  # --- Add constant params ---
  new_row$effective_pop_size <- combo_params$population_size
  new_row$sample_size <- combo_params$sample_structure[1]
  new_row$overall_sample_size <- combo_params$overall_samples

  results[[length(results) + 1]] <- as.data.frame(new_row, stringsAsFactors = FALSE)
}
close(con)

# --- Combine with existing dataframe ---
df_new <- do.call(rbind, results)
if (nrow(existing_df) > 0) {
  df <- rbind(existing_df, df_new)
} else {
  df <- df_new
}

write.csv(df, file = output_file, row.names = FALSE)
cat(sprintf("✅ Done: %s (%d trees skipped)\n", output_file, skipped))



