#!/usr/bin/env Rscript
source("src/preprocessing/laplace_matrix_stats.R")

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

# --- Feature functions ---
compute_adjacency_features <- function(tr) {
  g <- ape::as.igraph.phylo(tr, directed = FALSE)
  degs <- degree(g)
  bc <- betweenness(g)
  ec <- eigen_centrality(g)$vector
  data.frame(
    adj_sum = sum(degs),
    adj_max = max(degs),
    adj_var_degree = var(degs),
    adj_min_max_ratio = min(degs)/max(degs),
    adj_eigen_centrality_max = max(ec),
    adj_betweenness_max = max(bc)
  )
}

compute_subtree_features <- function(tr) {
  data.frame(
    pitchfork_freq = sum(table(tr$edge[,1]) == 3)/Ntip(tr),
    quartet_imbalance = tryCatch(treestats::q_stat(tr), error=function(e) NA),
    cherries_ratio = tryCatch(treestats::cherries(tr)/Ntip(tr), error=function(e) NA)
  )
}

compute_path_depth_features <- function(tr) {
  depths <- node.depth(tr)
  leaf_depths <- depths[1:Ntip(tr)]
  data.frame(
    median_depth_to_root = median(leaf_depths),
    var_leaf_depths = var(leaf_depths),
    mw_md_ratio = max(table(depths))/max(depths)
  )
}

compute_char_vertex_features <- function(tr) {
  # --- Graph erstellen mit gewichteten Kanten ---
  g <- ape::as.igraph.phylo(tr, directed = FALSE)
  if (!is.null(tr$edge.length)) {
    E(g)$weight <- tr$edge.length
  } else {
    E(g)$weight <- 1
  }

  # --- Laplace-Matrix ---
  mat_size <- max(tr$edge)
  lap_mat <- matrix(0, mat_size, mat_size)
  lap_mat[tr$edge] <- ifelse(rep(TRUE, mat_size - 1), # weight is always true
                                tr$edge.length, 1)
  degs <- rowSums(lap_mat)
  lap_mat <- diag(degs) - lap_mat

  # --- Eigenzerlegung ---
  eig <- eigen(lap_mat, symmetric = TRUE)
  vals <- eig$values
  vecs <- eig$vectors
  fiedler_val <- min(vals[vals > 0])
  fiedler_idx <- which(vals == fiedler_val)[1]
  fiedler_vec <- vecs[, fiedler_idx]

  # --- Characteristic vertices bestimmen ---
  char_vertices <- NULL
  if (any(fiedler_vec == 0)) {
    # Type I
    tree_type <- 1
    zero_idx <- which(fiedler_vec == 0)
    char_vertex <- NA
    for (v in zero_idx) {
      neighbors <- as.integer(igraph::neighbors(g, v))
      if (any(fiedler_vec[neighbors] != 0)) { 
        char_vertex <- v
        break 
      }
    }
    char_vertices <- char_vertex
  } else {
    # Type II
    tree_type <- 2
    pos_idx <- which(fiedler_vec > 0)
    neg_idx <- which(fiedler_vec < 0)
    for (i in pos_idx) {
      for (j in neg_idx) {
        if (are.connected(g, i, j)) {
          char_vertices <- c(i, j)
          break
        }
      }
      if (!is.null(char_vertices)) break
    }
  }

  if (is.null(char_vertices)) {
    stop("❌ No characteristic vertices found, although tree is connected. This should not happen.")
  }

  # --- Featureberechnung ---
  leaves <- 1:length(tr$tip.label)

  # Gewichtete Distanzen!
  dist_to_leaves <- igraph::distances(g, v = char_vertices, to = leaves, weights = E(g)$weight)

  if (length(char_vertices) == 1) {
    dist_between_chars <- 0
    num_pendant_zeros <- sum(fiedler_vec == 0)
  } else {
    dist_between_chars <- igraph::distances(
      g,
      v = char_vertices[1],
      to = char_vertices[2],
      weights = E(g)$weight
    )
    num_pendant_zeros <- 0
  }

  df <- data.frame(
    tree_type = tree_type,
    dist_to_leaves_mean = mean(dist_to_leaves),
    dist_to_leaves_max  = max(dist_to_leaves),
    dist_between_chars = as.numeric(dist_between_chars),
    num_pendant_zeros = num_pendant_zeros
  )

  return(df)
}


# --- Main file processor ---
process_file <- function(infile, input_base, output_base) {
  rel_path <- sub(paste0("^", normalizePath(input_base, winslash = "/"), "/?"), "", normalizePath(infile, winslash = "/"))
  outfile <- file.path(output_base, paste0(tools::file_path_sans_ext(rel_path), "_stats.csv"))
  dir.create(dirname(outfile), recursive = TRUE, showWarnings = FALSE)

  lines <- readLines(infile, warn = FALSE)
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
    if (inherits(tr, "error")) { skipped <- skipped + 1L; next }

    tree_stats <- tryCatch(treestats::calc_all_stats(tr), error = function(e) NULL)
    if (is.null(tree_stats)) next
    if (!is.data.frame(tree_stats)) tree_stats <- as.data.frame(tree_stats)
    if (nrow(tree_stats) != 1) tree_stats <- data.frame(t(tree_stats))

    lap_stats <- tryCatch(
      stats_laplace(tr),
      error = function(e) {
        message("Laplace failed for tree ", i, ": ", e$message)
        list(mean = NA, var = NA, skew = NA, kurt = NA, fiedler = NA, entropy = NA, sum = NA)
      }
    )
    lap_df <- data.frame(
      laplace_mean = lap_stats$mean,
      laplace_var  = lap_stats$var,
      laplace_skew = lap_stats$skew,
      laplace_kurt = lap_stats$kurt,
      laplace_fiedler = lap_stats$fiedler,
      laplace_entropy = lap_stats$entropy,
      laplace_sum = lap_stats$sum
    )

    char_vertex_df <- tryCatch(
      compute_char_vertex_features(tr),
      error = function(e) {
        message("⚠️ Characteristic vertex computation failed for tree ", i, ": ", e$message)
        data.frame(
          tree_type = NA,
          dist_to_leaves_mean = NA,
          dist_to_leaves_max = NA,
          dist_between_chars = NA,
          num_pendant_zeros = NA
        )
      }
    )

    lap_extra <- tryCatch(compute_laplacian_extra(tr), error = function(e) data.frame(lap_fiedler = NA, lap_sum = NA, lap_entropy = NA))
    adj_feats <- tryCatch(compute_adjacency_features(tr), error = function(e) data.frame(adj_sum = NA, adj_max = NA, adj_var_degree = NA, adj_min_max_ratio = NA, adj_eigen_centrality_max = NA, adj_betweenness_max = NA))
    subtree_feats <- tryCatch(compute_subtree_features(tr), error = function(e) data.frame(pitchfork_freq = NA, quartet_imbalance = NA, cherries_ratio = NA))
    path_feats <- tryCatch(compute_path_depth_features(tr), error = function(e) data.frame(median_depth_to_root = NA, var_leaf_depths = NA, mw_md_ratio = NA))

    combined <- cbind(tree_stats, lap_df, lap_extra, adj_feats, subtree_feats, path_feats, char_vertex_df)
    results_list[[length(results_list) + 1L]] <- combined
  }

  if (length(results_list) == 0L) {
    warning("No trees processed successfully in ", infile)
  } else {
    stats_df <- do.call(rbind, results_list)
    write.csv(stats_df, file = outfile, row.names = FALSE)
    cat(sprintf("✅ Done: %s -> %s (processed %d trees, skipped %d)\n", infile, outfile, length(results_list), skipped))
  }
}

# --- Process all files ---
for (f in txt_files) {
  process_file(f, input_path, output_dir)
}
