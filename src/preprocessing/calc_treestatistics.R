source("src/preprocessing/extra_features.R")

library(ape)
library(treestats)
library(igraph)
library(yaml)

# --------------------------------------------
# Robust stat wrapper from your original code
# --------------------------------------------
#' @keywords internal
try_stat <- function(phylo,
                     func,
                     normalize = FALSE,
                     norm_res = c(TRUE, FALSE)) {
  res <- NA
  if (normalize) {
    res <- tryCatch(
      expr = {
        func(
          phylo,
          normalization = ifelse(normalize, norm_res[1], norm_res[2])
        )
      },
      error = function(e) { return(NA) }
    )
  } else {
    res <- tryCatch(expr = { func(phylo) }, error = function(e) { return(NA) })
  }
  return(res)
}

calc_non_binary_node_features <- function(tr) {
  if (is.null(tr$edge) || nrow(tr$edge) == 0) {
    return(list(non_binary_nodes = NA_real_, non_binary_nodes_frac = NA_real_))
  }

  n_tip  <- length(tr$tip.label)
  n_node <- tr$Nnode

  if (is.null(n_node) || n_node <= 0) {
    return(list(non_binary_nodes = 0, non_binary_nodes_frac = 0))
  }

  internal_nodes <- (n_tip + 1):(n_tip + n_node)

  parent <- tr$edge[, 1]
  outdeg <- tabulate(match(parent, internal_nodes), nbins = length(internal_nodes))

  non_bin <- sum(outdeg != 2)
  frac <- non_bin / n_node

  list(non_binary_nodes = as.numeric(non_bin),
       non_binary_nodes_frac = as.numeric(frac))
}

# ----------------------------------------------------------
# Final calculate_all_treestats with averaging over resolutions
# ----------------------------------------------------------
calculate_all_treestats <- function(phylo,
                                   normalize = FALSE,
                                   weighted = TRUE,
                                   resolve_binary_random = TRUE,
                                   n_resolutions = 30,
                                   eps = 1e-8) {
  
                                    
  safe_any <- function(expr) {
    tryCatch(expr, error = function(e) NULL)
  }

  as_num_scalar <- function(x) {
    if (is.null(x)) return(NA_real_)
    x <- as.numeric(x)
    if (length(x) == 0) return(NA_real_)
    if (length(x) > 1) return(mean(x, na.rm = TRUE))
    x
  }

  stat_num <- function(expr) as_num_scalar(safe_any(expr))

  fix_zero_edges <- function(tr) {
    if (!is.null(tr$edge.length)) {
      tr$edge.length[tr$edge.length == 0] <- eps
    }
    tr
  }

  make_binary <- function(tr) {
    trb <- ape::multi2di(tr, random = resolve_binary_random)
    fix_zero_edges(trb)
  }

  # Use ape::is.binary() (replaces deprecated is.binary.tree())
  is_bin <- safe_any(ape::is.binary(phylo))
  if (is.null(is_bin) || length(is_bin) != 1) is_bin <- FALSE

  need_binary_avg <- (!is_bin) && (n_resolutions > 1)

  compute_weighted_once <- function(tr_orig, tr_bin) {
    stats <- list()

    # --- original-ok ---
    nb <- calc_non_binary_node_features(tr_orig)
    stats$non_binary_nodes <- nb$non_binary_nodes
    stats$non_binary_nodes_frac <- nb$non_binary_nodes_frac
    
    stats$zero_branch_count <- as_num_scalar(sum(tr_orig$edge.length == 0))

    stats$crown_age          <- stat_num(treestats::crown_age(tr_orig))
    stats$tree_height        <- stat_num(treestats::tree_height(tr_orig))
    stats$pigot_rho          <- stat_num(as.numeric(calc_pigot_rho(tr_orig)))

    stats$number_of_lineages <- stat_num(treestats::number_of_lineages(tr_orig))
    stats$phylogenetic_div   <- stat_num(treestats::phylogenetic_diversity(tr_orig))

    lap_spec <- safe_any(treestats::laplacian_spectrum(tr_orig))
    if (!is.null(lap_spec) && length(lap_spec) == 5) {
      stats$laplace_spectrum_a <- as_num_scalar(lap_spec$asymmetry)
      stats$laplace_spectrum_p <- as_num_scalar(lap_spec$peakedness)
      stats$laplace_spectrum_e <- as_num_scalar(log(lap_spec$principal_eigenvalue))
      stats$laplace_spectrum_g <- as_num_scalar(lap_spec$eigengap[[1]])
    } else {
      stats$laplace_spectrum_a <- NA_real_
      stats$laplace_spectrum_p <- NA_real_
      stats$laplace_spectrum_e <- NA_real_
      stats$laplace_spectrum_g <- NA_real_
    }

    stats$mpd <- stat_num(
      if (normalize) treestats::mean_pair_dist(tr_orig, normalization = "tips")
      else treestats::mean_pair_dist(tr_orig)
    )
    stats$psv <- stat_num(
      if (normalize) treestats::psv(tr_orig, normalization = "tips")
      else treestats::psv(tr_orig)
    )
    stats$mntd   <- stat_num(treestats::mntd(tr_orig))
    stats$j_stat <- stat_num(treestats::entropy_j(tr_orig))

    stats$rquartet <- stat_num(
      if (normalize) treestats::rquartet(tr_orig, normalization = "yule")
      else treestats::rquartet(tr_orig)
    )

    stats$mean_branch_length <- stat_num(treestats::mean_branch_length(tr_orig))
    stats$var_branch_length  <- stat_num(treestats::var_branch_length(tr_orig))

    stats$mean_branch_length_int <- stat_num(treestats::mean_branch_length_int(tr_orig))
    stats$mean_branch_length_ext <- stat_num(treestats::mean_branch_length_ext(tr_orig))
    stats$var_branch_length_int  <- stat_num(treestats::var_branch_length_int(tr_orig))
    stats$var_branch_length_ext  <- stat_num(treestats::var_branch_length_ext(tr_orig))

    stats$treeness <- stat_num(treestats::treeness(tr_orig))
    stats$tot_internal_path <- stat_num(treestats::tot_internal_path(tr_orig))

    # --- binary-required / safer-on-binary ---
    stats$sackin <- stat_num(
      if (normalize) treestats::sackin(tr_bin, normalization = "yule")
      else treestats::sackin(tr_bin)
    )

    stats$colless <- stat_num(
      if (normalize) treestats::colless(tr_bin, normalization = "yule")
      else treestats::colless(tr_bin)
    )
    stats$colless_corr <- stat_num(
      if (normalize) treestats::colless_corr(tr_bin, normalization = "yule")
      else treestats::colless_corr(tr_bin)
    )
    stats$colless_quad <- stat_num(
      if (normalize) treestats::colless_quad(tr_bin, normalization = "yule")
      else treestats::colless_quad(tr_bin)
    )

    stats$beta <- stat_num(treestats::beta_statistic(tr_bin))

    stats$blum <- stat_num(
      if (normalize) treestats::blum(tr_bin, normalization = TRUE)
      else treestats::blum(tr_bin)
    )

    stats$avg_ladder <- stat_num(treestats::avg_ladder(tr_bin))
    stats$max_ladder <- stat_num(treestats::max_ladder(tr_bin))

    stats$cherries <- stat_num(
      if (normalize) treestats::cherries(tr_bin, normalization = "yule")
      else treestats::cherries(tr_bin)
    )

    stats$double_cherries <- stat_num(treestats::double_cherries(tr_bin))
    stats$four_prong      <- stat_num(treestats::four_prong(tr_bin))

    stats$il_number <- stat_num(
      if (normalize) treestats::ILnumber(tr_orig, normalization = "tips")
      else treestats::ILnumber(tr_orig)
    )

    stats$stairs  <- stat_num(treestats::stairs(tr_bin))
    stats$stairs2 <- stat_num(treestats::stairs2(tr_bin))

    mm_lap <- safe_any(treestats::minmax_laplace(tr_bin, TRUE))
    stats$min_laplace <- if (is.null(mm_lap)) NA_real_ else as_num_scalar(mm_lap$min)
    stats$max_laplace <- if (is.null(mm_lap)) NA_real_ else as_num_scalar(mm_lap$max)

    mm_adj <- safe_any(treestats::minmax_adj(tr_bin, TRUE))
    stats$min_adj <- if (is.null(mm_adj)) NA_real_ else as_num_scalar(mm_adj$min)
    stats$max_adj <- if (is.null(mm_adj)) NA_real_ else as_num_scalar(mm_adj$max)

    stats$j_one <- stat_num(treestats::j_one(tr_bin))

    stats$b1 <- stat_num(
      if (normalize) treestats::b1(tr_bin, normalization = "tips")
      else treestats::b1(tr_bin)
    )
    stats$b2 <- stat_num(
      if (normalize) treestats::b2(tr_bin, normalization = "yule")
      else treestats::b2(tr_bin)
    )

    stats$area_per_pair <- stat_num(
      if (normalize) treestats::area_per_pair(tr_bin, normalization = "yule")
      else treestats::area_per_pair(tr_bin)
    )
    stats$average_leaf_depth <- stat_num(
      if (normalize) treestats::average_leaf_depth(tr_bin, normalization = "yule")
      else treestats::average_leaf_depth(tr_bin)
    )

    stats$i_stat     <- stat_num(treestats::mean_i(tr_bin))
    stats$ew_colless <- stat_num(treestats::ew_colless(tr_bin))

    stats$max_del_width <- stat_num(
      if (normalize) treestats::max_del_width(tr_bin, normalization = "tips")
      else treestats::max_del_width(tr_bin)
    )
    stats$max_depth <- stat_num(
      if (normalize) treestats::max_depth(tr_bin, normalization = "tips")
      else treestats::max_depth(tr_bin)
    )
    stats$avg_vert_depth <- stat_num(treestats::avg_vert_depth(tr_bin))
    stats$max_width <- stat_num(
      if (normalize) treestats::max_width(tr_bin, normalization = "tips")
      else treestats::max_width(tr_bin)
    )
    stats$mw_over_md <- stat_num(treestats::mw_over_md(tr_bin))

    stats$tot_path <- stat_num(treestats::tot_path_length(tr_bin))

    stats$rogers <- stat_num(
      if (normalize) treestats::rogers(tr_bin, normalization = "tips")
      else treestats::rogers(tr_bin)
    )
    stats$tot_coph <- stat_num(
      if (normalize) treestats::tot_coph(tr_bin, normalization = "yule")
      else treestats::tot_coph(tr_bin)
    )

    stats$symmetry_nodes <- stat_num(
      if (normalize) treestats::sym_nodes(tr_bin, normalization = "tips")
      else treestats::sym_nodes(tr_bin)
    )

    stats$vpd <- stat_num(treestats::var_pair_dist(tr_bin))

    stats$wiener   <- stat_num(treestats::wiener(tr_bin, normalize))
    stats$diameter <- stat_num(treestats::diameter(tr_bin))

    stats$max_betweenness <- stat_num(
      if (normalize) treestats::max_betweenness(tr_bin, normalization = "tips")
      else treestats::max_betweenness(tr_bin)
    )

    stats$max_closeness <- stat_num(
      treestats::max_closeness(tr_bin, weight = FALSE,
                               normalization = if (normalize) "tips" else "none")
    )
    stats$max_closenessW <- stat_num(
      treestats::max_closeness(tr_bin, weight = TRUE,
                               normalization = if (normalize) "tips" else "none")
    )

    eig0 <- safe_any(treestats::eigen_centrality(tr_bin, weight = FALSE))
    stats$eigen_centrality <- if (is.null(eig0)) NA_real_ else as_num_scalar(max(eig0$eigenvector))

    eigW <- safe_any(treestats::eigen_centrality(tr_bin, weight = TRUE))
    stats$eigen_centralityW <- if (is.null(eigW)) NA_real_ else as_num_scalar(max(eigW$eigenvector))

    stats$root_imbalance <- stat_num(treestats::root_imbalance(tr_bin))

    lap_stats <- safe_any(stats_laplace(tr_bin, weighted = TRUE))
    if (is.null(lap_stats)) {
      lap_stats <- data.frame(
        lap_max = NA, lap_fiedler = NA, lap_var = NA, lap_skew = NA, lap_mean = NA,
        lap_kurt = NA, lap_second_min = NA, lap_entropy = NA, lap_sum = NA,
        cut_weight = NA, conductance = NA, normalized_cut = NA, bottleneck_strength = NA,
        density_pos = NA, density_neg = NA, modularity = NA, fiedler_sd = NA
      )
    }
    stats <- c(stats, as.list(lap_stats))

    stats <- lapply(stats, as_num_scalar)
    stats <- unlist(stats)
    stats <- stats[order(names(stats))]
    stats
  }

  compute_unweighted_once <- function(tr_orig, tr_bin) {
    stats <- list()

    stats$tree_height_uw <- stat_num(treestats::tree_height(tr_orig))
    stats$num_branches   <- stat_num(treestats::phylogenetic_diversity(tr_orig))

    lap_spec <- safe_any(treestats::laplacian_spectrum(tr_orig))
    if (!is.null(lap_spec) && length(lap_spec) == 5) {
      stats$laplace_spectrum_a_uw <- as_num_scalar(lap_spec$asymmetry)
      stats$laplace_spectrum_p_uw <- as_num_scalar(lap_spec$peakedness)
      stats$laplace_spectrum_e_uw <- as_num_scalar(log(lap_spec$principal_eigenvalue))
      stats$laplace_spectrum_g_uw <- as_num_scalar(lap_spec$eigengap[[1]])
    } else {
      stats$laplace_spectrum_a_uw <- NA_real_
      stats$laplace_spectrum_p_uw <- NA_real_
      stats$laplace_spectrum_e_uw <- NA_real_
      stats$laplace_spectrum_g_uw <- NA_real_
    }

    mm_lap <- safe_any(treestats::minmax_laplace(tr_bin, TRUE))
    stats$min_laplace_uw <- if (is.null(mm_lap)) NA_real_ else as_num_scalar(mm_lap$min)
    stats$max_laplace_uw <- if (is.null(mm_lap)) NA_real_ else as_num_scalar(mm_lap$max)

    mm_adj <- safe_any(treestats::minmax_adj(tr_bin, TRUE))
    stats$min_adj_uw <- if (is.null(mm_adj)) NA_real_ else as_num_scalar(mm_adj$min)
    stats$max_adj_uw <- if (is.null(mm_adj)) NA_real_ else as_num_scalar(mm_adj$max)

    stats$mpd_uw <- stat_num(
      if (normalize) treestats::mean_pair_dist(tr_orig, normalization = "tips")
      else treestats::mean_pair_dist(tr_orig)
    )
    stats$psv_uw <- stat_num(
      if (normalize) treestats::psv(tr_orig, normalization = "tips")
      else treestats::psv(tr_orig)
    )

    stats$vpd_uw  <- stat_num(treestats::var_pair_dist(tr_bin))
    stats$mntd_uw <- stat_num(treestats::mntd(tr_orig))

    stats$j_stat_uw <- stat_num(treestats::entropy_j(tr_orig))
    stats$wiener_uw  <- stat_num(treestats::wiener(tr_bin, normalize))

    stats$max_closenessW_uw <- stat_num(
      treestats::max_closeness(tr_bin, weight = TRUE,
                               normalization = if (normalize) "tips" else "none")
    )

    eigW <- safe_any(treestats::eigen_centrality(tr_bin, weight = TRUE))
    stats$eigen_centralityW_uw <- if (is.null(eigW)) NA_real_ else as_num_scalar(max(eigW$eigenvector))

    stats$treeness_uw <- stat_num(treestats::treeness(tr_orig))

    stats <- lapply(stats, as_num_scalar)
    stats <- unlist(stats)
    stats <- stats[order(names(stats))]
    stats
  }

  if (!need_binary_avg) {
    tr_bin <- make_binary(phylo)
    if (weighted) return(compute_weighted_once(phylo, tr_bin))
    return(compute_unweighted_once(phylo, tr_bin))
  }

  reps <- vector("list", n_resolutions)
  for (i in seq_len(n_resolutions)) {
    tr_bin <- make_binary(phylo)
    reps[[i]] <- if (weighted) compute_weighted_once(phylo, tr_bin) else compute_unweighted_once(phylo, tr_bin)
  }

  all_names <- sort(unique(unlist(lapply(reps, names))))
  mat <- do.call(rbind, lapply(reps, function(v) {
    out <- rep(NA_real_, length(all_names))
    names(out) <- all_names
    out[names(v)] <- v
    out
  }))

  mu <- apply(mat, 2, function(v) mean(v, na.rm = TRUE))
  mu <- mu[order(names(mu))]
  mu
}

# --------------------------------------------
# --- Argument parsing ---
# --------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_treestats_clean.R <input_file> <output_file> <combo_yaml>")
}

input_file  <- args[1]
output_file <- args[2]
combo_yaml  <- args[3]

# --------------------------------------------
# --- Checks ---
# --------------------------------------------
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

combo_params <- yaml::read_yaml(combo_yaml)
required <- c("sequence_length")
missing <- setdiff(required, names(combo_params))
if (length(missing) > 0) {
  stop("Missing fields in ", combo_yaml, ": ", paste(missing, collapse = ", "))
}

seq_length <- combo_params$sequence_length

has_recomb <- "recombination_rate" %in% names(combo_params) &&
  !is.null(combo_params$recombination_rate) &&
  combo_params$recombination_rate != 0

# Optional YAML override (default 20)
n_resolutions <- 30
if ("n_resolutions" %in% names(combo_params) && !is.null(combo_params$n_resolutions)) {
  n_resolutions <- as.integer(combo_params$n_resolutions)
  if (is.na(n_resolutions) || n_resolutions < 1) n_resolutions <- 20
}

# --------------------------------------------
# Helpers for parsing tree lines
# --------------------------------------------
is_tree_line <- function(line) {
  grepl("\\(", line) & grepl(";+", line) &
    !grepl("^\\s*//", line) &
    !grepl("^\\s*segsites", line, ignore.case = TRUE) &
    !grepl("^\\s*positions", line, ignore.case = TRUE)
}

clean_tree_line <- function(line) {
  trimws(sub(";+$", ";", line))
}

skipped <- 0L
header_written <- FALSE

# --------------------------------------------
# Main loop
# --------------------------------------------
con <- file(input_file, open = "r")
while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
  if (!is_tree_line(line)) next

  line <- clean_tree_line(line)

  tr <- ape::read.tree(text = line)
  if (is.null(tr)) {
    skipped <- skipped + 1L
    next
  }

  tr_unweighted <- tr
  tr_unweighted$edge.length <- rep(1, nrow(tr_unweighted$edge))

  tree_stats <- calculate_all_treestats(
    tr,
    normalize = FALSE,
    weighted = TRUE,
    n_resolutions = n_resolutions
  )
  tree_stats_unweighted <- calculate_all_treestats(
    tr_unweighted,
    normalize = FALSE,
    weighted = FALSE,
    n_resolutions = n_resolutions
  )

  if (is.null(tree_stats) || is.null(tree_stats_unweighted)) {
    skipped <- skipped + 1L
    next
  }

  combined <- as.data.frame(as.list(c(tree_stats, tree_stats_unweighted)))

  if (has_recomb) {
    segment_length <- as.numeric(gsub("^\\[(\\d+)\\].*$", "\\1", line))
    combined$segment_length <- segment_length / seq_length
  }

  df <- as.data.frame(as.list(combined), stringsAsFactors = FALSE)

  if (!header_written) {
    write.csv(df, file = output_file, row.names = FALSE)
    header_written <- TRUE
  } else {
    write.table(
      df,
      file = output_file,
      append = TRUE,
      sep = ",",
      row.names = FALSE,
      col.names = FALSE,
      quote = TRUE,
      na = "NA"
    )
  }
}
close(con)

cat(sprintf("✅ Done: %s (%d trees skipped)\n", output_file, skipped))