source("src/preprocessing/extra_features.R")

library(ape)
library(treestats)
library(igraph)
library(yaml)

calculate_all_treestats <- function(phylo, normalize = FALSE, weighted = TRUE) {

  if (weighted) {
    stats <- list()
    
    stats$pigot_rho <- tryCatch(
      as.numeric(calc_pigot_rho(phylo)),
      error = function(e) NA_real_
    )

    stats$zero_branch_count <- sum(phylo$edge.length == 0)

    stats$sackin             <- try_stat(phylo, treestats::sackin, normalize, c("yule", "none"))

    stats$colless            <- try_stat(phylo, treestats::colless, normalize, c("yule", "none"))
    stats$colless_corr       <- try_stat(phylo, treestats::colless_corr, normalize, c("yule", "none"))
    stats$colless_quad       <- try_stat(phylo, treestats::colless_quad, normalize, c("yule", "none"))

    stats$beta               <- try_stat(phylo, treestats::beta_statistic)

    stats$blum               <- try_stat(phylo, treestats::blum, normalize, c(TRUE, FALSE))

    stats$crown_age          <- try_stat(phylo, treestats::crown_age)
    stats$tree_height        <- try_stat(phylo, treestats::tree_height)

    stats$number_of_lineages <- try_stat(phylo, treestats::number_of_lineages)
    stats$phylogenetic_div   <- try_stat(phylo, treestats::phylogenetic_diversity)

    stats$avg_ladder         <- try_stat(phylo, treestats::avg_ladder)
    stats$max_ladder         <- try_stat(phylo, treestats::max_ladder)

    stats$cherries           <- try_stat(phylo, treestats::cherries, normalize, c("yule", "none"))

    stats$double_cherries    <- try_stat(phylo, treestats::double_cherries)
    stats$four_prong         <- try_stat(phylo, treestats::four_prong)

    stats$il_number          <- try_stat(phylo, treestats::ILnumber, normalize, c("tips", "none"))

    stats$stairs             <- try_stat(phylo, treestats::stairs)

    temp_stats <- tryCatch(expr = { treestats::laplacian_spectrum(phylo) },
                           error = function(e) { return(NA) })

    if (length(temp_stats) == 5) {
      stats$laplace_spectrum_a  <- temp_stats$asymmetry
      stats$laplace_spectrum_p  <- temp_stats$peakedness
      stats$laplace_spectrum_e  <- log(temp_stats$principal_eigenvalue)
      stats$laplace_spectrum_g  <- temp_stats$eigengap[[1]]
    } else {
      stats$laplace_spectrum_a  <- NA
      stats$laplace_spectrum_p  <- NA
      stats$laplace_spectrum_e  <- NA
      stats$laplace_spectrum_g  <- NA
    }

    temp_stats <- try_stat(phylo, function(x) treestats::minmax_laplace(x, TRUE))
    if (length(temp_stats) >= 2) {
      stats$min_laplace <- temp_stats$min
      stats$max_laplace <- temp_stats$max
    } else {
      stats$min_laplace <- NA
      stats$max_laplace <- NA
    }

    temp_stats <- try_stat(phylo, function(x) treestats::minmax_adj(x, TRUE))
    if (length(temp_stats) >= 2) {
      stats$min_adj <- temp_stats$min
      stats$max_adj <- temp_stats$max
    } else {
      stats$min_adj <- NA
      stats$max_adj <- NA
    }

    stats$j_one              <- try_stat(phylo, treestats::j_one)

    stats$b1                 <- try_stat(phylo, treestats::b1, normalize, c("tips", "none"))
    stats$b2                 <- try_stat(phylo, treestats::b2, normalize, c("yule", "none"))

    stats$area_per_pair      <- try_stat(phylo, treestats::area_per_pair, normalize, c("yule", "none"))
    stats$average_leaf_depth <- try_stat(phylo, treestats::average_leaf_depth, normalize, c("yule", "none"))

    stats$i_stat             <- try_stat(phylo, treestats::mean_i)

    stats$ew_colless         <- try_stat(phylo, treestats::ew_colless)

    stats$max_del_width      <- try_stat(phylo, treestats::max_del_width, normalize, c("tips", "none"))
    stats$max_depth          <- try_stat(phylo, treestats::max_depth, normalize, c("tips", "none"))
    stats$avg_vert_depth     <- try_stat(phylo, treestats::avg_vert_depth)

    stats$max_width          <- try_stat(phylo, treestats::max_width, normalize, c("tips", "none"))
    stats$mw_over_md         <- try_stat(phylo, treestats::mw_over_md)

    stats$tot_path           <- try_stat(phylo, treestats::tot_path_length)
    stats$tot_internal_path  <- try_stat(phylo, treestats::tot_internal_path)

    stats$rogers             <- try_stat(phylo, treestats::rogers, normalize, c("tips", "none"))

    stats$stairs2            <- try_stat(phylo, treestats::stairs2)

    stats$tot_coph           <- try_stat(phylo, treestats::tot_coph, normalize, c("yule", "none"))

    stats$symmetry_nodes     <- try_stat(phylo, treestats::sym_nodes, normalize, c("tips", "none"))

    stats$mpd                <- try_stat(phylo, treestats::mean_pair_dist, normalize, c("tips", "none"))
    stats$psv                <- try_stat(phylo, treestats::psv, normalize, c("tips", "none"))
    stats$vpd                <- try_stat(phylo, treestats::var_pair_dist)
    stats$mntd               <- try_stat(phylo, treestats::mntd)

    stats$j_stat             <- try_stat(phylo, treestats::entropy_j)

    stats$rquartet           <- try_stat(phylo, treestats::rquartet, normalize, c("yule", "none"))
    stats$wiener             <- try_stat(phylo, treestats::wiener, normalize)

    stats$max_betweenness    <- try_stat(phylo, treestats::max_betweenness, normalize, c("tips", "none"))

    local_closeness <- function(tree, w, n) {
      treestats::max_closeness(tree, weight = w,
                               normalization = ifelse(n == TRUE, "tips", "none"))
    }

    stats$max_closeness      <- try_stat(phylo, function(x) local_closeness(x, FALSE, normalize))
    stats$max_closenessW     <- try_stat(phylo, function(x) local_closeness(x, TRUE,  normalize))

    stats$diameter           <- try_stat(phylo, treestats::diameter)

    stats$eigen_centrality   <- try_stat(phylo, function(x) max(treestats::eigen_centrality(x, weight = FALSE)$eigenvector))
    stats$eigen_centralityW  <- try_stat(phylo, function(x) max(treestats::eigen_centrality(x, weight = TRUE)$eigenvector))

    stats$mean_branch_length <- try_stat(phylo, treestats::mean_branch_length)
    stats$var_branch_length  <- try_stat(phylo, treestats::var_branch_length)

    stats$mean_branch_length_int <- try_stat(phylo, treestats::mean_branch_length_int)
    stats$mean_branch_length_ext <- try_stat(phylo, treestats::mean_branch_length_ext)
    stats$var_branch_length_int  <- try_stat(phylo, treestats::var_branch_length_int)
    stats$var_branch_length_ext  <- try_stat(phylo, treestats::var_branch_length_ext)

    stats$treeness <- try_stat(phylo, treestats::treeness)
    stats$root_imbalance <- try_stat(phylo, treestats::root_imbalance)

    lap_stats <- tryCatch(
      stats_laplace(phylo, weighted = TRUE),
      error = function(e) data.frame(
        lap_max = NA, lap_fiedler = NA, lap_var = NA, lap_skew = NA, lap_mean = NA,
        lap_kurt = NA, lap_second_min = NA, lap_entropy = NA, lap_sum = NA,
        cut_weight = NA, conductance = NA, normalized_cut = NA, bottleneck_strength = NA,
        density_pos = NA, density_neg = NA, modularity = NA, fiedler_sd = NA
      )
    )

    stats <- c(stats, as.list(lap_stats))

    stats <- lapply(stats, function(x) {
      if (length(x) == 0) return(NA_real_)
      if (length(x) > 1) return(mean(x, na.rm = TRUE))
      as.numeric(x)
    })

    stats <- unlist(stats)
    stats <- stats[order(names(stats))]
    return(stats)

  } else {
    stats <- list()

    stats$tree_height_uw <- try_stat(phylo, treestats::tree_height)
    stats$num_branches   <- try_stat(phylo, treestats::phylogenetic_diversity)

    temp_stats <- tryCatch(expr = { treestats::laplacian_spectrum(phylo) },
                           error = function(e) { return(NA) })
    if (length(temp_stats) == 5) {
      stats$laplace_spectrum_a_uw <- temp_stats$asymmetry
      stats$laplace_spectrum_p_uw <- temp_stats$peakedness
      stats$laplace_spectrum_e_uw <- log(temp_stats$principal_eigenvalue)
      stats$laplace_spectrum_g_uw <- temp_stats$eigengap[[1]]
    } else {
      stats$laplace_spectrum_a_uw <- NA
      stats$laplace_spectrum_p_uw <- NA
      stats$laplace_spectrum_e_uw <- NA
      stats$laplace_spectrum_g_uw <- NA
    }

    temp_stats <- try_stat(phylo, function(x) treestats::minmax_laplace(x, TRUE))
    if (length(temp_stats) >= 2) {
      stats$min_laplace_uw <- temp_stats$min
      stats$max_laplace_uw <- temp_stats$max
    } else {
      stats$min_laplace_uw <- NA
      stats$max_laplace_uw <- NA
    }

    temp_stats <- try_stat(phylo, function(x) treestats::minmax_adj(x, TRUE))
    if (length(temp_stats) >= 2) {
      stats$min_adj_uw <- temp_stats$min
      stats$max_adj_uw <- temp_stats$max
    } else {
      stats$min_adj_uw <- NA
      stats$max_adj_uw <- NA
    }

    stats$mpd_uw <- try_stat(phylo, treestats::mean_pair_dist, normalize, c("tips", "none"))
    stats$psv_uw <- try_stat(phylo, treestats::psv,            normalize, c("tips", "none"))
    stats$vpd_uw <- try_stat(phylo, treestats::var_pair_dist)
    stats$mntd_uw <- try_stat(phylo, treestats::mntd)
    stats$j_stat_uw <- try_stat(phylo, treestats::entropy_j)
    stats$wiener_uw <- try_stat(phylo, treestats::wiener, normalize)

    local_closeness <- function(tree, w, n) {
      treestats::max_closeness(tree, weight = w,
                               normalization = ifelse(n == TRUE, "tips", "none"))
    }

    stats$max_closenessW_uw <- try_stat(phylo, function(x) local_closeness(x, TRUE, normalize))
    stats$eigen_centralityW_uw <- try_stat(phylo, function(x) max(treestats::eigen_centrality(x, weight = TRUE)$eigenvector))
    stats$treeness_uw <- try_stat(phylo, treestats::treeness)

    stats <- unlist(stats)
    stats <- stats[order(names(stats))]
    return(stats)
  }
}

#' @keywords internal
try_stat <- function(phylo,
                     func,
                     normalize = FALSE,
                     norm_res = c(TRUE, FALSE)) {
  if (normalize) {
    return(tryCatch(
      expr = func(phylo, normalization = ifelse(normalize, norm_res[1], norm_res[2])),
      error = function(e) NA
    ))
  } else {
    return(tryCatch(expr = func(phylo), error = function(e) NA))
  }
}

# --- Argument parsing ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_treestats_clean.R <input_file> <output_file> <combo_yaml> [copy_tree_file] [copy_clean_file]")
}

input_file  <- args[1]
output_file <- args[2]
combo_yaml  <- args[3]
copy_tree_file  <- if (length(args) >= 4) args[4] else NA_character_
copy_clean_file <- if (length(args) >= 5) args[5] else NA_character_

if (file.exists(output_file)) {
  cat(sprintf("Output file already exists, skipping: %s\n", output_file))
  quit(status = 0)
}

# --- Checks ---
if (!file.exists(input_file)) stop("Input file does not exist: ", input_file)
if (!file.exists(combo_yaml)) stop("YAML config file does not exist: ", combo_yaml)

# --- Ensure output directory exists ---
output_dir <- dirname(output_file)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# FAST PATH: patch pigot_rho in an existing feature CSV
# ------------------------------------------------------------

if (!is.na(copy_tree_file) && !is.na(copy_clean_file) &&
    file.exists(copy_tree_file) && file.exists(copy_clean_file)) {

  message("🔁 Patch mode: replacing pigot_rho in existing CSV")
  message("  copy_tree_file : ", copy_tree_file)
  message("  copy_clean_file: ", copy_clean_file)

  # 1) Load existing CSV
  df_feat <- read.csv(copy_tree_file, stringsAsFactors = FALSE, check.names = FALSE)

  if (!("pigot_rho" %in% colnames(df_feat))) {
    stop("Column 'pigot_rho' not found in copy_tree_file: ", copy_tree_file)
  }

  # drop pigot_rho_uw if present
  if ("pigot_rho_uw" %in% colnames(df_feat)) {
    df_feat$pigot_rho_uw <- NULL
    message("🗑️ Removed column: pigot_rho_uw")
  }

  # 2) Recompute pigot_rho per tree from raw tree file in the SAME order
  is_tree_line <- function(line) {
    grepl("\\(", line) && grepl(";+", line) &&
      !grepl("^\\s*//", line) &&
      !grepl("^\\s*segsites", line, ignore.case = TRUE) &&
      !grepl("^\\s*positions", line, ignore.case = TRUE)
  }

  clean_tree_line <- function(line) trimws(sub(";+$", ";", line))

  # pre-allocate result vector with target length
  new_rho <- rep(NA_real_, nrow(df_feat))
  i <- 0L
  skipped <- 0L

  con2 <- file(copy_clean_file, open = "r")
  on.exit(close(con2), add = TRUE)

  while (length(line <- readLines(con2, n = 1, warn = FALSE)) > 0) {
    if (!is_tree_line(line)) next
    line <- clean_tree_line(line)

    tr <- tryCatch(ape::read.tree(text = line), error = function(e) NULL)
    if (is.null(tr)) {
      skipped <- skipped + 1L
      next
    }

    i <- i + 1L
    if (i > nrow(df_feat)) {
      stop("More tree lines found in copy_clean_file than rows in copy_tree_file. Inputs do not match.")
    }

    new_rho[i] <- tryCatch(as.numeric(calc_pigot_rho(tr)), error = function(e) NA_real_)
  }

  if (i != nrow(df_feat)) {
    stop(
      "Row mismatch: computed pigot_rho for ", i, " trees, but CSV has ", nrow(df_feat), " rows.\n",
      "This usually means copy_clean_file is not the same tree stream used to create copy_tree_file,\n",
      "or the tree-line filter differs."
    )
  }

  # 3) Replace and write output
  df_feat$pigot_rho <- new_rho
  write.csv(df_feat, file = output_file, row.names = FALSE)
  message("✅ Patched pigot_rho written to: ", output_file, " (", skipped, " trees skipped)")

  quit(status = 0)
}

# ------------------------------------------------------------
# Normal mode: compute everything from scratch
# ------------------------------------------------------------
combo_params <- yaml::read_yaml(combo_yaml)
required <- c("sequence_length")
missing <- setdiff(required, names(combo_params))
if (length(missing) > 0) stop("Missing fields in ", combo_yaml, ": ", paste(missing, collapse = ", "))

seq_length <- combo_params$sequence_length
has_recomb <- "recombination_rate" %in% names(combo_params) &&
  !is.null(combo_params$recombination_rate) &&
  combo_params$recombination_rate != 0

is_tree_line <- function(line) {
  grepl("\\(", line) && grepl(";+", line) &&
    !grepl("^\\s*//", line) &&
    !grepl("^\\s*segsites", line, ignore.case = TRUE) &&
    !grepl("^\\s*positions", line, ignore.case = TRUE)
}

clean_tree_line <- function(line) trimws(sub(";+$", ";", line))

skipped <- 0L
header_written <- FALSE

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

  tree_stats <- calculate_all_treestats(tr, normalize = FALSE, weighted = TRUE)
  tree_stats_unweighted <- calculate_all_treestats(tr_unweighted, normalize = FALSE, weighted = FALSE)
  if (is.null(tree_stats) || is.null(tree_stats_unweighted)) {
    skipped <- skipped + 1L
    next
  }

  combined <- as.data.frame(as.list(c(tree_stats, tree_stats_unweighted)))

  # ensure pigot_rho_uw never ends up in outputs
  if ("pigot_rho_uw" %in% names(combined)) combined$pigot_rho_uw <- NULL

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
