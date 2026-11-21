source("src/preprocessing/extra_features.R")

library(ape)
library(treestats)
library(igraph)
library(yaml)


eps = 1e-8

fix_zero_edges <- function(tr) {
    if (!is.null(tr$edge.length)) {
      tr$edge.length[tr$edge.length == 0] <- eps
    }
    tr
  }

make_binary <- function(tr) {
    trb <- ape::multi2di(tr, random = TRUE)
    fix_zero_edges(trb)
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

calculate_all_treestats <- function(phylo, normalize = FALSE, n_resolutions = 30) {

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

  is_bin <- safe_any(ape::is.binary(phylo))
  if (is.null(is_bin) || length(is_bin) != 1 || is.na(is_bin)) {
    is_bin <- FALSE
  }

  need_binary_avg <- (!is_bin) && (n_resolutions > 1)

  stats_original <- list()

  nb <- calc_non_binary_node_features(phylo)
  stats_original$non_binary_nodes <- nb$non_binary_nodes
  stats_original$non_binary_nodes_frac <- nb$non_binary_nodes_frac
  
  stats_original$pigot_rho <- tryCatch(
    as.numeric(calc_pigot_rho(phylo)),
    error = function(e) NA_real_
  )

  stats_original$phylogenetic_div <- tryCatch(
    as.numeric(calc_phylogenetic_diversity(phylo)),
    error = function(e) NA_real_
  )

  
  stats_original$crown_age          <- try_stat(phylo, treestats::crown_age)
  stats_original$tree_height        <- try_stat(phylo, treestats::tree_height)

  stats_original$number_of_lineages <- try_stat(phylo, treestats::number_of_lineages)

  stats_original$il_number          <- try_stat(phylo, treestats::ILnumber, normalize, c("tips", "none"))

  

  temp_stats <- tryCatch(expr = { treestats::laplacian_spectrum(phylo) },
                          error = function(e) { return(NA) })

  if (length(temp_stats) == 5) {
    stats_original$laplace_spectrum_a  <- temp_stats$asymmetry
    stats_original$laplace_spectrum_p  <- temp_stats$peakedness
    stats_original$laplace_spectrum_e  <- log(temp_stats$principal_eigenvalue)
    stats_original$laplace_spectrum_g  <- temp_stats$eigengap[[1]]
  } else {
    stats_original$laplace_spectrum_a  <- NA
    stats_original$laplace_spectrum_p  <- NA
    stats_original$laplace_spectrum_e  <- NA
    stats_original$laplace_spectrum_g  <- NA
  }


  stats_original$mpd                <- try_stat(phylo, treestats::mean_pair_dist, normalize, c("tips", "none"))
  stats_original$psv                <- try_stat(phylo, treestats::psv, normalize, c("tips", "none"))
  stats_original$mntd               <- try_stat(phylo, treestats::mntd)

  stats_original$j_stat             <- try_stat(phylo, treestats::entropy_j)

  stats_original$rquartet           <- try_stat(phylo, treestats::rquartet, normalize, c("yule", "none"))
  stats_original$mean_branch_length <- try_stat(phylo, treestats::mean_branch_length)
  stats_original$var_branch_length  <- try_stat(phylo, treestats::var_branch_length)

  stats_original$mean_branch_length_int <- try_stat(phylo, treestats::mean_branch_length_int)
  stats_original$mean_branch_length_ext <- try_stat(phylo, treestats::mean_branch_length_ext)
  stats_original$var_branch_length_int  <- try_stat(phylo, treestats::var_branch_length_int)
  stats_original$var_branch_length_ext  <- try_stat(phylo, treestats::var_branch_length_ext)

  stats_original$treeness <- try_stat(phylo, treestats::treeness)

  phylo_unweighted <- phylo
  phylo_unweighted$edge.length <- rep(1, nrow(phylo_unweighted$edge))

  stats_original$tree_depth <- max(node.depth(phylo_unweighted))
  stats_original$num_branches <- nrow(phylo$edge)

  calculate_binary_stats <- function(tr_bin) {
    stats_bin <- list()

    stats_bin$area_per_pair      <- try_stat(tr_bin, treestats::area_per_pair, normalize, c("yule", "none"))
    stats_bin$average_leaf_depth <- try_stat(tr_bin, treestats::average_leaf_depth, normalize, c("yule", "none"))

    stats_bin$blum <- tryCatch(
      as.numeric(calc_blum_index(tr_bin)),
      error = function(e) NA_real_
    )

    stats_bin$colless            <- try_stat(tr_bin, treestats::colless, normalize, c("yule", "none"))
    stats_bin$colless_corr       <- try_stat(tr_bin, treestats::colless_corr, normalize, c("yule", "none"))
    stats_bin$colless_quad       <- try_stat(tr_bin, treestats::colless_quad, normalize, c("yule", "none"))

    stats_bin$beta               <- try_stat(tr_bin, treestats::beta_statistic)

    stats_bin$avg_ladder         <- try_stat(tr_bin, treestats::avg_ladder)
    stats_bin$max_ladder         <- try_stat(tr_bin, treestats::max_ladder)
    stats_bin$cherries           <- try_stat(tr_bin, treestats::cherries, normalize, c("yule", "none"))

    stats_bin$double_cherries    <- try_stat(tr_bin, treestats::double_cherries)
    stats_bin$four_prong         <- try_stat(tr_bin, treestats::four_prong)
    stats_bin$stairs             <- try_stat(tr_bin, treestats::stairs)
    stats_bin$stairs2            <- try_stat(tr_bin, treestats::stairs2)

    temp_stats <- try_stat(tr_bin, function(x) treestats::minmax_laplace(x, TRUE))
    if (length(temp_stats) >= 2) {
      stats_bin$min_laplace <- temp_stats$min
      stats_bin$max_laplace <- temp_stats$max
    } else {
      stats_bin$min_laplace <- NA
      stats_bin$max_laplace <- NA
    }

    temp_stats <- try_stat(tr_bin, function(x) treestats::minmax_adj(x, TRUE))
    if (length(temp_stats) >= 2) {
      stats_bin$min_adj <- temp_stats$min
      stats_bin$max_adj <- temp_stats$max
    } else {
      stats_bin$min_adj <- NA
      stats_bin$max_adj <- NA
    }

    stats_bin$j_one              <- try_stat(tr_bin, treestats::j_one)
    stats_bin$b1                 <- try_stat(tr_bin, treestats::b1, normalize, c("tips", "none"))
    stats_bin$b2                 <- try_stat(tr_bin, treestats::b2, normalize, c("yule", "none"))
    stats_bin$eigen_centrality   <- try_stat(tr_bin, function(x) max(treestats::eigen_centrality(x, weight = FALSE)$eigenvector))
    stats_bin$eigen_centralityW  <- try_stat(tr_bin, function(x) max(treestats::eigen_centrality(x, weight = TRUE)$eigenvector))
    stats_bin$root_imbalance <- try_stat(tr_bin, treestats::root_imbalance)

    local_closeness <- function(tree, w, n) {
      treestats::max_closeness(tree, weight = w,
                                normalization = ifelse(n == TRUE, "tips", "none"))
    }

    stats_bin$max_closeness      <- try_stat(tr_bin, function(x) local_closeness(x, FALSE, normalize))
    stats_bin$max_closenessW     <- try_stat(tr_bin, function(x) local_closeness(x, TRUE,  normalize))
    stats_bin$wiener             <- try_stat(tr_bin, treestats::wiener, normalize)

    stats_bin$max_betweenness    <- try_stat(tr_bin, treestats::max_betweenness, normalize, c("tips", "none"))
    stats_bin$diameter           <- try_stat(tr_bin, treestats::diameter)
    stats_bin$i_stat             <- try_stat(tr_bin, treestats::mean_i)

    stats_bin$ew_colless         <- try_stat(tr_bin, treestats::ew_colless)

    stats_bin$max_del_width      <- try_stat(tr_bin, treestats::max_del_width, normalize, c("tips", "none"))

    stats_bin$max_depth          <- try_stat(tr_bin, treestats::max_depth, normalize, c("tips", "none"))
    stats_bin$avg_vert_depth     <- try_stat(tr_bin, treestats::avg_vert_depth)

    stats_bin$max_width          <- try_stat(tr_bin, treestats::max_width, normalize, c("tips", "none"))
    stats_bin$mw_over_md         <- try_stat(tr_bin, treestats::mw_over_md)
    stats_bin$tot_path           <- try_stat(tr_bin, treestats::tot_path_length)
    stats_bin$tot_internal_path  <- try_stat(tr_bin, treestats::tot_internal_path)

    stats_bin$rogers             <- try_stat(tr_bin, treestats::rogers, normalize, c("tips", "none"))

    stats_bin$tot_coph           <- try_stat(tr_bin, treestats::tot_coph, normalize, c("yule", "none"))

    stats_bin$symmetry_nodes     <- try_stat(tr_bin, treestats::sym_nodes, normalize, c("tips", "none"))
    
    stats_bin$vpd                <- try_stat(tr_bin, treestats::var_pair_dist)

    stats_bin <- lapply(stats_bin, as_num_scalar)
    stats_bin <- unlist(stats_bin)
    stats_bin[order(names(stats_bin))]
  }

  if (!need_binary_avg) {

    stats_binary <- calculate_binary_stats(phylo)

  } else {

    reps <- vector("list", n_resolutions)

    for (i in seq_len(n_resolutions)) {

      tr_bin <- make_binary(phylo)

      reps[[i]] <- calculate_binary_stats(tr_bin)
    }

    # Ergebnisse der einzelnen Resolutions zusammenführen
    all_names <- sort(unique(unlist(lapply(reps, names))))

    mat <- do.call(rbind, lapply(reps, function(v) {
      out <- rep(NA_real_, length(all_names))
      names(out) <- all_names
      out[names(v)] <- v
      out
    }))

    # Mittelwert pro Feature über alle Resolutions
    stats_binary <- apply(
      mat, 2,
      function(v) mean(v, na.rm = TRUE)
    )

    stats_binary <- stats_binary[order(names(stats_binary))]
  }

  stats <- c(stats_original, as.list(stats_binary))

  stats <- lapply(stats, as_num_scalar)
  stats <- unlist(stats)
  stats <- stats[order(names(stats))]

  return(stats)
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
  stop("Usage: Rscript run_treestats_clean.R <input_file> <output_file> <combo_yaml>")
}

input_file  <- args[1]
output_file <- args[2]
combo_yaml  <- args[3]

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
blocked <- function(p) grepl("zigzag|one_population", p) 

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

  tree_stats <- calculate_all_treestats(tr, normalize = FALSE)

  if (is.null(tree_stats)) {
    skipped <- skipped + 1L
    next
  }

  df <- as.data.frame(as.list(tree_stats), stringsAsFactors = FALSE)

  if (has_recomb) {
    segment_length <- as.numeric(gsub("^\\[(\\d+)\\].*$", "\\1", line))
    df$segment_length <- segment_length / seq_length
  }

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
