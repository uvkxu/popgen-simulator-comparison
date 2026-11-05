source("src/preprocessing/extra_features.R")

library(ape)
library(treestats)
library(igraph)
library(yaml)

calculate_all_treestats <- function(phylo, normalize = FALSE) {

  stats <- list()
  
  stats$pigot_rho <- tryCatch(
    as.numeric(calc_pigot_rho(phylo)),
    error = function(e) NA_real_
  )

  stats$blum <- tryCatch(
    as.numeric(calc_blum_index(phylo)),
    error = function(e) NA_real_
  )

  stats$phylogenetic_div <- tryCatch(
    as.numeric(calc_phylogenetic_diversity(phylo)),
    error = function(e) NA_real_
  )

  stats$sackin             <- try_stat(phylo, treestats::sackin, normalize, c("yule", "none"))

  stats$colless            <- try_stat(phylo, treestats::colless, normalize, c("yule", "none"))
  stats$colless_corr       <- try_stat(phylo, treestats::colless_corr, normalize, c("yule", "none"))
  stats$colless_quad       <- try_stat(phylo, treestats::colless_quad, normalize, c("yule", "none"))

  stats$beta               <- try_stat(phylo, treestats::beta_statistic)
  stats$crown_age          <- try_stat(phylo, treestats::crown_age)
  stats$tree_height        <- try_stat(phylo, treestats::tree_height)

  stats$number_of_lineages <- try_stat(phylo, treestats::number_of_lineages)
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

  phylo_unweighted <- phylo
  phylo_unweighted$edge.length <- rep(1, nrow(phylo_unweighted$edge))

  stats$tree_depth <- max(node.depth(phylo_unweighted))
  stats$num_branches <- nrow(phylo$edge)

  stats <- lapply(stats, function(x) {
    if (length(x) == 0) return(NA_real_)
    if (length(x) > 1) return(mean(x, na.rm = TRUE))
    as.numeric(x)
  })

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
