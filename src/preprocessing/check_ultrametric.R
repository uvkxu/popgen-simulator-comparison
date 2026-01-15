library(ape)
library(treestats)
library(yaml)

# ============================================================
# C++-NAHE HILFSFUNKTIONEN
# ============================================================

get_start_date <- function(branchset, parent_label) {
  for (b in branchset) {
    if (b$own_label == parent_label) {
      return(b$end_date)
    }
  }
  stop("Parent label not found in branchset")
}

remove_from_branchset <- function(branchset, tip_label) {
  to_remove <- tip_label
  repeat {
    parents <- vapply(branchset, function(b) b$parent_label, numeric(1))
    own     <- vapply(branchset, function(b) b$own_label, numeric(1))
    idx <- which(own %in% to_remove)
    if (length(idx) == 0) break
    to_remove <- parents[idx]
    branchset <- branchset[-idx]
  }
  branchset
}

create_branch_set_cpp_like <- function(tr,
                                       max_t = 1e12,
                                       crown_age,
                                       extinct_acc = 1e-6) {

  edge <- tr$edge
  el   <- tr$edge.length
  crown <- edge[1, 1]

  branchset <- list()
  tip_times <- list()

  for (i in seq_len(nrow(edge))) {

    parent_label <- edge[i, 1]
    start_date <- if (parent_label == crown) 0 else
      get_start_date(branchset, parent_label)

    if (start_date > max_t) next

    own_label <- edge[i, 2]
    bl <- el[i]
    end_date <- start_date + bl

    if (end_date > max_t) {
      end_date <- max_t
      bl <- end_date - start_date
    }

    if (own_label < crown && end_date < crown_age && end_date < max_t) {
      tip_times[[length(tip_times) + 1]] <- c(own_label, end_date)
    }

    branchset[[length(branchset) + 1]] <- list(
      start_date   = start_date,
      parent_label = parent_label,
      own_label    = own_label,
      end_date     = end_date,
      bl           = bl
    )
  }

  if (length(tip_times) > 0) {
    limit <- min(crown_age, max_t)
    for (tt in tip_times) {
      if (abs(limit - tt[2]) > extinct_acc) {
        branchset <- remove_from_branchset(branchset, tt[1])
      }
    }
  }

  branchset
}

num_lineages_cpp_like <- function(branchset, t) {
  sum(vapply(branchset, function(b)
    b$start_date < t && b$end_date >= t,
    logical(1)))
}

# ============================================================
# PIGOT RHO DIAGNOSE (C++-NAH)
# ============================================================

calc_pigot_rho_diagnose <- function(tr) {

  edge <- tr$edge
  el   <- tr$edge.length
  crown_age <- treestats::crown_age(tr)
  mid_point <- crown_age * 0.5
  crown_node <- edge[1, 1]

  # exakt wie C++
  n3 <- sum(edge[, 2] < crown_node)

  branchset <- create_branch_set_cpp_like(tr, crown_age = crown_age)
  n2 <- num_lineages_cpp_like(branchset, mid_point)

  n1 <- 2L
  r1 <- (log(n2) - log(n1)) / mid_point
  r2 <- (log(n3) - log(n2)) / mid_point
  rho <- (r2 - r1) / (r1 + r2)

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

  new_row$crown_age <- if (is.null(existing_row) || !"crown_age" %in% names(existing_df)) {
    diag$crown_age
  } else existing_row$crown_age

  # --- Branching times ---
  brts <- sort(branching.times(tr), decreasing = TRUE)
  new_row$n3 <- if (is.null(existing_row) || !"n3" %in% names(existing_df)) 1 + length(brts) else existing_row$n3
  mid_point <- brts[1] / 2
  new_row$n2 <- if (is.null(existing_row) || !"n2" %in% names(existing_df)) 1 + sum(brts >= mid_point) else existing_row$n2

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