library(ape)
library(yaml)
library(treestats)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    "Usage:\n",
    "Rscript check_tree_structure.R <cleaned_file> <combo_yaml> <output_csv>"
  )
}

input_file <- args[1]
combo_yaml <- args[2]
output_file <- args[3]

combo <- yaml::read_yaml(combo_yaml)

expected_leaves <- sum(combo$sample_structure)

is_tree_line <- function(line) {
  grepl("\\(", line) &&
    grepl(";+", line) &&
    !grepl("^\\s*//", line) &&
    !grepl("^\\s*segsites", line, ignore.case = TRUE) &&
    !grepl("^\\s*positions", line, ignore.case = TRUE)
}

clean_tree_line <- function(line) {
  trimws(sub(";+$", ";", line))
}

results <- list()
tree_id <- 1

con <- file(input_file, open = "r")

while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {

  if (!is_tree_line(line)) {
    next
  }

  line <- clean_tree_line(line)

  tr <- tryCatch(
    ape::read.tree(text = line),
    error = function(e) NULL
  )

  if (is.null(tr)) {
    next
  }

  tr_unweighted <- tr
  tr_unweighted$edge.length <- rep(1, nrow(tr_unweighted$edge))

  n_leaves <- length(tr$tip.label)

  n_nodes <- length(unique(as.vector(tr$edge)))

  n_branches <- nrow(tr$edge)

  n_branches_pd <- treestats::phylogenetic_diversity(
    tr_unweighted
  )

  expected_branches <- 2 * expected_leaves - 2

  branch_difference <- n_branches - expected_branches

  child_counts <- table(tr$edge[,1])

  n_unary_nodes <- sum(child_counts == 1)

  n_multifurcations <- sum(child_counts > 2)

  tree_height_uw <- treestats::tree_height(
    tr_unweighted
  )
  max_depth_ape <- max(
    node.depth(tr_unweighted)
  )

  branch_pd_difference <-
    n_branches_pd - n_branches

  height_difference <-
    tree_height_uw - max_depth_ape

  lens <- c(
    tree_id = length(tree_id),
    expected_leaves = length(expected_leaves),
    observed_leaves = length(n_leaves),
    leaves_ok = length(n_leaves == expected_leaves),
    expected_branches = length(expected_branches),
    observed_branches = length(n_branches),
    observed_branches_pd = length(n_branches_pd),
    branch_difference = length(branch_difference),
    n_unary_nodes = length(n_unary_nodes),
    n_multifurcations = length(n_multifurcations),
    unary_matches_difference =
      length(branch_difference == n_unary_nodes),
    tree_height_uw = length(tree_height_uw),
    max_depth_ape = length(max_depth_ape),
    branch_pd_difference = length(branch_pd_difference),
    height_difference = length(height_difference)
  )

  if (any(lens != 1)) {
    stop(
      paste0(
        "Invalid lengths for tree_id=", tree_id, "\n",
        paste(names(lens), lens, sep = "=", collapse = "\n")
      )
    )
  }

  results[[tree_id]] <- data.frame(
    tree_id = tree_id,
    expected_leaves = expected_leaves,
    observed_leaves = n_leaves,
    leaves_ok = n_leaves == expected_leaves,
    expected_branches = expected_branches,
    observed_branches = n_branches,
    observed_branches_pd = n_branches_pd,
    branch_difference = branch_difference,
    n_unary_nodes = n_unary_nodes,
    n_multifurcations = n_multifurcations,
    unary_matches_difference =
      branch_difference == n_unary_nodes,
    tree_height_uw = tree_height_uw,
    max_depth_ape = max_depth_ape,
    branch_pd_difference = branch_pd_difference,
    height_difference = height_difference
  )

  tree_id <- tree_id + 1
}

close(con)

df <- do.call(rbind, results)

output_dir <- dirname(output_file)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(
  df,
  output_file,
  row.names = FALSE
)

cat("Trees analysed:", nrow(df), "\n")
cat(
  "Wrong leaf count:",
  sum(!df$leaves_ok),
  "\n"
)
cat(
  "Unary mismatch:",
  sum(!df$unary_matches_difference),
  "\n"
)
cat(
  "Trees with polytomies:",
  sum(df$n_multifurcations > 0),
  "\n"
)