library(ape)
library(yaml)

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

  n_leaves <- length(tr$tip.label)

  n_nodes <- length(unique(as.vector(tr$edge)))

  n_branches <- nrow(tr$edge)

  expected_branches <- 2 * expected_leaves - 2

  branch_difference <- n_branches - expected_branches

  child_counts <- table(tr$edge[,1])

  n_unary_nodes <- sum(child_counts == 1)

  n_multifurcations <- sum(child_counts > 2)

  results[[tree_id]] <- data.frame(
    tree_id = tree_id,
    expected_leaves = expected_leaves,
    observed_leaves = n_leaves,
    leaves_ok = n_leaves == expected_leaves,
    expected_branches = expected_branches,
    observed_branches = n_branches,
    branch_difference = branch_difference,
    n_unary_nodes = n_unary_nodes,
    n_multifurcations = n_multifurcations,
    unary_matches_difference =
      branch_difference == n_unary_nodes
  )

  tree_id <- tree_id + 1
}

close(con)

df <- do.call(rbind, results)

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