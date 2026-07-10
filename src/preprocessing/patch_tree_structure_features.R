library(ape)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(
    "Usage:\n",
    "Rscript patch_tree_structure_features.R <tree_features.csv> <cleaned_output.txt> <output.csv>"
  )
}

feature_file <- args[1]
tree_file <- args[2]
output_file <- args[3]

# ------------------------------------------------------------
# Read existing feature CSV
# ------------------------------------------------------------

df <- read.csv(
  feature_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if (!("num_branches" %in% names(df))) {
  stop("Column 'num_branches' not found in feature CSV.")
}

if (!("tree_height_uw" %in% names(df))) {
  stop("Column 'tree_height_uw' not found in feature CSV.")
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Recompute features
# ------------------------------------------------------------

new_num_branches <- numeric(nrow(df))
new_tree_height <- numeric(nrow(df))

tree_index <- 0L

con <- file(tree_file, open = "r")
on.exit(close(con))

while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {

  if (!is_tree_line(line))
    next

  line <- clean_tree_line(line)

  tr <- tryCatch(
    ape::read.tree(text = line),
    error = function(e) NULL
  )

  if (is.null(tr))
    next

  tree_index <- tree_index + 1L

  if (tree_index > nrow(df)) {
    stop(
      paste0(
        "More trees found than rows in feature CSV.\n",
        "CSV rows: ", nrow(df), "\n",
        "Trees: ", tree_index
      )
    )
  }

  tr_unweighted <- tr
  tr_unweighted$edge.length <- rep(
    1,
    nrow(tr_unweighted$edge)
  )

  new_num_branches[tree_index] <-
    nrow(tr$edge)

  new_tree_height[tree_index] <-
    max(node.depth(tr_unweighted))
}

if (tree_index != nrow(df)) {
  stop(
    paste0(
      "Tree count mismatch.\n",
      "CSV rows: ", nrow(df), "\n",
      "Trees read: ", tree_index
    )
  )
}

# ------------------------------------------------------------
# Replace columns
# ------------------------------------------------------------

df$num_branches <- new_num_branches
df$tree_height_uw <- new_tree_height

# ------------------------------------------------------------
# Write output
# ------------------------------------------------------------

output_dir <- dirname(output_file)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

write.csv(
  df,
  output_file,
  row.names = FALSE
)

cat(
  sprintf(
    "Successfully updated %d trees.\nOutput written to:\n%s\n",
    tree_index,
    output_file
  )
)