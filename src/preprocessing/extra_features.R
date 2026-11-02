calc_pigot_rho <- function(tr) {
  out <- NA_real_

  if (is.null(tr$edge) || nrow(tr$edge) == 0) return(out)

  crown_age <- tryCatch(treestats::crown_age(tr), error = function(e) NA)
  if (is.na(crown_age) || length(crown_age) != 1) return(out)

  mid_point <- crown_age / 2
  n1 <- 2

  root_node <- tr$edge[1, 1]
  tip_nodes <- seq_along(tr$tip.label)

  n3 <- 0
  for (i in seq_len(nrow(tr$edge))) {
    child <- tr$edge[i, 2]
    if (child < root_node) n3 <- n3 + 1
  }

  brts <- sort(branching.times(tr), decreasing = TRUE)
  n2 <- 1 + sum(brts >= mid_point)

  if (n2 <= 0 || n3 <= 0) return(out)

  r1 <- (log(n2) - log(n1)) / mid_point
  r2 <- (log(n3) - log(n2)) / mid_point
  rho <- (r2 - r1) / (r1 + r2)
  return(rho)
}

calc_phylogenetic_diversity <- function(tree) {
  sum(tree$edge.length, na.rm = TRUE)
}

calc_blum_index <- function(tree) {
  score <- 0

  count_tips <- function(node) {
    # Terminal node = Tip
    children <- which(tree$edge[, 1] == node)

    if (length(children) == 0) {
      return(1)
    }

    # Anzahl der Tips unterhalb dieses Knotens
    n <- sum(vapply(
      tree$edge[children, 2],
      count_tips,
      numeric(1)
    ))

    if (n > 1) {
      score <<- score + log(n - 1)
    }

    return(n)
  }

  count_tips(tree$edge[1, 1])

  return(score)
}