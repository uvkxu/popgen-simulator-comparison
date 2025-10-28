#' Laplacian Matrix properties
#' @description Calculates the eigenvalues of the Laplacian Matrix, where the
#' Laplacian matrix is the matrix representation of a graph, in this case a
#' phylogeny.
#' When the R package \pkg{RSpectra} is available, a faster
#' calculation can be used, which does not calculate all eigenvalues, but only
#' the maximum and minimum. As such, when using this option, the vector of all
#' eigenvalues is not returned.
#' @param tr phylo object or ltable
#' @param use_rspectra boolean to indicate whether the helping package RSpectra
#' should be used, in which case only the minimum and maximum values are
#' returned
#' @return List with the minimum and maximum eigenvalues
#' @references  Chindelevitch, Leonid, et al. "Network science inspires novel
#' tree shape statistics." Plos one 16.12 (2021): e0259877.
#' @export
stats_laplace <- function(tr, weighted=TRUE) {
  treestats:::check_tree(tr,
             require_binary = TRUE,
             require_ultrametric = FALSE,
             require_rooted = FALSE)

  if (inherits(tr, "phylo")) {
    g <- ape::as.igraph.phylo(tr, directed = FALSE)
    if (!is.null(tr$edge.length)) {
      E(g)$weight <- tr$edge.length
    } else {
      E(g)$weight <- 1
    }

    mat_size <- max(tr$edge)

    # prepping matrix in Rcpp yields no speed gain
    mat_size <- max(tr$edge)
    if (mat_size > 46340) {  # floor(sqrt(2^31 - 1)))
      stop("tree too big, memory allocation fail")
    }

    lap_mat <- matrix(0, mat_size, mat_size)
    lap_mat[tr$edge] <- ifelse(rep(TRUE, mat_size - 1), # weight is always true
                                tr$edge.length, 1)

    lap_mat <- lap_mat + t(lap_mat)
    degrees <- rowSums(lap_mat)
    lap_mat <- diag(degrees) - lap_mat

  
    eig <- eigen(lap_mat, symmetric = TRUE)
    eigen_vals <- eig$values
    eigen_vals <- round(eigen_vals, digits = 10)

    fiedler_val <- min(eigen_vals[eigen_vals > 0])

    max_val <- max(eigen_vals)
    mean_val <- mean(eigen_vals)
    var_val <- var(eigen_vals)
    skew_val <- e1071::skewness(eigen_vals)
    kurt_val <- e1071::kurtosis(eigen_vals)
    lap_vals <- eigen_vals[eigen_vals > 0]
    second_min <- if(length(lap_vals) > 1) sort(lap_vals)[2] else NA
    lap_entropy <- -sum((lap_vals/sum(lap_vals)) * log(lap_vals/sum(lap_vals)))
    lap_sum <- sum(lap_vals)

    if (weighted) {
      return(data.frame(
        lap_max = max_val,
        lap_fiedler = fiedler_val,
        lap_var = var_val,
        lap_skew = skew_val,
        lap_mean = mean_val,
        lap_kurt = kurt_val,
        lap_second_min = second_min,
        lap_entropy = lap_entropy,
        lap_sum = lap_sum
      ))
    } else {
      return(data.frame(
        lap_max_uw = max_val,
        lap_fiedler_uw = fiedler_val,
        lap_var_uw = var_val,
        lap_skew_uw = skew_val,
        lap_mean_uw = mean_val,
        lap_kurt_uw = kurt_val,
        lap_second_min_uw = second_min,
        lap_entropy_uw = lap_entropy,
        lap_sum_uw = lap_sum
      ))
    }

    
  }
  stop("input object has to be phylo or ltable")
}

# --- Feature functions ---
compute_adjacency_features <- function(tr) {
  g <- ape::as.igraph.phylo(tr, directed = FALSE)
  degs <- degree(g)
  data.frame(
    adj_sum = sum(degs),
    adj_var_degree = var(degs)
  )
}