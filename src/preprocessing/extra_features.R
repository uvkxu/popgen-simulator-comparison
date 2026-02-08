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
    adj_mat <- lap_mat
    lap_mat <- diag(degrees) - lap_mat

  
    eig <- eigen(lap_mat, symmetric = TRUE)
    eigen_vals <- eig$values
    eigen_vals <- round(eigen_vals, digits = 10)
    eigen_vectors <- eig$vectors
    eigen_vectors <- round(eigen_vectors, digits = 10)

    positive_vals <- eigen_vals[eigen_vals > 0]
    fiedler_val <- min(positive_vals)

    fiedler_index <- which(eigen_vals == fiedler_val)[1]
    fiedler_vector <- eigen_vectors[, fiedler_index]

    zero_indices <- which(abs(fiedler_vector) == 0.0)
    char_vertices <- integer(0)
    cut_edge <- integer(0)
    tree_type <- 0

    if(length(zero_indices) == 0) {
        # find edge with sign change

        for(u in 1:mat_size) {
            for(v in 1:mat_size) {
                if(adj_mat[u,v] != 0 && fiedler_vector[u] > 0.0 && fiedler_vector[v] < 0.0) {
                    char_vertices <- unique(c(char_vertices, u, v))
                    cut_edge <- unique(c(cut_edge, u, v))
                }
            }
        }

        if(length(char_vertices) == 2) {
            tree_type <- 2
        } 
    } else {
        # connected subgraph of zeros
        for(u in zero_indices) {
            for(v in setdiff(1:mat_size, zero_indices)) {
                if(adj_mat[u,v] != 0.0) {
                    char_vertices <- unique(c(char_vertices, u))
                    cut_edge <- unique(c(cut_edge, u, v))
                }
            }
        }

        if(length(char_vertices) == 1) {
            tree_type <- 1
        }
    }

    max_val <- max(eigen_vals)
    mean_val <- mean(eigen_vals)
    var_val <- var(eigen_vals)
    skew_val <- e1071::skewness(eigen_vals)
    kurt_val <- e1071::kurtosis(eigen_vals)
    lap_vals <- eigen_vals[eigen_vals > 0]
    second_min <- if(length(lap_vals) > 1) sort(lap_vals)[2] else NA
    p <- lap_vals / sum(lap_vals)
    p <- p[p > 0]
    lap_entropy <- -sum(p * log(p))
    lap_sum <- sum(lap_vals)


    if (weighted) {
      if (tree_type != 0) {
        # Falls gültiger Tree-Typ vorhanden
        cut_edge_weight <- if (exists("adj_mat") && length(cut_edge) == 2)
          adj_mat[cut_edge[1], cut_edge[2]]
        else
          -1

        share_pos <- mean(fiedler_vector > 0)
        share_neg <- mean(fiedler_vector < 0)
        share_zero <- mean(fiedler_vector == 0)
        ratio_neg_pos <- if (share_pos > 0) share_neg / share_pos else 0

        dist_root <- ape::node.depth.edgelength(tr)
        dist_char <- if (length(char_vertices) > 0) dist_root[char_vertices] else -1
        dist_char <- mean(dist_char)

      } else {
        # Dummywerte für ungültige oder unklassifizierte Bäume
        cut_edge_weight <- -1
        share_pos <- -1
        share_neg <- -1
        share_zero <- -1
        ratio_neg_pos <- -1
        dist_char <- -1
      }

      return(data.frame(
        lap_max = max_val,
        lap_fiedler = fiedler_val,
        lap_var = var_val,
        lap_skew = skew_val,
        lap_mean = mean_val,
        lap_kurt = kurt_val,
        lap_second_min = second_min,
        lap_entropy = lap_entropy,
        lap_sum = lap_sum,
        tree_type = tree_type,
        cut_edge_weight = cut_edge_weight,
        share_pos = share_pos,
        share_neg = share_neg,
        share_zero = share_zero,
        ratio_neg_pos = ratio_neg_pos,
        dist_char = dist_char
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
        lap_sum_uw = lap_sum,
        tree_type_uw = tree_type
      ))
    }

    
  }
  stop("input object has to be phylo or ltable")
}

compute_characteristic_features <- function(g,
                                            lap_mat,
                                            fiedler_vector,
                                            char_vertices = integer(0),
                                            tree_type = 0) {
  # --- Kein charakteristischer Typ: Dummywerte ---
  if (tree_type == 0 || is.null(fiedler_vector)) {
    return(list(
      cut_weight = 0,
      conductance = 0,
      normalized_cut = 0,
      bottleneck_strength = 0,
      density_pos = -1,
      density_neg = -1,
      modularity = -1,
      fiedler_sd = 0
    ))
  }

  # Partitionen
  V_pos <- which(fiedler_vector > 0)
  V_neg <- which(fiedler_vector < 0)
  if (length(V_pos) == 0 || length(V_neg) == 0) {
    return(list(
      cut_weight = 0,
      conductance = 0,
      normalized_cut = 0,
      bottleneck_strength = 0,
      density_pos = -1,
      density_neg = -1,
      modularity = -1,
      fiedler_sd = sd(fiedler_vector)
    ))
  }

  el <- igraph::as_edgelist(g, names = FALSE)
  signs <- fiedler_vector[el[, 1]] * fiedler_vector[el[, 2]]
  cut_edges <- which(signs < 0)
  cut_weight <- sum(igraph::E(g)$weight[cut_edges])

  # Volumina
  wdeg <- igraph::strength(g)
  vol_pos <- sum(wdeg[V_pos])
  vol_neg <- sum(wdeg[V_neg])

  small_eps <- 1e-12
  conductance <- cut_weight / (min(vol_pos + small_eps, vol_neg + small_eps))
  normalized_cut <- cut_weight * (1 / (vol_pos + small_eps) + 1 / (vol_neg + small_eps))
  bottleneck_strength <- cut_weight / min(vol_pos + small_eps, vol_neg + small_eps)

  induced_pos <- igraph::induced_subgraph(g, V_pos)
  induced_neg <- igraph::induced_subgraph(g, V_neg)

  dens_pos <- igraph::edge_density(induced_pos, loops = FALSE)
  dens_neg <- igraph::edge_density(induced_neg, loops = FALSE)

  n <- length(fiedler_vector)
  membership <- rep(1, n)
  membership[V_neg] <- 2
  modularity_val <- igraph::modularity(g, membership)

  return(list(
    cut_weight = cut_weight,
    conductance = conductance,
    normalized_cut = normalized_cut,
    bottleneck_strength = bottleneck_strength,
    density_pos = dens_pos,
    density_neg = dens_neg,
    modularity = modularity_val,
    fiedler_sd = sd(fiedler_vector)
  ))
}