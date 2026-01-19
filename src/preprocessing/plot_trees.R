library(phytools)
library(ape)
library(igraph)

char_vertices <- function(tr) {
    g <- ape::as.igraph.phylo(tr, directed = FALSE)
    if (!is.null(tr$edge.length)) {
      E(g)$weight <- tr$edge.length
    } else {
      E(g)$weight <- 1
    }

    weights <- tr$edge.length
    invalid_edges <- which(weights <= 0)

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
    tree_type <- 0

    if(length(zero_indices) == 0) {
        # find edge with sign change

        for(u in 1:mat_size) {
            for(v in 1:mat_size) {
                if(adj_mat[u,v] != 0 && fiedler_vector[u] > 0.0 && fiedler_vector[v] < 0.0) {
                    cat("Char vertices: ", u, v, "\n")
                    char_vertices <- unique(c(char_vertices, u, v))
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
                if(adj_mat[u,v] > 0.0) {
                    char_vertices <- unique(c(char_vertices, u))
                }
            }
        }

        if(length(char_vertices) == 1) {
            tree_type <- 1
        }
    }

    if (length(char_vertices) < 1) {
        if(any(weights <= 0)) {
          print("⚠️ No characteristic vertex was found and invalid edge lengths (non positive)")
          print(length(invalid_edges))
        } else {
          print("⚠️ No characteristic vertex was found but no invalid edge lengths")
        }
    } else if (tree_type == 2 && length(char_vertices) > 2) {
        if(any(weights <= 0)) {
          print("⚠️ More than one pair of characteristic vertices found and invalid edge lengths (non positive)!")
        } else {
          print("⚠️ More than one pair of characteristic vertices found but no invalid edge lengths")
        }
    } else if (tree_type == 1 && length(char_vertices) > 1) {
        if(any(weights <= 0)) {
          print("⚠️ More than one characteristic vertex found in zero_indices case and invalid edge lengths (non positive)!")
        } else {
          print("⚠️ More than one characteristic vertex found in zero_indices case but no invalid edge lengths")
        }
    } else if (any(weights <= 0)) {
        print("⚠️ Invalid edges do exist!")
    }

    el <- igraph::as_edgelist(g, names = FALSE)
    signs <- fiedler_vector[el[, 1]] * fiedler_vector[el[, 2]]
    cut_edges <- which(signs < 0)

    for (idx in cut_edges) {
      u <- el[idx, 1]
      v <- el[idx, 2]
      cat("Edge", idx, "between vertices", u, "and", v, 
          "has Fiedler values:", fiedler_vector[u], ",", fiedler_vector[v],
          "adj matrix entry:", adj_mat[u,v], "\n")
    }
    cat("Number of cut edges: ", length(cut_edges), "\n")
    cat("Tree type: ", tree_type, "\n")

    return(list(char_vertices = char_vertices, cut_edges = cut_edges, fiedler_vector = fiedler_vector, tree_type = tree_type))
}


plot_trees <- function(trees_list, char_vertices_list, cut_edges_list, 
                       fiedler_vector_list, tree_type_list, sim_name_list, 
                       output_file = "trees.pdf") {
  
  pdf(output_file, width = 10, height = 6)
  
  n_trees <- length(trees_list)
  
  for(i in seq_len(n_trees)) {
    tr <- trees_list[[i]]
    char_v <- char_vertices_list[[i]]
    cut_edges <- cut_edges_list[[i]]
    fiedler_v <- fiedler_vector_list[[i]]
    tree_type <- tree_type_list[[i]]
    sim_name <- sim_name_list[[i]]

    if (tree_type == 0) next
    
    # 🟢 Edge width proportional zu branch length
    edge_w <- tr$edge.length / max(tr$edge.length) * 4
    
    # 🔵 Edges einfärben – Cut edges hervorheben
    edge_col <- rep("grey40", nrow(tr$edge))
    edge_col[cut_edges] <- "deepskyblue3"
    
    plot(tr, show.node.label = FALSE, show.tip.label = FALSE, 
         edge.width = edge_w, edge.color = edge_col)
    
    # 🔹 Koordinaten holen
    coords <- get("last_plot.phylo", envir = .PlotPhyloEnv)
    n_nodes <- length(tr$tip.label) + tr$Nnode
    
    for(v in seq_len(n_nodes)) {
      val <- fiedler_v[v]
      
      # Farbe und Größe nach Kategorie
      if(v %in% char_v) {
        col <- "orange"
        cex <- 1.8
        pch <- 21
      } else if(abs(val) == 0) {
        col <- "blue"
        cex <- 1.3
        pch <- 19
      } else if(val > 0) {
        col <- "green3"
        cex <- 1.2
        pch <- 19
      } else if(val < 0) {
        col <- "red3"
        cex <- 1.2
        pch <- 19
      } else {
        col <- "black"
        cex <- 1
        pch <- 19
      }
      
      points(coords$xx[v], coords$yy[v], col = col, pch = pch, cex = cex)
    }
    
    title(main = paste(sim_name, "- tree", i))
    
    # Get plot limits
    usr <- par("usr")  # returns c(x1, x2, y1, y2)

    # place legend slightly below the visible tree
    legend(x = usr[1] + 0.8*(usr[2] - usr[1]),  # x position
          y = usr[3] + 0.05*(usr[4] - usr[3]), # y position
          legend = c("char vertex", "cut edge", "fiedler > 0", "fiedler < 0", "fiedler ~ 0"),
          col = c("orange", "deepskyblue3", "green3", "red3", "blue"),
          pch = c(21, NA, 19, 19, 19),
          lty = c(NA, 1, NA, NA, NA),
          pt.cex = c(1.5, NA, 1.2, 1.2, 1.2),
          bty = "n", cex = 0.8)
  }
  
  dev.off()
}


# --- Hauptfunktion zur Verarbeitung aller Ordner ---
process_all_folders <- function(input_path, output_path) {
  tol <- 1e-9
  all_dirs <- list.dirs(input_path, recursive = TRUE, full.names = TRUE)
  
  for(d in all_dirs) {
    txt_files <- list.files(d, pattern = "\\.txt$", full.names = TRUE)
    if(length(txt_files) == 0) next
    
    # Liste zum Sammeln von Bäumen
    trees_to_plot <- list()
    char_vertices_list <- list()
    cut_edges_list <- list()
    fiedler_vector_list <- list()
    tree_type_list <- list()
    sim_name_list <- list()
    
    # Drei Dateien max
    txt_files <- head(txt_files, 3)
    
    for(f in txt_files) {
      lines <- readLines(f, warn = FALSE)
      # Filter Newick lines
      tree_lines <- trimws(sub(";+$",";", lines[grepl("\\(", lines) & grepl(";+", lines)]))
      
      # Drei Bäume pro Datei
      tree_lines <- head(tree_lines, 3)

      sim_name <- tools::file_path_sans_ext(basename(f))
      
      for(txt in tree_lines) {
        tr <- tryCatch(ape::read.tree(text = txt), error = function(e) NULL)
        if(is.null(tr)) next
        trees_to_plot <- append(trees_to_plot, list(tr))
        
        char_info <- char_vertices(tr)
        char_vertices_list <- append(char_vertices_list, list(char_info$char_vertices))
        cut_edges_list <- append(cut_edges_list, list(char_info$cut_edges))
        fiedler_vector_list <- append(fiedler_vector_list, list(char_info$fiedler_vector))
        tree_type_list <- c(tree_type_list, char_info$tree_type)
        sim_name_list <- c(sim_name_list, sim_name)
      }
    }
    
    if(length(trees_to_plot) == 0) next
    
    # Output-Pfad vorbereiten
    rel_dir <- sub(paste0("^", normalizePath(input_path, winslash="/"), "/?"), "", normalizePath(d, winslash="/"))
    out_dir <- file.path(output_path, rel_dir)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    
    # PDF-Plot
    out_file <- file.path(out_dir, "trees_plot.pdf")
    plot_trees(trees_to_plot, char_vertices_list, cut_edges_list, fiedler_vector_list, tree_type_list, sim_name_list, out_file)
  }
}

# --- Usage ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_treestats_clean.R <input_txt|input_dir> <output_dir>")

input_path <- args[1]
output_dir <- args[2]

if (!file.exists(input_path)) stop("Input file/directory does not exist: ", input_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

process_all_folders(input_path, output_dir)