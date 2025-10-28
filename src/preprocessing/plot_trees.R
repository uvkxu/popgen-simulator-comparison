library(phytools)

char_vertices <- function(tr, label, tol = 1e-5) {
    edges <- tr$edge
    weights <- tr$edge.length
    n <- length(unique(as.vector(edges)))
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

    res <- RSpectra::eigs_sym(
      lap_mat,
      k = 1,              
      which = "BE",       
      sigma = 1e-10,
      opts = list(retvec = TRUE)
    )

    # Extract
    min_val <- res$values[1]      # smallest eigenvalue
    fiedler_vector <- res$vectors[, 1]  # corresponding eigenvector

    zero_indices <- which(abs(fiedler_vector) < tol)
    result <- integer(0)

    if(length(zero_indices) == 0) {
        # find edge with sign change
        pairs <- list()

        for(u in 1:mat_size) {
            for(v in 1:mat_size) {
                if(lap_mat[u,v] != 0 && fiedler_vector[u] >= tol && fiedler_vector[v] <= tol) {
                    pairs <- append(pairs, list(c(u,v)))
                    result <- unique(c(result, u, v))
                }
            }
        }

        if(length(pairs) > 1) {
            print("⚠️ More than one pair of characteristic vertices found!")
            for(p in pairs) {
                cat("Pair:", p[1], p[2], "Fiedler values:", fiedler_vector[p[1]], fiedler_vector[p[2]], "\n")
            }
        } 
    } else {
        # connected subgraph of zeros
        for(u in zero_indices) {
            for(v in setdiff(1:mat_size, zero_indices)) {
                if(lap_mat[u,v] >= tol) {
                    result <- unique(c(result, u))
                }
            }
        }

        if(length(result) > 1) {
            print("⚠️ More than one characteristic vertex found in zero_indices case!")
            print(length(result))
        }
    }
    invalid_edges <- which(weights <= 0)

    if(any(weights <= 0)){
      print(label)
      if(length(result) == 0) {
        print("⚠️ No characteristic vertex was found and invalid edge lengths (non positive)")
        print(length(invalid_edges))
      } else {
        print("Invalid edge lengths (non positive).")
      }
    } else if(length(result) == 0) {
        print("⚠️ No characteristic vertex was found")
    }

    return(list(char_vertices = result, fiedler_vector = fiedler_vector))
}


plot_trees <- function(trees_list, char_vertices_list, fiedler_vector_list, sim_name_list, output_file = "trees.pdf", tol = 1e-5) {

  pdf(output_file, width = 10, height = 6)
  
  n_trees <- length(trees_list)
  
  for(i in seq_len(n_trees)) {
    tr <- trees_list[[i]]
    char_v <- char_vertices_list[[i]]
    fiedler_v <- fiedler_vector_list[[i]]
    sim_name <- sim_name_list[[i]]
    
    # Edge width proportional zu branch length
    edge_w <- tr$edge.length / max(tr$edge.length) * 5
    
    plot(tr, show.node.label = FALSE, show.tip.label = FALSE, edge.width = edge_w)
    
    # Koordinaten holen
    coords <- get("last_plot.phylo", envir = .PlotPhyloEnv)
    n_nodes <- length(tr$tip.label) + tr$Nnode
    
    for(v in seq_len(n_nodes)) {
      val <- fiedler_v[v]
      
      # Farbe und Größe nach Kategorie
      if(v %in% char_v) {
        col <- "orange"  # Charakteristische Vertices
        cex <- 1.8
        pch <- 19
      } else if(abs(val) < tol) {
        col <- "blue"    # Fiedler ~0
        cex <- 1.2
        pch <- 19
      } else if(val >= tol) {
        col <- "green"   # Positiv
        cex <- 1.2
        pch <- 19
      } else if(val <= -tol) {
        col <- "red"     # Negativ
        cex <- 1.2
        pch <- 19
      } else {
        col <- "black"   # Alle anderen
        cex <- 1
        pch <- 19
      }
      
      # Tip oder internal node
      if(v <= length(tr$tip.label)) {
        points(coords$xx[v], coords$yy[v], col = col, pch = pch, cex = cex)
      } else {
        int_idx <- v - length(tr$tip.label)
        points(coords$xx[length(tr$tip.label) + int_idx], 
               coords$yy[length(tr$tip.label) + int_idx], 
               col = col, pch = pch, cex = cex)
      }
    }
    
    # Titel mit Simulatorname
    title(main = paste(sim_name, "tree", i))
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
    fiedler_vector_list <- list()
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
        
        char_info <- char_vertices(tr, sim_name, tol)
        char_vertices_list <- append(char_vertices_list, list(char_info$char_vertices))
        fiedler_vector_list <- append(fiedler_vector_list, list(char_info$fiedler_vector))
        sim_name_list <- append(sim_name_list, sim_name)
      }
    }
    
    if(length(trees_to_plot) == 0) next
    
    # Output-Pfad vorbereiten
    rel_dir <- sub(paste0("^", normalizePath(input_path, winslash="/"), "/?"), "", normalizePath(d, winslash="/"))
    out_dir <- file.path(output_path, rel_dir)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    
    # PDF-Plot
    out_file <- file.path(out_dir, "trees_plot.pdf")
    plot_trees(trees_to_plot, char_vertices_list, fiedler_vector_list, sim_name_list, out_file, tol)
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