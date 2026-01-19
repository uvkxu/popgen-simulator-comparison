library(ape)

# --- Argument parsing ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript plot_ultrametric.R <input_dir> <output_file>")
input_dir <- args[1]
output_file <- args[2]

# --- Listen für die Simulatoren ---
trees_ms <- list()
trees_msms <- list()
trees_msprime <- list()

is_tree_line <- function(line) {
  grepl("\\(", line) & grepl(";+", line) &
    !grepl("^\\s*//", line) &
    !grepl("^\\s*segsites", line, ignore.case = TRUE) &
    !grepl("^\\s*positions", line, ignore.case = TRUE)
}

clean_tree_line <- function(line) {
  trimws(sub(";+$", ";", line))
}

# --- Alle txt-Dateien im Input-Ordner durchgehen ---
files <- list.files(path = input_dir, pattern = "\\.txt$", full.names = TRUE, recursive = TRUE)
for (input_file in files) {
  con <- file(input_file, open="r")
  while (length(line <- readLines(con, n=1, warn=FALSE)) > 0) {
    if (!is_tree_line(line)) next
    line <- clean_tree_line(line)
    tr <- ape::read.tree(text=line)

    # je nach Dateiname in Liste packen
    fname <- basename(input_file)
    if (grepl("ms_", fname)) {
      trees_ms[[length(trees_ms) + 1]] <- tr
    } else if (grepl("msms_", fname)) {
      trees_msms[[length(trees_msms) + 1]] <- tr
    } else if (grepl("msprime_", fname)) {
      trees_msprime[[length(trees_msprime) + 1]] <- tr
    }
  }
  close(con)
}

# --- Toleranzen definieren ---
tols <- 10^seq(-10, -5, length.out=20)  # von 1e-10 bis 1e-5

# --- Funktion: Fraction of ultrametric trees bei gegebener tol ---
frac_ultrametric <- function(trees, tols) {
  sapply(tols, function(t) mean(sapply(trees, ape::is.ultrametric, tol=t)))
}

# --- Fractions berechnen ---
fractions_ms      <- frac_ultrametric(trees_ms, tols)
fractions_msms    <- frac_ultrametric(trees_msms, tols)
fractions_msprime <- frac_ultrametric(trees_msprime, tols)

# --- Plot ---
png(output_file, width=800, height=600)
plot(tols, fractions_ms, type="b", log="x", ylim=c(0,1),
     xlab="Tolerance (tol)", ylab="Fraction of ultrametric trees",
     col="red", pch=16, lwd=2)
lines(tols, fractions_msms, type="b", col="blue", pch=17, lwd=2)
lines(tols, fractions_msprime, type="b", col="green", pch=15, lwd=2)
legend("bottomleft", legend=c("ms", "msms", "msprime"), 
       col=c("red","blue","green"), pch=c(16,17,15), lwd=2)
abline(v=1e-8, lty=2, col="darkgrey")
dev.off()
