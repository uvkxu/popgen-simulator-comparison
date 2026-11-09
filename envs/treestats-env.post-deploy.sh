
#!/bin/bash

set -eu

Rscript -e '
if (!requireNamespace("treestats", quietly = TRUE)) {
    devtools::install_github("thijsjanzen/treestats")
}
'
