#!/usr/bin/env bash
set -euo pipefail

# Usage:
# run_with_errorlog.sh <error_log_path> <command...>

ERRFILE="$1"
shift

TMP_ERR=$(mktemp)

if ! "$@" 2> "$TMP_ERR"; then
    mkdir -p "$(dirname "$ERRFILE")"
    mv "$TMP_ERR" "$ERRFILE"
    exit 1
fi

rm -f "$TMP_ERR"
