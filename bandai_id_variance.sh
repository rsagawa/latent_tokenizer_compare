#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-experiments/bandai}"
OUT_BASE="${2:-bandai_out_id_var}"

if [ ! -d "$BASE_DIR" ]; then
    echo "[ERROR] base directory not found: $BASE_DIR" >&2
    exit 1
fi

for root in "$BASE_DIR"/*; do
    [ -d "$root" ] || continue
    name="$(basename "$root")"
    out_dir="$OUT_BASE/$name"

    echo "[RUN] root=$root -> out=$out_dir"
    python bandai_id_variance_within_between_style.py \
        --root "$root" \
        --out "$out_dir" \
        --min-total-count 2 \
        --max-plot 50000 \
        --annotate-top 50 \
        --binary 0 \
        --var-normalization id_mean \
        --score-mode distance \
        --probability-mode per_file_then_id_l1
done
