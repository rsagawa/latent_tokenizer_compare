#!/bin/bash
set -euo pipefail

# python3 visualize_id_contrib_bandai2.py \
#   --base_dir experiments/bandai \
#   --out_suffix id_contrib_test \
#   --out_dir experiments/bandai/id_contrib_viz

# python3 visualize_id_contrib_bandai2.py \
#   --base_dir experiments/bandai \
#   --out_suffix id_contrib_test \
#   --out_dir experiments/bandai/id_contrib_viz \
#   --top_k_ids_per_class 5 \
#   --top_k_classes 20

# python3 visualize_id_contrib_bandai2.py \
#   --base_dir experiments/bandai \
#   --out_suffix id_contrib_test \
#   --out_dir experiments/bandai/id_contrib_viz \
#   --top_k_ids 20

python3 visualize_id_contrib_bandai2.py \
  --base_dir experiments/bandai \
  --out_suffix id_contrib_test \
  --out_dir experiments/bandai/id_contrib_viz \
  # --top_k_ids 50 \
  # --attr_sample_aggregation_mode "${ATTR_SAMPLE_AGG_MODE:-sample_topk_count}" \
  # --sample_top_k_ids "${SAMPLE_TOP_K_IDS:-10}" \

  # --attr_sample_aggregation_mode "${ATTR_SAMPLE_AGG_MODE:-abs_sum}" \
  # --attr_sample_aggregation_mode "${ATTR_SAMPLE_AGG_MODE:-sample_topk_count}" \
  # --sample_top_k_ids "${SAMPLE_TOP_K_IDS:-30}" \
  # --attr_sample_aggregation_mode "${ATTR_SAMPLE_AGG_MODE:-abs_sum}" \
  # --sample_top_p "${SAMPLE_TOP_P:-0.8}"


python3 ./make_id_contrib_dashboard.py

python3 visualize_sparse_id_sequences.py \
  --id_contrib_dir experiments/bandai/actionrec2_latent_tokenizer/id_contrib_test \
  --top_k_clusters 25
