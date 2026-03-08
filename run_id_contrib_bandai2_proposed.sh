#!/usr/bin/env bash
set -euo pipefail

# Run ID contribution analysis for 4 tokenizer sets:
#   MotionGPT, m2dm, proposed, latent_tokenizer
#
# Dataset/ckpt naming convention:
#   tokens_out2_{name}
#   retrieval2_{name}/best.pt
#   actionrec2_{name}/best.pt

PYTHON_BIN="${PYTHON_BIN:-python3}"
HML_ROOT="${HML_ROOT:-../Bandai/HumanML3D_Bandai2_20FPS}"
MOTIONGPT_ROOT="${MOTIONGPT_ROOT:-../MotionGPT}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-20}"
DEVICE="${DEVICE:-cuda}"

TOP_K_IDS="${TOP_K_IDS:-50}"
TOP_K_TYPES="${TOP_K_TYPES:-16}"
ID_TYPE_MODE="${ID_TYPE_MODE:-bucket}"
ID_TYPE_BUCKET_SIZE="${ID_TYPE_BUCKET_SIZE:-8192}"
ID_TYPE_MODULO="${ID_TYPE_MODULO:-4}"
ATTR_TOP_K="${ATTR_TOP_K:-0}"
ATTR_SCORE_TARGET_ACTIONREC="${ATTR_SCORE_TARGET_ACTIONREC:-true_logit}"
ATTR_SCORE_TARGET_RETRIEVAL="${ATTR_SCORE_TARGET_RETRIEVAL:-diag_cosine}"

BASE_DIR="${BASE_DIR:-experiments/bandai}"
OUT_SUFFIX="${OUT_SUFFIX:-id_contrib_test}"

NAMES=(MotionGPT m2dm proposed latent_tokenizer)
TOTAL=${#NAMES[@]}

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  idx=$((i + 1))

  if [[ "${name}" == "latent_tokenizer" ]]; then
    vocab_size=32064
  else
    vocab_size=8192
  fi

  token_root="${BASE_DIR}/tokens_out2_${name}"
  actionrec_ckpt="${BASE_DIR}/actionrec2_${name}/best.pt"
  retrieval_ckpt="${BASE_DIR}/retrieval2_${name}/best.pt"
  out_actionrec="${BASE_DIR}/actionrec2_${name}/${OUT_SUFFIX}"
  out_retrieval="${BASE_DIR}/retrieval2_${name}/${OUT_SUFFIX}"

  echo "[${idx}/${TOTAL}] actionrec: ${name} (vocab_size=${vocab_size})"
  "${PYTHON_BIN}" analyze_retrieval_actionrec_id_contrib.py \
    --task actionrec \
    --hml_root "${HML_ROOT}" \
    --token_root "${token_root}" \
    --pretrained_ckpt "${actionrec_ckpt}" \
    --vocab_size "${vocab_size}" \
    --out_dir "${out_actionrec}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --max_tokens "${MAX_TOKENS}" \
    --device "${DEVICE}" \
    --top_k_ids "${TOP_K_IDS}" \
    --top_k_types "${TOP_K_TYPES}" \
    --attr_top_k "${ATTR_TOP_K}" \
    --attr_score_target "${ATTR_SCORE_TARGET_ACTIONREC}" \
    --id_type_mode "${ID_TYPE_MODE}" \
    --id_type_bucket_size "${ID_TYPE_BUCKET_SIZE}" \
    --id_type_modulo "${ID_TYPE_MODULO}"

  echo "[${idx}/${TOTAL}] retrieval: ${name} (vocab_size=${vocab_size})"
  "${PYTHON_BIN}" analyze_retrieval_actionrec_id_contrib.py \
    --task retrieval \
    --motiongpt_root "${MOTIONGPT_ROOT}" \
    --hml_root "${HML_ROOT}" \
    --token_root "${token_root}" \
    --pretrained_ckpt "${retrieval_ckpt}" \
    --vocab_size "${vocab_size}" \
    --out_dir "${out_retrieval}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --max_tokens "${MAX_TOKENS}" \
    --max_text_len "${MAX_TEXT_LEN}" \
    --device "${DEVICE}" \
    --top_k_ids "${TOP_K_IDS}" \
    --top_k_types "${TOP_K_TYPES}" \
    --attr_top_k "${ATTR_TOP_K}" \
    --attr_score_target "${ATTR_SCORE_TARGET_RETRIEVAL}" \
    --id_type_mode "${ID_TYPE_MODE}" \
    --id_type_bucket_size "${ID_TYPE_BUCKET_SIZE}" \
    --id_type_modulo "${ID_TYPE_MODULO}"
done

echo "done."
