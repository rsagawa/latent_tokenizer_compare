#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DIR="${BASE_DIR:-experiments/bandai}"
HML_ROOT="${HML_ROOT:-../Bandai/HumanML3D_Bandai2_20FPS}"
MOTIONGPT_ROOT="${MOTIONGPT_ROOT:-../MotionGPT}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-20}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-200}"

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
  out_actionrec="${BASE_DIR}/actionrec2_${name}/id_attr_direct"
  out_retrieval="${BASE_DIR}/retrieval2_${name}/id_attr_direct"

  echo "[${idx}/${TOTAL}] actionrec attribution: ${name} (vocab_size=${vocab_size})"
  "${PYTHON_BIN}" estimate_id_attribution.py \
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
    --top_k "${TOP_K}"

  echo "[${idx}/${TOTAL}] retrieval attribution: ${name} (vocab_size=${vocab_size})"
  "${PYTHON_BIN}" estimate_id_attribution.py \
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
    --top_k "${TOP_K}"
done

echo "done."
