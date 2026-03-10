#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DIR="${BASE_DIR:-experiments/bandai}"
HML_ROOT="${HML_ROOT:-../Bandai/HumanML3D_Bandai2_20FPS}"

BATCH_SIZE="${BATCH_SIZE:-64}"
BASE_EPOCHS="${EPOCHS:-20}"
BASE_LR="${LR:-2e-4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
DEVICE="${DEVICE:-cuda}"

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

  if [[ "${name}" == "proposed" ]]; then
    EPOCHS=40
    LR=5e-5
  else
    EPOCHS="${BASE_EPOCHS}"
    LR="${BASE_LR}"
  fi

  token_root="${BASE_DIR}/tokens_out2_${name}"
  out_dir="${BASE_DIR}/retrieval2_${name}"

  echo "[${idx}/${TOTAL}] retrieval: ${name} (vocab_size=${vocab_size})"
  "${PYTHON_BIN}" train_eval_retrieval_hml_tokens.py \
    --vocab_size "${vocab_size}" \
    --hml_root "${HML_ROOT}" \
    --token_root "${token_root}" \
    --out_dir "${out_dir}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --max_tokens "${MAX_TOKENS}" \
    --device "${DEVICE}"

  EPOCHS="${BASE_EPOCHS}"
  LR="${BASE_LR}"
  out_dir="${BASE_DIR}/actionrec2_${name}"

  echo "[${idx}/${TOTAL}] actionrec: ${name} (vocab_size=${vocab_size})"
  "${PYTHON_BIN}" train_eval_actionrec_hml_tokens.py \
    --vocab_size "${vocab_size}" \
    --hml_root "${HML_ROOT}" \
    --token_root "${token_root}" \
    --out_dir "${out_dir}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --max_tokens "${MAX_TOKENS}" \
    --device "${DEVICE}"

done

echo "done."
