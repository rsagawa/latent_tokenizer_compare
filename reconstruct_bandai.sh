#!/usr/bin/env bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gae50879

set -euo pipefail

# DATA_ROOT=${DATA_ROOT:-../Bandai/HumanML3D_Bandai_20FPS}
DATA_ROOT=${DATA_ROOT:-../Bandai/HumanML3D_Bandai2_20FPS}
MOTIONGPT_METRIC_ROOT=${MOTIONGPT_METRIC_ROOT:-../MotionGPT_100FPS}
MOTIONGPT_RECON_ROOT=${MOTIONGPT_RECON_ROOT:-../MotionGPT}

# If SPLIT=all, run train/val/test sequentially for fair evaluation.
SPLIT=${SPLIT:-all}
if [[ "$SPLIT" == "all" ]]; then
  # SPLITS=(train val test)
  SPLITS=(all)
else
  SPLITS=("$SPLIT")
fi

SEQ2SEQ_CKPT=${SEQ2SEQ_CKPT:-experiments/seq2seq_tok/checkpoints_tf_logits_scale10_ce001_pos001_marg001_info001_comp4/best.pt}
# SEQ2SEQ_CKPT=${SEQ2SEQ_CKPT:-experiments/seq2seq_tok/checkpoints_logits_scale10_ce001_pos001_marg001_info001_l2pdrop05/best.pt}
SEQ2SEQ_A2B_MODE=${SEQ2SEQ_A2B_MODE:-tf_logits}
SEQ2SEQ_B2A_MODE=${SEQ2SEQ_B2A_MODE:-tf_proj}
SEQ2SEQ_COMPRESSION_RATIO=${SEQ2SEQ_COMPRESSION_RATIO:-4}
# SEQ2SEQ_COMPRESSION_RATIO=${SEQ2SEQ_COMPRESSION_RATIO:-1}
SEQ2SEQ_OUT=${SEQ2SEQ_OUT:-experiments/bandai/recon_out_proposed}

M2DM_SAVE_DIR=${M2DM_SAVE_DIR:-experiments/tvqvae_m2dm_20FPS}
M2DM_CKPT=${M2DM_CKPT:-$M2DM_SAVE_DIR/checkpoints/ckpt_best.pt}
M2DM_OUT=${M2DM_OUT:-experiments/bandai/recon_out_m2dm}

MOTIONGPT_CFG_ASSETS=${MOTIONGPT_CFG_ASSETS:-configs/assets.yaml}
MOTIONGPT_CFG_STAGE1=${MOTIONGPT_CFG_STAGE1:-configs/config_h3d_stage1.yaml}
MOTIONGPT_CKPT=${MOTIONGPT_CKPT:-experiments/mgpt/VQVAE_HumanML3D/checkpoints/last.ckpt}
MOTIONGPT_OUT=${MOTIONGPT_OUT:-experiments/bandai/recon_out_MotionGPT}

METRIC_CFG_ASSETS=${METRIC_CFG_ASSETS:-$MOTIONGPT_METRIC_ROOT/configs/assets.yaml}
METRIC_CFG=${METRIC_CFG:-$MOTIONGPT_METRIC_ROOT/configs/config_h3d_stage3.yaml}
T2M_DIR=${T2M_DIR:-$MOTIONGPT_METRIC_ROOT/deps/t2m}
META_DIR=${META_DIR:-$MOTIONGPT_METRIC_ROOT/assets/meta}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-experiments/bandai/recon_eval}

mkdir -p "$EVAL_OUT_DIR"

run_eval() {
  local recon_dir="$1"
  local split="$2"
  local tag="$3"
  local out_json="$EVAL_OUT_DIR/${tag}_${split}.json"

  python eval_recon_humanml3d_motiongpt_metrics.py \
    --cfg_assets "$METRIC_CFG_ASSETS" \
    --cfg "$METRIC_CFG" \
    --data_root "$DATA_ROOT" \
    --recon_dir "$recon_dir/$split" \
    --split "$split" \
    --t2m_path "$T2M_DIR" \
    --meta_dir "$META_DIR" \
    --out_json "$out_json"

  echo "[DONE][eval] ${tag} split=${split}: ${out_json}"
}

for split in "${SPLITS[@]}"; do
  echo "[RUN] seq2seq reconstruct split=${split}"
  python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
    reconstruct \
    --split "$split" \
    --data_root "$DATA_ROOT" \
    --ckpt "$SEQ2SEQ_CKPT" \
    --a2b_mode "$SEQ2SEQ_A2B_MODE" \
    --b2a_mode "$SEQ2SEQ_B2A_MODE" \
    --compression_ratio "$SEQ2SEQ_COMPRESSION_RATIO" \
    --out_dir "$SEQ2SEQ_OUT"
  run_eval "$SEQ2SEQ_OUT" "$split" "seq2seq"
done

# for split in "${SPLITS[@]}"; do
#   echo "[RUN] m2dm reconstruct split=${split}"
#   python transformer_vqvae_humanml3d_m2dm.py \
#     reconstruct \
#     --split "$split" \
#     --data_root "$DATA_ROOT" \
#     --save_dir "$M2DM_SAVE_DIR" \
#     --ckpt "$M2DM_CKPT" \
#     --out_recon_dir "$M2DM_OUT"
#   run_eval "$M2DM_OUT" "$split" "m2dm"
# done

# for split in "${SPLITS[@]}"; do
#   echo "[RUN] MotionGPT VQVAE reconstruct split=${split}"
#   python reconstruct_vae_feats.py \
#     --motiongpt_root "$MOTIONGPT_RECON_ROOT" \
#     --dataset_root "$DATA_ROOT" \
#     --split "$split" \
#     --out_dir "$MOTIONGPT_OUT" \
#     --cfg_assets "$MOTIONGPT_CFG_ASSETS" \
#     --cfg "$MOTIONGPT_CFG_STAGE1" \
#     --ckpt_tar "$MOTIONGPT_CKPT"
#   run_eval "$MOTIONGPT_OUT" "$split" "motiongpt"
# done

echo "[DONE] Bandai reconstruction+evaluation finished."
