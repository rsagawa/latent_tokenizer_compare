#!/usr/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gae50879

# #PBS -l walltime=8:00:00
# #PBS -l walltime=12:00:00
# #PBS -l walltime=24:00:00

# cd ${PBS_O_WORKDIR}

# source ~/.bashrc

# conda activate mgpt

# DATA_ROOT_20FPS=../Bandai/HumanML3D_Bandai_20FPS
DATA_ROOT_20FPS=../Bandai/HumanML3D_Bandai2_20FPS
SAVE_DIR_20FPS=./experiments/tvqvae_m2dm_20FPS

CKPT_PATH=$SAVE_DIR_20FPS/checkpoints/ckpt_best.pt
# OUT_DIR=./experiments/bandai/tokens_out_m2dm
OUT_DIR=./experiments/bandai/tokens_out2_m2dm

# SPLITS=(train)
# SPLITS=(val)
# SPLITS=(test)
SPLITS=(train val test)
# SPLITS=(all)

for SPLIT in "${SPLITS[@]}"; do
  echo "[INFO] Encoding split=${SPLIT}"
  python transformer_vqvae_humanml3d_m2dm.py \
    encode \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT_20FPS" \
    --save_dir "$SAVE_DIR_20FPS" \
    --ckpt "$CKPT_PATH" \
    --out_tokens_dir "$OUT_DIR"
done

echo "[DONE] token IDs saved under: ${OUT_DIR}/{train,val,test,all}"
