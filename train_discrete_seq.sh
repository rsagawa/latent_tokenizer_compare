#!/usr/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gae50879

# #PBS -l walltime=8:00:00
# #PBS -l walltime=12:00:00
# #PBS -l walltime=24:00:00

cd ${PBS_O_WORKDIR}

# export PYTHONPATH=.:./dlimp
# export HF_HOME=/groups/gae50879/sagawa/hugging_face
# export NCCL_P2P_DISABLE=1

source ~/.bashrc

conda activate mgpt

DATA_ROOT_20FPS=../HumanML3D/HumanML3D_20FPS
DATA_ROOT_100FPS=../HumanML3D/HumanML3D_100FPS
SAVE_DIR_20FPS=./experiments/tvqvae_discrete_seq_20FPS
SAVE_DIR_100FPS=./experiments/tvqvae_discrete_seq_100FPS
MOTIONGPT_DIR=../MotionGPT_100FPS
T2M_DIR=../MotionGPT_100FPS/deps/t2m/

# python discrete_seq_autoencoder_humanml3d_baseline_b.py train \
#     --data_root $DATA_ROOT_20FPS \
#     --save_dir $SAVE_DIR_20FPS \
#     --normalize \
#     --epochs 500 --batch_size 64 --lr 1e-4 \
#     --max_motion_len 64 --patch_len 1 \
#     --d_model 512 --n_layers 4 --n_heads 8 \
#     --vocab_size 8192 \
#     --kl_weight 0.1 --kl_anneal_steps 20000 \
#     --tau_start 1.0 --tau_end 0.3 --tau_anneal_steps 20000

# python discrete_seq_autoencoder_humanml3d_baseline_b.py train \
#     --data_root $DATA_ROOT_100FPS \
#     --save_dir $SAVE_DIR_100FPS \
#     --normalize \
#     --epochs 500 --batch_size 64 --lr 1e-4 \
#     --max_motion_len 64 --patch_len 1 \
#     --d_model 512 --n_layers 4 --n_heads 8 \
#     --vocab_size 8192 \
#     --kl_weight 0.1 --kl_anneal_steps 20000 \
#     --tau_start 1.0 --tau_end 0.3 --tau_anneal_steps 20000

# python discrete_seq_autoencoder_humanml3d_baseline_b.py reconstruct \
#   --data_root $DATA_ROOT_20FPS \
#   --save_dir  $SAVE_DIR_20FPS \
#   --ckpt      $SAVE_DIR_20FPS/checkpoints/ae_ckpt_best.pt \
#   --split test \
#   --normalize \
#   --save_input \
#   --out_recon_dir $SAVE_DIR_20FPS/recon_out \
#   --batch_size 64

python discrete_seq_autoencoder_humanml3d_baseline_b.py reconstruct \
  --data_root $DATA_ROOT_100FPS \
  --save_dir  $SAVE_DIR_100FPS \
  --ckpt      $SAVE_DIR_100FPS/checkpoints/ae_ckpt_best.pt \
  --split test \
  --normalize \
  --save_input \
  --out_recon_dir $SAVE_DIR_100FPS/recon_out \
  --batch_size 64

# python eval_recon_humanml3d_motiongpt_metrics.py \
#   --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
#   --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
#   --data_root  $DATA_ROOT_20FPS \
#   --recon_root $SAVE_DIR_20FPS/recon_out/test \
#   --split test \
#   --t2m_path   $T2M_DIR \
#   --meta_dir   $MOTIONGPT_DIR/assets/meta \
#   --out_json   $SAVE_DIR_20FPS/recon_eval_test.json

python eval_recon_humanml3d_motiongpt_metrics.py \
  --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
  --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
  --data_root  $DATA_ROOT_100FPS \
  --recon_root $SAVE_DIR_100FPS/recon_out/test \
  --split test \
  --t2m_path   $T2M_DIR \
  --meta_dir   $MOTIONGPT_DIR/assets/meta \
  --out_json   $SAVE_DIR_100FPS/recon_eval_test.json