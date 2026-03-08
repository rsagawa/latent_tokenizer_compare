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

DATA_ROOT_20FPS=../HumanML3D/HumanML3D_20FPS
DATA_ROOT_100FPS=../HumanML3D/HumanML3D_100FPS
SAVE_DIR_20FPS=./experiments/seq2seq_tok
SAVE_DIR_100FPS=./experiments/seq2seq_tok_100FPS
MOTIONGPT_DIR=../MotionGPT_100FPS
T2M_DIR=../MotionGPT_100FPS/deps/t2m/

# CKPT_PATH=experiments/seq2seq_tok/checkpoints_para_supervised_scale10_ce1_pos001_marg001_info001/last.pt
# A2B_MODE=tf_logits
# B2A_MODE=tf_proj

# CKPT_PATH=experiments/seq2seq_tok/checkpoints_tf_teacher_scale10_ce001_pos001_marg001_info001/best.pt
# A2B_MODE=tf_logits
# B2A_MODE=tf_proj

# CKPT_PATH=experiments/seq2seq_tok/checkpoints_tf_logits_scale10_ce001_pos001_marg001_info001/best.pt
# A2B_MODE=tf_logits
# B2A_MODE=tf_proj

CKPT_PATH=experiments/seq2seq_tok/checkpoints_tf_logits_scale10_ce001_pos001_marg001_info001_comp4_lr1e-6/best.pt
A2B_MODE=tf_logits
B2A_MODE=tf_proj
# A2B_MODE=ar
# B2A_MODE=ar

# SPLIT=train
# SPLIT=val
SPLIT=test

COMPRESSION_RATIO=4

# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     reconstruct \
#     --split $SPLIT \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --ckpt $CKPT_PATH \
#     --a2b_mode $A2B_MODE \
#     --b2a_mode $B2A_MODE \
#     --compression_ratio $COMPRESSION_RATIO \
#     --out_dir experiments/seq2seq_tok/recon_out

# python eval_recon_humanml3d_motiongpt_metrics.py \
#   --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
#   --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
#   --data_root  $DATA_ROOT_20FPS \
#   --recon_root $SAVE_DIR_20FPS/recon_out/test \
#   --split test \
#   --t2m_path   $T2M_DIR \
#   --meta_dir   $MOTIONGPT_DIR/assets/meta \
#   --out_json   $SAVE_DIR_20FPS/recon_eval_test.json

python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
    encode \
    --split $SPLIT \
    --data_root ../HumanML3D/HumanML3D_20FPS \
    --ckpt $CKPT_PATH \
    --a2b_mode $A2B_MODE \
    --compression_ratio $COMPRESSION_RATIO \
    --out_dir experiments/HumanML3D/tokens_out_seq2seq2

