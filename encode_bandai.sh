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
# DATA_ROOT_100FPS=../HumanML3D/HumanML3D_100FPS
SAVE_DIR_20FPS=./experiments/bandai
# SAVE_DIR_100FPS=./experiments/seq2seq_tok_100FPS
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

CKPT_PATH=experiments/seq2seq_tok/checkpoints_tf_logits_scale10_ce001_pos001_marg001_info001_comp4//best.pt
A2B_MODE=tf_logits
B2A_MODE=tf_proj
# A2B_MODE=ar
# B2A_MODE=ar

# SPLIT=train
# SPLIT=val
# SPLIT=test
SPLIT=all

COMPRESSION_RATIO=4

python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
    encode \
    --split $SPLIT \
    --data_root $DATA_ROOT_20FPS \
    --ckpt $CKPT_PATH \
    --a2b_mode $A2B_MODE \
    --compression_ratio $COMPRESSION_RATIO \
    --out_dir $SAVE_DIR_20FPS/tokens_out2_proposed

    # --out_dir $SAVE_DIR_20FPS/tokens_out_proposed