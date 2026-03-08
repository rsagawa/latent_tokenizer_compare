#!/usr/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gae50879

# #PBS -l walltime=8:00:00
# #PBS -l walltime=12:00:00
# #PBS -l walltime=24:00:00

cd ${PBS_O_WORKDIR}

source ~/.bashrc

conda activate mgpt

DATA_ROOT_20FPS=../HumanML3D/HumanML3D_20FPS
DATA_ROOT_100FPS=../HumanML3D/HumanML3D_100FPS
SAVE_DIR_20FPS=./experiments/seq2seq_tok
SAVE_DIR_100FPS=./experiments/seq2seq_tok_100FPS
MOTIONGPT_DIR=../MotionGPT_100FPS
T2M_DIR=../MotionGPT_100FPS/deps/t2m/

python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
    train \
    --train_stage ae \
    --a2b_mode tf_logits \
    --b2a_mode tf_proj \
    --data_root ../HumanML3D/HumanML3D_100FPS \
    --save_dir experiments/seq2seq_tok_100FPS \
    --max_motion_len 1000 \
    --patch_len 1 \
    --vocab_size 8192 \
    --d_model 512 \
    --enc_layers 6 \
    --dec_layers 6 \
    --recon_layers 6 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --gumbel_hard \
    --gumbel_scale 10.0 \
    --pretrain_ckpt experiments/seq2seq_tok/checkpoints_tf_logits_scale10_ce001_pos001_marg001_info001_comp4_lr1e-6/best.pt \
    --proposal_enable \
    --possharp_w 0.01 \
    --marg_w 0.01 \
    --info_w 0.01 \
    --token_ce_w 0.01 \
    --compression_ratio_min 4.0 \
    --compression_ratio_max 4.0 \
    --token_posenc_scale_with_compression \
    # --a2b_teacher_forcing_prob 0.5 \
    # --l2p_a_drop_prob 0.5 \

    # --batch_size 64 \
    # --lr 1e-4 \
    # --lr 1e-5 \

    # --a2b_mode ar \
    # --a2b_mode tf_teacher \
    # --freeze_netB \
    # --token_ce_w 1.0 \
    # --token_ce_w 0.2 \
    # --batch_size 16 \

    # --compression_ratio 1 \

    # --l2p_a_drop_mode mse_thresh \
    # --l2p_a_drop_mse_thresh 60.0 \
    # --a2b_teacher_forcing_prob 0.5 \
    # --a2b_supervised_kl_w 1.0 \

    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_scale10_ce1_pos001_marg001_info001_varcomp4/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_supervised_scale10_ce1_pos001_marg001_info001/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_tf_teacher_scale10_ce001_pos001_marg001_info001/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_tf_logits_scale10_ce001_pos001_marg001_info001/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_logits_scale10_ce001_pos001_marg001_info001_l2pdrop05/best.pt \
