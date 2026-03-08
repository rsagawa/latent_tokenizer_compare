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

# python motion_tokenizer_humanml3d_seq2seq_gaussalign.py \
#     train \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 1 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 24 \
#     --epochs 10 \
#     --lr 1e-5 \
#     --gauss_align_enable \
#     --gauss_align_w 0.1 \
#     --gauss_apply mid \
#     --gauss_sigma 6.0 \
#     --proposal_enable \
#     --possharp_w 0.0 \
#     --marg_w 0.1 \
#     --info_w 0.1 \
#     --gumbel_soft \
#     --gumbel_tau 1.0 \
#     --softptr_enable \
#     --softptr_w 0.1 \
#     --temporal_diff_w 1.0 \
#     --chunk_size 20 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_2/best.pt \
#     --grad_clip 0

    # --compression_ratio 4 \
    # --batch_size 64 \
    # --vocab_size 1024 \
    # --gauss_apply all \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_2/best.pt \

    # --train_log_interval 1

    # --possharp_w 0.1 \
    # --info_w 0.1 \
    # --rec_w 0.01 \

# python motion_tokenizer_humanml3d_seq2seq_gaussalign.py \
#     train \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 4 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 16 \
#     --epochs 10 \
#     --lr 1e-5 \
#     --gumbel_soft \
#     --gumbel_tau 1.0 \
#     --grad_clip 0

    # --vocab_size 1024 \

    # --train_log_interval 1

    # --gumbel_tau 1.0 \
    # --gumbel_scale 0.2 \
    # --latent_embed_mode "softmax" \


# python motion_tokenizer_humanml3d_seq2seq_gaussalign.py \
#     reconstruct \
#     --data_root ../HumanML3D/HumanML3D_20FPS/new_joint_vecs \
#     --ckpt experiments/seq2seq_tok/checkpoints_2/best.pt \
#     --out_dir experiments/seq2seq_tok/recon

# python eval_recon_humanml3d_motiongpt_metrics.py \
#   --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
#   --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
#   --data_root  $DATA_ROOT_20FPS \
#   --recon_root $SAVE_DIR_20FPS/recon_out_2/test \
#   --split test \
#   --t2m_path   $T2M_DIR \
#   --meta_dir   $MOTIONGPT_DIR/assets/meta \
#   --out_json   $SAVE_DIR_20FPS/recon_eval_test.json


# python motion_tokenizer_humanml3d_seq2seq_gaussalign_vposer.py \
#     train \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 1 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 16 \
#     --epochs 10 \
#     --lr 1e-5 \
#     --gauss_align_enable \
#     --gauss_align_w 0.1 \
#     --gauss_apply mid \
#     --gauss_sigma 6.0 \
#     --proposal_enable \
#     --possharp_w 0.0 \
#     --marg_w 0.1 \
#     --info_w 0.1 \
#     --gumbel_soft \
#     --gumbel_tau 1.0 \
#     --softptr_enable \
#     --softptr_w 0.1 \
#     --temporal_diff_w 0.1 \
#     --grad_clip 0 \
#     --train_log_interval 50

# python motion_tokenizer_humanml3d_seq2seq_gaussalign_vposer.py \
#     reconstruct \
#     --data_root ../HumanML3D/HumanML3D_20FPS/new_vposer_root_vecs \
#     --ckpt experiments/seq2seq_tok/checkpoints_vp1/best.pt \
#     --out_dir experiments/seq2seq_tok/recon_out

# python3 convert_rootvposer_to_humanml263.py \
#   --input experiments/seq2seq_tok/recon_out \
#   --out_dir experiments/seq2seq_tok/recon_out_263 \
#   --recursive \
#   --vposer_expr_dir ../HumanML3D/human_body_prior/train/V02_05 \
#   --body_models_root ../HumanML3D/body_models

# python eval_recon_humanml3d_motiongpt_metrics.py \
#   --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
#   --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
#   --data_root  $DATA_ROOT_20FPS \
#   --recon_root $SAVE_DIR_20FPS/recon_out_263/test \
#   --split test \
#   --t2m_path   $T2M_DIR \
#   --meta_dir   $MOTIONGPT_DIR/assets/meta \
#   --out_json   $SAVE_DIR_20FPS/recon_eval_test.json

# python3 convert_rootvposer_to_humanml263.py \
#   --input ../HumanML3D/HumanML3D_20FPS/new_vposer_root_vecs \
#   --out_dir experiments/seq2seq_tok/test_263 \
#   --recursive \
#   --vposer_expr_dir ../HumanML3D/human_body_prior/train/V02_05 \
#   --body_models_root ../HumanML3D/body_models

# python eval_recon_humanml3d_motiongpt_metrics.py \
#   --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
#   --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
#   --data_root  $DATA_ROOT_20FPS \
#   --recon_root $SAVE_DIR_20FPS/test_263/ \
#   --split test \
#   --t2m_path   $T2M_DIR \
#   --meta_dir   $MOTIONGPT_DIR/assets/meta \
#   --out_json   $SAVE_DIR_20FPS/recon_eval_test.json

# python eval_recon_humanml3d_motiongpt_metrics.py \
#   --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
#   --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
#   --data_root  $DATA_ROOT_20FPS \
#   --recon_root ../HumanML3D/HumanML3D_20FPS/new_vposer_263 \
#   --split test \
#   --t2m_path   $T2M_DIR \
#   --meta_dir   $MOTIONGPT_DIR/assets/meta \
#   --out_json   $SAVE_DIR_20FPS/recon_eval_test.json

#   --recon_root  $DATA_ROOT_20FPS/new_joint_vecs \



# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_joint \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 4 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 10 \
#     --lr 1e-4 \
#     --gumbel_soft \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_drop05/best.pt \
#     --l2p_a_drop_prob 0.8

    # --l2p_a_drop_prob 0.5

    # --train_stage a_lm \
    # --train_stage ae_freeze_a \

    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_a_lm/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_a_freeze/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_a_joint/best.pt \

    # --batch_size 32 \

# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_joint \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 4 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 10 \
#     --lr 1e-4 \
#     --gumbel_soft \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_ce_1/best.pt \
#     --l2p_a_drop_prob 0.5 \
#     --token_ce_w 0.001

    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_drop05/best.pt \

# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_joint \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 4 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 10 \
#     --lr 1e-4 \
#     --gumbel_soft \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_ce/best.pt \
#     --l2p_a_drop_prob 0.5 \
#     --token_ce_w 0.0001 \
#     --softptr_enable \
#     --softptr_w 0.1 \
#     --softptr_apply mid




# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_joint \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 4 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 20 \
#     --lr 1e-4 \
#     --gumbel_hard \
#     --gumbel_scale 10.0 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_info01_ce001_soft01_hard_scale10_msethr/last.pt \
#     --l2p_a_drop_prob 0.8 \
#     --gauss_align_enable \
#     --gauss_align_w 0.1 \
#     --gauss_sigma 6.0 \
#     --gauss_apply mid \
#     --proposal_enable \
#     --possharp_w 0.01 \
#     --marg_w 0.01 \
#     --info_w 0.1 \
#     --softptr_enable \
#     --softptr_w 0.1 \
#     --softptr_apply mid \
#     --token_ce_w 0.1 \
#     --l2p_a_drop_mode mse_thresh \
#     --l2p_a_drop_mse_thresh 60.0 \
#     --norm_gap_enable \
#     --norm_gap_w 0.1 \



    # --past_kv_recent_frames 4 \

    # --train_log_interval 1

    # --self_attn_drop_path 0.9 \


    # --gumbel_hard \
    # --gumbel_soft \
    # --gumbel_scale 5.0 \
    # --gumbel_scale 10.0 \
    # --gumbel_scale 20.0 \

    # --token_ce_w 0.1 \
    # --possharp_w 0.1 \
    # --marg_w 0.1 \
    # --info_w 0.1 \

    # --l2p_a_drop_prob 0.5 \
    # --l2p_a_drop_prob 0.7 \
    # --l2p_a_drop_prob 0.9 \

    # --softptr_enable \
    # --softptr_w 0.1 \
    # --softptr_apply mid

    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_drop05/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg_pos001_drop07/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg_pos001_soft/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg_pos001_soft_selfdrop09/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_soft_hard/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_b_lm/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_soft_hard/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg01_info01_soft01_hard_ce001_scale2/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_info01_ce001_soft01_hard_scale5/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_info01_ce001_soft01_hard_scale5/last.pt \




# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     encode \
#     --split val \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_soft_hard/last.pt \
#     --out_dir experiments/seq2seq_tok/tokens_out

    # --split train \
    # --split test \
    # --ckpt experiments/seq2seq_tok/checkpoints_ce/best.pt \

# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     reconstruct \
#     --split test \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --ckpt experiments/seq2seq_tok/checkpoints_para_supervised_scale10_ce1_pos001_marg001_info001/last.pt \
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



# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage b2a_nexttok_from_tokens \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 4 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --gumbel_hard \
#     --gumbel_scale 2.0 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_soft_hard/last.pt \
#     --l2p_a_drop_prob 0.9 \
#     --gauss_align_enable \
#     --gauss_align_w 0.1 \
#     --gauss_sigma 6.0 \
#     --gauss_apply mid \
#     --proposal_enable \
#     --possharp_w 0.01 \
#     --marg_w 0.01 \
#     --info_w 0.01 \
#     --softptr_enable \
#     --softptr_w 0.1 \
#     --softptr_apply mid \
#     --self_attn_drop_path 0.9 \
#     --token_ce_w 0.1 \
#     --token_dir experiments/seq2seq_tok/tokens_out \

    # --train_stage b2a_from_tokens \

# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_parallel \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 4 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --gumbel_hard \
#     --gumbel_scale 2.0 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_info01_ce001_soft01_hard_scale2_msethr_norm01_para/last.pt \
#     --l2p_a_drop_prob 0.8 \
#     --gauss_align_enable \
#     --gauss_align_w 0.1 \
#     --gauss_sigma 6.0 \
#     --gauss_apply mid \
#     --proposal_enable \
#     --possharp_w 0.01 \
#     --marg_w 0.1 \
#     --info_w 0.1 \
#     --softptr_enable \
#     --softptr_w 0.1 \
#     --softptr_apply mid \
#     --token_ce_w 0.01 \
#     --l2p_a_drop_mode mse_thresh \
#     --l2p_a_drop_mse_thresh 60.0 \

    # --norm_gap_enable \
    # --norm_gap_w 0.1 \


# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_joint \
#     --a2b_mode parallel \
#     --b2a_mode parallel \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 1 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --gumbel_hard \
#     --gumbel_scale 10.0 \
#     --a_mem_next_w 0.1 \
#     --token_ce_w 1.0 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_scale10_next01_ce01/best.pt \
#     --proposal_enable \
#     --possharp_w 0.01 \
#     --marg_w 0 \
#     --marg_w 0.01 \
#     --info_w 0.01 \

    # --gumbel_scale 10.0 \
    # --token_ce_w 0.1 \
    # --info_w 0.1 \

    # --latent_embed_mode "softmax" \
    # --proposal_enable \
    # --possharp_w 0.1 \
    # --proposal_h_cap 2.0 \


    # --compression_ratio 4 \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_align_marg001_pos001_info01_ce001_soft01_hard_scale2_msethr_norm01_para/last.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_scale10_next01_ce01/best.pt \
    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_scale10_next01_ce01_info01/best.pt \



# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_parallel_supervised \
#     --a2b_mode ar \
#     --b2a_mode ar \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 1 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --gumbel_hard \
#     --gumbel_scale 10.0 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_scale10_next01_ce1_pos001_marg001_info001/best.pt \
#     --proposal_enable \
#     --possharp_w 0.01 \
#     --marg_w 0 \
#     --marg_w 0.01 \
#     --info_w 0.01 \
#     --token_ce_w 1.0 \
#     --a2b_supervised_kl_w 1.0 \

    # --a2b_teacher_forcing_prob 0.7 \
    # --l2p_a_drop_mode mse_thresh \
    # --l2p_a_drop_mse_thresh 60.0 \

    # --l2p_a_drop_prob 0.8 \
    # --gauss_align_enable \
    # --gauss_align_w 0.1 \
    # --gauss_sigma 6.0 \
    # --gauss_apply mid \
    # --softptr_enable \
    # --softptr_w 0.1 \
    # --softptr_apply mid \
    # --norm_gap_enable \
    # --norm_gap_w 0.1 \



# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_parallel_supervised \
#     --a2b_mode ar \
#     --b2a_mode ar \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --compression_ratio 1 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 64 \
#     --epochs 100 \
#     --lr 1e-4 \
#     --gumbel_hard \
#     --gumbel_scale 10.0 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_supervised_scale10_ce1_pos001_marg001_info001/last.pt \
#     --proposal_enable \
#     --possharp_w 0.01 \
#     --marg_w 0 \
#     --marg_w 0.01 \
#     --info_w 0.01 \
#     --token_ce_w 1.0 \
#     --a2b_supervised_kl_w 1.0 \
#     --compression_ratio_min 1.0 \
#     --compression_ratio_max 4.0 \


# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     train \
#     --train_stage ae_joint \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --save_dir experiments/seq2seq_tok \
#     --max_motion_len 196 \
#     --patch_len 1 \
#     --vocab_size 8192 \
#     --d_model 512 \
#     --enc_layers 6 \
#     --dec_layers 6 \
#     --recon_layers 6 \
#     --batch_size 16 \
#     --epochs 20 \
#     --lr 1e-4 \
#     --gumbel_hard \
#     --gumbel_scale 10.0 \
#     --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_scale10_ce1_pos001_marg001_info001_varcomp4/last.pt \
#     --proposal_enable \
#     --possharp_w 0.01 \
#     --marg_w 0.1 \
#     --info_w 0.01 \
#     --token_ce_w 0.01 \
#     --compression_ratio_min 1.0 \
#     --compression_ratio_max 4.0 \
#     --l2p_a_drop_mode mse_thresh \
#     --l2p_a_drop_mse_thresh 60.0 \

    # --train_stage ae_joint \
    # --train_stage ae_parallel_mema_teacher \

    # --batch_size 64 \
    # --marg_w 0.01 \
    # --token_ce_w 1.0 \
    # --token_ce_w 0.01 \
    # --l2p_a_drop_prob 0.8 \

    # --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_supervised_scale10_ce1_pos001_marg001_info001_varcomp4/best.pt \
    # --a2b_mode parallel \

#    --pretrain_ckpt experiments/seq2seq_tok/checkpoints_para_supervised_scale10_ce1_pos001_marg001_info001/best.pt \

python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
    train \
    --train_stage ae \
    --a2b_mode tf_logits \
    --b2a_mode tf_proj \
    --data_root ../HumanML3D/HumanML3D_20FPS \
    --save_dir experiments/seq2seq_tok \
    --max_motion_len 196 \
    --patch_len 1 \
    --vocab_size 8192 \
    --d_model 512 \
    --enc_layers 6 \
    --dec_layers 6 \
    --recon_layers 6 \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-6 \
    --gumbel_hard \
    --gumbel_scale 10.0 \
    --pretrain_ckpt experiments/seq2seq_tok/checkpoints_logits_scale10_ce001_pos001_marg001_info001_l2pdrop05/best.pt \
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
