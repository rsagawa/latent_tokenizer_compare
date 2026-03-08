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
MOTIONGPT_DIR=../MotionGPT_100FPS
T2M_DIR=../MotionGPT_100FPS/deps/t2m/

TOK_SAVE_DIR_20FPS=./experiments/seq2seq_tok
T2M_SAVE_DIR_20FPS=./experiments/t2m_tok

TOK_CKPT_PATH=$TOK_SAVE_DIR_20FPS/checkpoints_logits_scale10_ce001_pos001_marg001_info001_l2pdrop05/best.pt
# TOK_CKPT_PATH=$TOK_SAVE_DIR_20FPS/checkpoints_tf_logits_scale10_ce001_pos001_marg001_info001_comp4/best.pt

T2M_CKPT=$T2M_SAVE_DIR_20FPS/checkpoints/best.pt

# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     encode \
#     --split train \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --ckpt $TOK_CKPT_PATH \
#     --out_dir experiments/seq2seq_tok/tokens_out

# python motion_tokenizer_humanml3d_seq2seq_two_networks.py \
#     encode \
#     --split val \
#     --data_root ../HumanML3D/HumanML3D_20FPS \
#     --ckpt $TOK_CKPT_PATH \
#     --out_dir experiments/seq2seq_tok/tokens_out

# python train_t2m_from_motion_tokens.py train \
#     --hml_root $DATA_ROOT_20FPS \
#     --token_root $TOK_SAVE_DIR_20FPS/tokens_out \
#     --save_dir $T2M_SAVE_DIR_20FPS \
#     --motion_vocab_size 8192 \
#     --d_model 512 --n_heads 8 --enc_layers 4 --dec_layers 4 \
#     --batch_size 64 --epochs 30

python eval_t2m_from_motion_tokens_motiongpt_metrics.py \
  --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
  --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
  --meta_dir   $MOTIONGPT_DIR/assets/meta \
  --hml_root   $DATA_ROOT_20FPS \
  --split      test \
  --tokenizer_py ./motion_tokenizer_humanml3d_seq2seq_two_networks.py \
  --tokenizer_ckpt $TOK_CKPT_PATH \
  --t2m_py     ./train_t2m_from_motion_tokens.py \
  --t2m_ckpt   $T2M_CKPT \
  --t2m_save_dir $T2M_SAVE_DIR_20FPS \
  --t2m_path   $MOTIONGPT_DIR/deps/t2m/ \
  --out_json   t2m_eval.json \
  --decode_mode sample \
  --sample_top_k 10 \
  --max_text_len 196 \
