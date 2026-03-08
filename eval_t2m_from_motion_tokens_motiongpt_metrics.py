#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_t2m_from_motion_tokens_motiongpt_metrics.py

MotionGPT流儀 (TM2TMetrics) で Text-to-Motion (T2M) を評価するスクリプト。

想定パイプライン
----------------
(1) text -> 離散トークン列: train_t2m_from_motion_tokens.py で学習した小型モデル
(2) トークン列 -> motion(263D): motion_tokenizer_humanml3d_seq2seq_two_networks.py の B->A 復元(デコード)部分を利用
(3) 評価: MotionGPT の TM2TMetrics (Matching/R-Precision, FID, Diversity)

注意
----
* MotionGPT の実装では、TM2TMetrics に入力する motion 特徴は datamodule.renorm4t2m を通した
  "eval mean/std" 正規化空間に変換してから使います（val_t2m_forward -> renorm4t2m -> TM2TMetrics.update）。
  このスクリプトも同様に、raw(非正規化)の 263D から mean_eval/std_eval で正規化して入力します。
* Matching/R-Precision を有効にするため、cfg.TRAIN.STAGE は 'lm' を含む値に強制します
  （TM2TMetrics.text = 'lm' in cfg.TRAIN.STAGE and task=='t2m'）。

実行例
------
MOTIONGPT_DIR=/path/to/MotionGPT
HML_ROOT=/path/to/HumanML3D/HumanML3D_20FPS
TOK_CKPT=/path/to/runs/seq2seq_tok/checkpoints/last.pt
T2M_CKPT=/path/to/runs/t2m_tok/checkpoints/best.pt

python eval_t2m_from_motion_tokens_motiongpt_metrics.py \
  --cfg_assets $MOTIONGPT_DIR/configs/assets.yaml \
  --cfg        $MOTIONGPT_DIR/configs/config_h3d_stage3.yaml \
  --meta_dir   $MOTIONGPT_DIR/assets/meta \
  --hml_root   $HML_ROOT \
  --split      test \
  --tokenizer_py /path/to/motion_tokenizer_humanml3d_seq2seq_two_networks.py \
  --tokenizer_ckpt $TOK_CKPT \
  --t2m_py     /path/to/train_t2m_from_motion_tokens.py \
  --t2m_ckpt   $T2M_CKPT \
  --t2m_save_dir /path/to/runs/t2m_tok \
  --t2m_path   $MOTIONGPT_DIR/deps/t2m/ \
  --out_json   t2m_eval.json

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tqdm import tqdm

try:
    from omegaconf import OmegaConf
except Exception as e:
    raise RuntimeError("OmegaConf が必要です: pip install omegaconf") from e

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../MotionGPT_100FPS")))


# -------------------------
# Generic helpers
# -------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_py_module(py_path: Path, module_name: str):
    """Load a .py file as a module (no package install needed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass internals can access sys.modules[cls.__module__] during class creation.
    # Register before execution to avoid AttributeError when importing large training scripts.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _read_split_ids(hml_root: Path, split: str) -> List[str]:
    p = hml_root / f"{split}.txt"
    if not p.is_file():
        raise FileNotFoundError(f"split file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _safe_load_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if isinstance(arr, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected .npy, got npz: {path}")
    return np.asarray(arr)


def _pad_3d(arrs: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """Pad list of (T,D) -> (B,Tmax,D)"""
    assert len(arrs) > 0
    Tmax = max(int(a.shape[0]) for a in arrs)
    D = int(arrs[0].shape[1])
    out = np.full((len(arrs), Tmax, D), pad_value, dtype=np.float32)
    for i, a in enumerate(arrs):
        t = int(a.shape[0])
        out[i, :t, :] = a.astype(np.float32, copy=False)
    return out


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        if x.ndim == 0:
            return _to_jsonable(x.item())
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (torch.Tensor,)):
        return _to_jsonable(x.detach().cpu().numpy())
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


# -------------------------
# HumanML3D text parsing (tokens are already POS-tagged in texts/*.txt)
# -------------------------

_LINE_RE = re.compile(r"^([^#]+)#([^#]+)#([^#]+)#([^#]+)$")


def _parse_hml_line(line: str) -> Optional[Tuple[str, List[str], float, float]]:
    line = line.strip()
    if not line:
        return None
    m = _LINE_RE.match(line)
    if not m:
        # Fallback: treat as caption only
        cap = line.split("#", 1)[0].strip()
        return (cap, [], 0.0, 0.0) if cap else None
    cap = m.group(1).strip()
    tok_str = m.group(2).strip()
    t_tokens = [t for t in tok_str.split(" ") if t]
    try:
        f_tag = float(m.group(3))
        to_tag = float(m.group(4))
    except Exception:
        f_tag, to_tag = 0.0, 0.0
    if math.isnan(f_tag):
        f_tag = 0.0
    if math.isnan(to_tag):
        to_tag = 0.0
    return cap, t_tokens, f_tag, to_tag


def pick_fullmotion_text_tokens(text_path: Path) -> Tuple[str, List[str]]:
    """Prefer a (f_tag,to_tag)=(0,0) line if present; else fallback to first valid line."""
    if not text_path.is_file():
        return ("", [])
    lines = text_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    parsed = [_parse_hml_line(ln) for ln in lines]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return ("", [])
    for cap, toks, f_tag, to_tag in parsed:
        if float(f_tag) == 0.0 and float(to_tag) == 0.0 and cap:
            return cap, toks
    cap, toks, _, _ = parsed[0]
    return cap, toks


def build_word_pos_tensors(
    wvec,
    t_tokens: List[str],
    max_text_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Reproduce the logic used in Text2MotionDatasetEval: add sos/eos, crop/pad, and vectorize."""
    # dataset_t2m_eval.py does: tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
    toks = ["sos/OTHER"] + list(t_tokens) + ["eos/OTHER"]
    if len(toks) < 2:
        toks = ["sos/OTHER", "eos/OTHER"]

    sent_len = min(max(len(toks), 2), max_text_len + 2)  # include sos/eos
    if len(toks) > max_text_len + 2:
        # keep first max_text_len tokens in the middle, but preserve sos/eos positions
        toks = toks[: max_text_len + 2]
        sent_len = max_text_len + 2

    # pad to fixed length (max_text_len+2)
    pad_len = (max_text_len + 2) - len(toks)
    if pad_len > 0:
        # dataset typically uses 'unk/OTHER' as pad token to keep shape fixed
        toks = toks + ["unk/OTHER"] * pad_len

    word_embs = []
    pos_ohots = []
    for tok in toks:
        w, p = wvec[tok]  # returns (word_vec[300], pos_onehot[15])
        word_embs.append(w)
        pos_ohots.append(p)

    word_embs = np.asarray(word_embs, dtype=np.float32)     # [L,300]
    pos_ohots = np.asarray(pos_ohots, dtype=np.float32)     # [L,15]
    return word_embs, pos_ohots, int(sent_len)


# -------------------------
# Load MotionGPT config and metrics
# -------------------------

def load_motiongpt_cfg(cfg_assets_path: str, cfg_path: str):
    """Load MotionGPT OmegaConf cfg in the same style as mGPT.config.parse_args()."""
    from mGPT.config import get_module_config

    OmegaConf.register_new_resolver("eval", eval)

    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not bool(getattr(cfg_exp, "FULL_CONFIG", False)):
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    if not hasattr(cfg, "DEBUG"):
        cfg.DEBUG = False

    # Matching/R-Precision を有効化（TM2TMetrics.text 条件）
    if hasattr(cfg, "TRAIN") and hasattr(cfg.TRAIN, "STAGE"):
        cfg.TRAIN.STAGE = "lm_instruct"
    try:
        cfg.model.params.task = "t2m"
    except Exception:
        pass
    return cfg


# -------------------------
# Text->token model wrapper (train_t2m_from_motion_tokens.py)
# -------------------------

@dataclass
class T2MGenBundle:
    model: torch.nn.Module
    text_vocab: Any
    motion_vocab_size: int
    motion_pad_id: int
    motion_bos_id: int
    motion_eos_id: int


@torch.no_grad()
def load_t2m_generator(t2m_py: Path, ckpt_path: Path, save_dir: Optional[Path], device: torch.device) -> T2MGenBundle:
    mod = _load_py_module(t2m_py, "t2m_tok_mod")
    sd = torch.load(str(ckpt_path), map_location="cpu")

    meta = sd.get("meta", {})
    train_args = meta.get("args", {})

    motion_vocab_size = int(train_args.get("motion_vocab_size", 1024))
    motion_pad_id = motion_vocab_size
    motion_bos_id = motion_vocab_size + 1
    motion_eos_id = motion_vocab_size + 2
    motion_vocab_total = motion_vocab_size + 3

    # resolve save_dir (where text_vocab.json lives)
    if save_dir is None:
        save_dir = ckpt_path.parent.parent  # .../checkpoints/best.pt -> .../
    vocab_path = save_dir / "text_vocab.json"

    hf_name = str(train_args.get("hf_tokenizer", "") or "").strip()

    if vocab_path.is_file():
        text_vocab = mod.load_word_vocab(vocab_path)
        text_vocab_size = len(text_vocab.itos)
    elif hf_name:
        from transformers import AutoTokenizer  # type: ignore
        hf_tok = AutoTokenizer.from_pretrained(hf_name)

        class _HFWrap:
            def __init__(self, tok):
                self.tok = tok
                self.pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else 0
                self.bos_id = int(tok.bos_token_id) if tok.bos_token_id is not None else self.pad_id
                self.eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else self.pad_id
                self.unk_id = int(tok.unk_token_id) if tok.unk_token_id is not None else self.pad_id
                self.itos = []
                self.stoi = {}

            def encode(self, text: str, max_len: int) -> List[int]:
                enc = self.tok(text, truncation=True, max_length=max_len, add_special_tokens=True)
                return list(map(int, enc["input_ids"]))

        text_vocab = _HFWrap(hf_tok)
        text_vocab_size = int(hf_tok.vocab_size) + int(len(getattr(hf_tok, "added_tokens_encoder", {})))
    else:
        raise FileNotFoundError(f"text vocab not found: {vocab_path} (and no hf_tokenizer in checkpoint meta)")

    model = mod.Text2MotionTokens(
        text_vocab_size=int(text_vocab_size),
        motion_vocab_size=int(motion_vocab_total),
        d_model=int(train_args.get("d_model", 512)),
        n_heads=int(train_args.get("n_heads", 8)),
        d_ff=int(train_args.get("d_ff", 2048)),
        enc_layers=int(train_args.get("enc_layers", 4)),
        dec_layers=int(train_args.get("dec_layers", 4)),
        dropout=float(train_args.get("dropout", 0.1)),
        max_text_len=max(64, int(train_args.get("max_text_len", 32)) + 4),
        max_motion_len=max(2048, int(train_args.get("max_motion_tokens", 2048)) + 4),
        text_pad_id=int(getattr(text_vocab, "pad_id", 0)),
        motion_pad_id=int(motion_pad_id),
    ).to(device)

    model.load_state_dict(sd["model"], strict=True)
    model.eval()

    return T2MGenBundle(
        model=model,
        text_vocab=text_vocab,
        motion_vocab_size=motion_vocab_size,
        motion_pad_id=motion_pad_id,
        motion_bos_id=motion_bos_id,
        motion_eos_id=motion_eos_id,
    )


@torch.no_grad()
def generate_motion_tokens_greedy(
    bundle: T2MGenBundle,
    caption: str,
    max_text_len: int,
    max_len_tokens: int,
    device: torch.device,
) -> List[int]:
    """Return *raw* generated ids (including BOS/EOS ids in the generator vocab)."""
    tv = bundle.text_vocab
    text_ids = torch.tensor([tv.encode(caption, max_len=int(max_text_len))], dtype=torch.long, device=device)
    # Truncate to the model's positional-encoding length (prevents PE length mismatch)
    pe_max = int(getattr(getattr(bundle.model, "pos_text", None), "pe", torch.empty(0)).shape[0]) if hasattr(getattr(bundle.model, "pos_text", None), "pe") else int(getattr(bundle.model, "max_text_len", max_text_len))
    if pe_max > 0 and text_ids.size(1) > pe_max:
        text_ids = text_ids[:, :pe_max]
    text_pad = int(getattr(tv, "pad_id", 0))
    text_mask = (text_ids != text_pad)

    # encoder memory
    m = bundle.model
    mem = m.encoder(m.pos_text(m.text_emb(text_ids)), src_key_padding_mask=~text_mask)

    ys: List[int] = [bundle.motion_bos_id]
    for _ in range(int(max_len_tokens)):
        y = torch.tensor([ys], dtype=torch.long, device=device)
        y_mask = (y != int(bundle.motion_pad_id))
        y_emb = m.pos_motion(m.motion_emb(y))
        causal = m._causal_mask(y.size(1), device=device)
        dec = m.decoder(
            tgt=y_emb,
            memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=~y_mask,
            memory_key_padding_mask=~text_mask,
        )
        logits = m.out_proj(dec[:, -1])  # [1,V]
        next_id = int(logits.argmax(dim=-1).item())
        ys.append(next_id)
        if next_id == bundle.motion_eos_id:
            break
    return ys


@torch.no_grad()
def generate_motion_tokens_sample(
    bundle: T2MGenBundle,
    caption: str,
    max_text_len: int,
    max_len_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    sample_top_k: int = 0,
    sample_top_p: float = 1.0,
) -> List[int]:
    """Sampling decode (no argmax): temperature + optional top-k/top-p."""
    tv = bundle.text_vocab
    text_ids = torch.tensor([tv.encode(caption, max_len=int(max_text_len))], dtype=torch.long, device=device)
    # Truncate to the model's positional-encoding length (prevents PE length mismatch)
    pe_max = int(getattr(getattr(bundle.model, "pos_text", None), "pe", torch.empty(0)).shape[0]) if hasattr(getattr(bundle.model, "pos_text", None), "pe") else int(getattr(bundle.model, "max_text_len", max_text_len))
    if pe_max > 0 and text_ids.size(1) > pe_max:
        text_ids = text_ids[:, :pe_max]
    text_pad = int(getattr(tv, "pad_id", 0))
    text_mask = (text_ids != text_pad)

    m = bundle.model
    mem = m.encoder(m.pos_text(m.text_emb(text_ids)), src_key_padding_mask=~text_mask)

    temp = float(max(1e-5, temperature))
    k = int(max(0, sample_top_k))
    p = float(min(1.0, max(0.0, sample_top_p)))

    ys: List[int] = [bundle.motion_bos_id]
    for _ in range(int(max_len_tokens)):
        y = torch.tensor([ys], dtype=torch.long, device=device)
        y_mask = (y != int(bundle.motion_pad_id))
        y_emb = m.pos_motion(m.motion_emb(y))
        causal = m._causal_mask(y.size(1), device=device)
        dec = m.decoder(
            tgt=y_emb,
            memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=~y_mask,
            memory_key_padding_mask=~text_mask,
        )
        logits = m.out_proj(dec[:, -1]).squeeze(0) / temp  # [V]

        if k > 0 and k < logits.numel():
            topk_vals, topk_idx = torch.topk(logits, k=k)
            probs_topk = torch.softmax(topk_vals, dim=-1)
            sampled_local = int(torch.multinomial(probs_topk, num_samples=1).item())
            next_id = int(topk_idx[sampled_local].item())
        else:
            if p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep = cumsum <= p
                keep[0] = True
                kept_idx = sorted_idx[keep]
                kept_logits = logits[kept_idx]
                probs_kept = torch.softmax(kept_logits, dim=-1)
                sampled_local = int(torch.multinomial(probs_kept, num_samples=1).item())
                next_id = int(kept_idx[sampled_local].item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        ys.append(next_id)
        if next_id == bundle.motion_eos_id:
            break
    return ys


def strip_to_tokenizer_vocab(
    gen_ids: Sequence[int],
    vocab_size: int,
    eos_id: int,
    bos_id: int,
    pad_id: int,
) -> List[int]:
    """Map generator ids -> tokenizer ids (0..vocab_size-1)."""
    out: List[int] = []
    for i in gen_ids:
        ii = int(i)
        if ii == eos_id:
            break
        if ii in (bos_id, pad_id):
            continue
        if 0 <= ii < vocab_size:
            out.append(ii)
        # else: ignore (special ids)
    return out


# -------------------------
# Tokenizer: tokens -> 263D motion (raw space)
# -------------------------

@torch.no_grad()
def decode_tokens_to_motion_raw(
    tok_mod,
    tok_model,
    mean: np.ndarray,
    std: np.ndarray,
    token_ids: List[int],
    out_len_frames: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Decode a token-id list into a raw (de-normalized) 263D motion array.

    Returns:
      motion_raw: (T,263) float32
      T_valid:    valid length
    """
    device = next(tok_model.parameters()).device
    if len(token_ids) == 0:
        # fallback: at least 1 token to avoid empty tensors
        token_ids = [0]

    patch_len = int(getattr(tok_model, "patch_len", 1))
    feat_dim = int(getattr(tok_model, "feat_dim", 263))
    compression_ratio = float(getattr(tok_model, "compression_ratio", 1.0))
    Tb = int(getattr(tok_model, "latent_len_max", len(token_ids)))

    # target patch steps N
    if out_len_frames is not None and int(out_len_frames) > 0:
        N = int(math.ceil(float(out_len_frames) / float(patch_len)))
    else:
        # heuristic: tokens * compression_ratio -> patch steps
        N = int(max(1, round(len(token_ids) * compression_ratio)))

    # latent mask and padded token tensor
    tlen = min(len(token_ids), Tb)
    tok_pad = token_ids[:tlen] + [0] * max(0, Tb - tlen)
    token_ids_t = torch.tensor([tok_pad], dtype=torch.long, device=device)  # [1,Tb]
    latent_mask = torch.zeros((1, Tb), dtype=torch.bool, device=device)
    latent_mask[:, :tlen] = True

    # condition embedding (same form as in forward)
    ratio = torch.tensor([compression_ratio], dtype=torch.float32, device=device)
    cond_feat = torch.stack([ratio, ratio.reciprocal()], dim=-1)  # [1,2]
    cond_emb = tok_model.compression_cond_proj(cond_feat).to(dtype=next(tok_model.parameters()).dtype)  # [1,D]

    # token embedding -> memory B
    if tok_model.netB.token_embed is None:
        raise RuntimeError("token_embed is None; tokenizer kind mismatch")
    tok_emb = tok_model.netB.token_embed(token_ids_t)  # [1,Tb,D]
    tok_emb_cond = tok_emb + cond_emb.unsqueeze(1)

    memB = tok_model.netB.encode(tok_emb_cond, latent_mask, causal=True)  # [1,Tb,D]
    memB_key_padding_mask = ~latent_mask  # True=pad

    # generate patches A <- B
    x_patch_hat = tok_model.netA.generate(
        memB,
        memB_key_padding_mask,
        N,
        return_attn=False,
        return_hidden=False,
        attn_apply="all",
        recent_kv_frames=0,
    )
    # Some implementations return tuple; normalize here.
    if isinstance(x_patch_hat, (tuple, list)):
        x_patch_hat = x_patch_hat[0]

    # unpatchify to [1,T_pad,feat_dim] (normalized space)
    x_hat = tok_mod.unpatchify(x_patch_hat, patch_len, feat_dim)  # [1,T_pad,D]
    x_hat = x_hat[0].detach().cpu().to(torch.float32).numpy()

    T_pad = int(x_hat.shape[0])
    if out_len_frames is None or int(out_len_frames) <= 0:
        T_valid = T_pad
    else:
        T_valid = int(min(int(out_len_frames), T_pad))

    x_hat = x_hat[:T_valid, :].astype(np.float32, copy=False)

    # de-normalize
    x_raw = x_hat * (std[None, :] + 1e-8) + mean[None, :]
    x_raw = x_raw.astype(np.float32, copy=False)
    return x_raw, T_valid


# -------------------------
# Main evaluation
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_assets", type=str, required=True)
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--meta_dir", type=str, required=True, help="MotionGPT/assets/meta (mean_eval.npy,std_eval.npy)")

    ap.add_argument("--hml_root", type=str, required=True, help="HumanML3D root (contains new_joint_vecs/, texts/, split files)")
    ap.add_argument("--split", type=str, default="test")

    # motion tokenizer (B->A decoder)
    ap.add_argument("--tokenizer_py", type=str, required=True)
    ap.add_argument("--tokenizer_ckpt", type=str, required=True)

    # text->token model
    ap.add_argument("--t2m_py", type=str, required=True)
    ap.add_argument("--t2m_ckpt", type=str, required=True)
    ap.add_argument("--t2m_save_dir", type=str, default="", help="Directory that contains text_vocab.json (default: <ckpt>/../..)")

    # MotionGPT t2m evaluator
    ap.add_argument("--t2m_path", type=str, default="", help="Override cfg.METRIC.TM2T.t2m_path")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--R_size", type=int, default=32)
    ap.add_argument("--diversity_times", type=int, default=300)

    # generation / truncation
    ap.add_argument("--max_text_len", type=int, default=32)
    ap.add_argument("--max_len_tokens", type=int, default=256)
    ap.add_argument("--decode_mode", type=str, default="greedy", choices=["greedy", "sample"],
                    help="Token generation mode for text->motion-tokens.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (used when --decode_mode sample).")
    ap.add_argument("--sample_top_k", type=int, default=0,
                    help="Top-k sampling cutoff (0 = disabled).")
    ap.add_argument("--sample_top_p", type=float, default=1.0,
                    help="Nucleus sampling p (<=1.0, 1.0 = disabled).")
    ap.add_argument("--use_gt_length", action="store_true", help="Decode to GT motion length (frame count)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MotionGPT cfg and metrics
    cfg = load_motiongpt_cfg(args.cfg_assets, args.cfg)
    if hasattr(cfg, "DATASET") and hasattr(cfg.DATASET, "HUMANML3D"):
        cfg.DATASET.HUMANML3D.ROOT = str(Path(args.hml_root))
    if args.t2m_path:
        cfg.METRIC.TM2T.t2m_path = args.t2m_path  # TM2TMetrics がここを見る

    # Unit length used inside TM2TMetrics.get_motion_embeddings
    unit_len = int(getattr(cfg.DATASET.HUMANML3D, "UNIT_LEN", 4))

    from mGPT.metrics.t2m import TM2TMetrics  # MotionGPT env
    tm2t = TM2TMetrics(
        cfg,
        dataname="humanml3d",
        top_k=int(args.top_k),
        R_size=int(args.R_size),
        diversity_times=int(args.diversity_times),
        dist_sync_on_step=False,
    ).to(device)

    # Meta stats
    meta_dir = Path(args.meta_dir)
    mean_eval = np.load(str(meta_dir / "mean_eval.npy")).astype(np.float32)
    std_eval = np.load(str(meta_dir / "std_eval.npy")).astype(np.float32)

    # Lazy-load heavy components only when cache miss occurs.
    wvec = None
    tok_mod = None
    tok_model = None
    tok_mean = None
    tok_std = None
    t2m_bundle = None

    def ensure_wvec_loaded():
        nonlocal wvec
        if wvec is not None:
            return
        from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # MotionGPT env

        wvec_root = Path(getattr(cfg.DATASET, "WORD_VERTILIZER_PATH", "deps/glove/"))
        prefix = None
        for cand in ["our_vab", "glove"]:
            if (wvec_root / f"{cand}_data.npy").is_file():
                prefix = cand
                break
        if prefix is None:
            data_files = list(wvec_root.glob("*_data.npy"))
            if not data_files:
                raise FileNotFoundError(f"No *_data.npy found in WORD_VERTILIZER_PATH: {wvec_root}")
            prefix = data_files[0].stem.replace("_data", "")
        wvec = WordVectorizer(str(wvec_root), prefix)

    def ensure_models_loaded():
        nonlocal tok_mod, tok_model, tok_mean, tok_std, t2m_bundle
        if tok_mod is not None and tok_model is not None and tok_mean is not None and tok_std is not None and t2m_bundle is not None:
            return
        tok_py = Path(args.tokenizer_py)
        tok_mod = _load_py_module(tok_py, "motion_tok_mod")
        tok_model, tok_mean, tok_std, _tok_cfg = tok_mod.load_model(Path(args.tokenizer_ckpt), device)
        t2m_bundle = load_t2m_generator(Path(args.t2m_py), Path(args.t2m_ckpt), t2m_save_dir, device)

    # cache directory under t2m_save_dir
    t2m_save_dir = Path(args.t2m_save_dir) if args.t2m_save_dir else Path(args.t2m_ckpt).parent.parent
    cache_dir = t2m_save_dir / "eval_intermediate" / str(args.split)
    cache_dir.mkdir(parents=True, exist_ok=True)

    hml_root = Path(args.hml_root)
    ids = _read_split_ids(hml_root, args.split)

    # Adjust TM2TMetrics constraints (avoid assertion failures)
    # TM2TMetrics.compute asserts: count_seq > R_size and count_seq > diversity_times
    count_seq = len(ids)
    if count_seq <= 1:
        raise RuntimeError(f"Too few sequences in split={args.split}: {count_seq}")
    if hasattr(tm2t, "R_size") and tm2t.R_size >= count_seq:
        tm2t.R_size = max(2, count_seq - 1)
    if tm2t.diversity_times >= count_seq:
        tm2t.diversity_times = max(1, count_seq - 1)

    # Buffers for batched update
    batch_ref: List[np.ndarray] = []
    batch_rst: List[np.ndarray] = []
    batch_len_ref: List[int] = []
    batch_len_rst: List[int] = []
    batch_word: List[np.ndarray] = []
    batch_pos: List[np.ndarray] = []
    batch_textlen: List[int] = []

    used = 0
    skipped = 0
    cache_hit = 0
    cache_saved = 0

    def flush():
        nonlocal batch_ref, batch_rst, batch_len_ref, batch_len_rst, batch_word, batch_pos, batch_textlen
        if not batch_ref:
            return
        # TextEncoderBiGRUCo uses pack_padded_sequence(enforce_sorted=True),
        # so sort the whole batch by text length in descending order.
        order = np.argsort(np.asarray(batch_textlen, dtype=np.int64))[::-1].copy()

        ref_sorted = [batch_ref[i] for i in order]
        rst_sorted = [batch_rst[i] for i in order]
        len_ref_sorted = [batch_len_ref[i] for i in order]
        len_rst_sorted = [batch_len_rst[i] for i in order]
        word_sorted = [batch_word[i] for i in order]
        pos_sorted = [batch_pos[i] for i in order]
        textlen_sorted = [batch_textlen[i] for i in order]

        ref_pad = _pad_3d(ref_sorted, pad_value=0.0)
        rst_pad = _pad_3d(rst_sorted, pad_value=0.0)
        word = np.stack(word_sorted, axis=0).astype(np.float32, copy=False)  # [B,L,300]
        pos = np.stack(pos_sorted, axis=0).astype(np.float32, copy=False)    # [B,L,15]
        textlen = torch.tensor(textlen_sorted, dtype=torch.long, device=device)

        tm2t.update(
            feats_ref=torch.from_numpy(ref_pad).to(device),
            feats_rst=torch.from_numpy(rst_pad).to(device),
            lengths_ref=list(len_ref_sorted),
            lengths_rst=list(len_rst_sorted),
            word_embs=torch.from_numpy(word).to(device),
            pos_ohot=torch.from_numpy(pos).to(device),
            text_lengths=textlen,
        )

        batch_ref, batch_rst = [], []
        batch_len_ref, batch_len_rst = [], []
        batch_word, batch_pos, batch_textlen = [], [], []

    motion_dir = hml_root / "new_joint_vecs"
    text_dir = hml_root / "texts"
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"new_joint_vecs not found: {motion_dir}")
    if not text_dir.is_dir():
        raise FileNotFoundError(f"texts not found: {text_dir}")

    for mid in tqdm(ids):
        # cache-first path: if full intermediate exists, skip recomputation.
        cache_path = cache_dir / f"{mid}.json"
        cache_npz_path = cache_dir / f"{mid}.npz"
        gen_ids: List[int] = []
        tok_ids: List[int] = []
        loaded_from_cache = False
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_mode_ok = str(cached.get("decode_mode", "greedy")) == str(args.decode_mode)
                cache_len_ok = int(cached.get("max_text_len", -1)) == int(args.max_text_len) and int(cached.get("max_len_tokens", -1)) == int(args.max_len_tokens)
                cache_sample_ok = (
                    str(args.decode_mode) != "sample"
                    or (
                        float(cached.get("temperature", -1.0)) == float(args.temperature)
                        and int(cached.get("sample_top_k", -1)) == int(args.sample_top_k)
                        and float(cached.get("sample_top_p", -1.0)) == float(args.sample_top_p)
                    )
                )
                if (
                    isinstance(cached.get("tok_ids", None), list)
                    and cache_mode_ok
                    and cache_len_ok
                    and cache_sample_ok
                    and cache_npz_path.is_file()
                ):
                    tok_ids = [int(x) for x in cached["tok_ids"]]
                    gen_ids = [int(x) for x in cached.get("gen_ids", [])]
                    pack = np.load(str(cache_npz_path))
                    required = {"gt_eval", "pred_eval", "word_embs", "pos_ohot", "sent_len", "gt_len", "pred_len"}
                    if required.issubset(set(pack.files)):
                        gt_eval = np.asarray(pack["gt_eval"], dtype=np.float32)
                        pred_eval = np.asarray(pack["pred_eval"], dtype=np.float32)
                        word_embs = np.asarray(pack["word_embs"], dtype=np.float32)
                        pos_ohot = np.asarray(pack["pos_ohot"], dtype=np.float32)
                        sent_len = int(np.asarray(pack["sent_len"]).item())
                        gt_len_cached = int(np.asarray(pack["gt_len"]).item())
                        pred_len_cached = int(np.asarray(pack["pred_len"]).item())

                        batch_ref.append(gt_eval)
                        batch_rst.append(pred_eval)
                        batch_len_ref.append(gt_len_cached)
                        batch_len_rst.append(pred_len_cached)
                        batch_word.append(word_embs)
                        batch_pos.append(pos_ohot)
                        batch_textlen.append(sent_len)

                        used += 1
                        cache_hit += 1
                        loaded_from_cache = True
            except Exception:
                tok_ids = []
                gen_ids = []

        if loaded_from_cache:
            if len(batch_ref) >= int(args.batch_size):
                flush()
            continue

        # cache miss path
        gt_path = motion_dir / f"{mid}.npy"
        if not gt_path.is_file():
            skipped += 1
            continue

        gt_raw = _safe_load_npy(gt_path).astype(np.float32, copy=False)
        if gt_raw.ndim != 2 or gt_raw.shape[1] != 263:
            skipped += 1
            continue

        # length (crop to unit_len)
        gt_len_full = int(gt_raw.shape[0])
        gt_len = (gt_len_full // unit_len) * unit_len
        if gt_len <= 0:
            skipped += 1
            continue
        gt_raw = gt_raw[:gt_len, :]

        cap, toks = pick_fullmotion_text_tokens(text_dir / f"{mid}.txt")
        if not cap:
            skipped += 1
            continue

        ensure_wvec_loaded()
        word_embs, pos_ohot, sent_len = build_word_pos_tensors(wvec, toks, max_text_len=20)

        if len(tok_ids) == 0:
            ensure_models_loaded()
            if args.decode_mode == "sample":
                gen_ids = generate_motion_tokens_sample(
                    t2m_bundle,  # type: ignore[arg-type]
                    caption=cap,
                    max_text_len=int(args.max_text_len),
                    max_len_tokens=int(args.max_len_tokens),
                    device=device,
                    temperature=float(args.temperature),
                    sample_top_k=int(args.sample_top_k),
                    sample_top_p=float(args.sample_top_p),
                )
            else:
                gen_ids = generate_motion_tokens_greedy(
                    t2m_bundle,  # type: ignore[arg-type]
                    caption=cap,
                    max_text_len=int(args.max_text_len),
                    max_len_tokens=int(args.max_len_tokens),
                    device=device,
                )
            tok_ids = strip_to_tokenizer_vocab(
                gen_ids,
                vocab_size=int(t2m_bundle.motion_vocab_size),  # type: ignore[union-attr]
                eos_id=int(t2m_bundle.motion_eos_id),          # type: ignore[union-attr]
                bos_id=int(t2m_bundle.motion_bos_id),          # type: ignore[union-attr]
                pad_id=int(t2m_bundle.motion_pad_id),          # type: ignore[union-attr]
            )

        # decode tokens -> raw motion (263D)
        ensure_models_loaded()
        target_len = gt_len if bool(args.use_gt_length) else None
        pred_raw, pred_len_full = decode_tokens_to_motion_raw(
            tok_mod,    # type: ignore[arg-type]
            tok_model,  # type: ignore[arg-type]
            mean=tok_mean,  # type: ignore[arg-type]
            std=tok_std,    # type: ignore[arg-type]
            token_ids=tok_ids,
            out_len_frames=target_len,
        )

        # align predicted length like MotionGPT val_t2m_forward: min(motion_len, gt_len)
        pred_len = int(min(int(pred_len_full), int(gt_len)))
        pred_raw = pred_raw[:pred_len, :]

        # crop to unit_len
        pred_len = (pred_len // unit_len) * unit_len
        if pred_len <= 0:
            skipped += 1
            continue
        pred_raw = pred_raw[:pred_len, :]

        # MotionGPT passes renorm4t2m outputs to TM2TMetrics.
        # Since renorm4t2m(x_norm) == (x_raw - mean_eval)/std_eval, we can do it directly here.
        gt_eval = (gt_raw - mean_eval) / (std_eval + 1e-8)
        pred_eval = (pred_raw - mean_eval) / (std_eval + 1e-8)

        # Save full intermediate so future runs can evaluate from cache directly.
        cache_obj = {
            "id": mid,
            "input_motion_file": str(gt_path),
            "input_text_file": str(text_dir / f"{mid}.txt"),
            "caption": cap,
            "gen_ids": [int(x) for x in gen_ids],
            "tok_ids": [int(x) for x in tok_ids],
            "t2m_ckpt": str(Path(args.t2m_ckpt)),
            "decode_mode": str(args.decode_mode),
            "max_text_len": int(args.max_text_len),
            "max_len_tokens": int(args.max_len_tokens),
            "temperature": float(args.temperature),
            "sample_top_k": int(args.sample_top_k),
            "sample_top_p": float(args.sample_top_p),
            "npz_file": cache_npz_path.name,
        }
        cache_path.write_text(json.dumps(cache_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(
            str(cache_npz_path),
            pred_raw=pred_raw.astype(np.float32, copy=False),
            gt_eval=gt_eval.astype(np.float32, copy=False),
            pred_eval=pred_eval.astype(np.float32, copy=False),
            word_embs=word_embs.astype(np.float32, copy=False),
            pos_ohot=pos_ohot.astype(np.float32, copy=False),
            sent_len=np.asarray(int(sent_len), dtype=np.int64),
            gt_len=np.asarray(int(gt_len), dtype=np.int64),
            pred_len=np.asarray(int(pred_len), dtype=np.int64),
        )
        cache_saved += 1

        batch_ref.append(gt_eval.astype(np.float32, copy=False))
        batch_rst.append(pred_eval.astype(np.float32, copy=False))
        batch_len_ref.append(int(gt_len))
        batch_len_rst.append(int(pred_len))
        batch_word.append(word_embs)
        batch_pos.append(pos_ohot)
        batch_textlen.append(int(sent_len))

        used += 1
        if len(batch_ref) >= int(args.batch_size):
            flush()

    flush()

    # compute metrics
    if used <= tm2t.diversity_times:
        tm2t.diversity_times = max(1, used - 1)
    if hasattr(tm2t, "R_size") and used <= tm2t.R_size:
        tm2t.R_size = max(2, used - 1)

    metrics = tm2t.compute(sanity_flag=False)
    out = {
        "split": args.split,
        "n_total": len(ids),
        "used": used,
        "skipped": skipped,
        "cache_dir": str(cache_dir),
        "cache_hit": cache_hit,
        "cache_saved": cache_saved,
        "metrics": _to_jsonable(metrics),
        "unit_len": unit_len,
        "cfg": {
            "top_k": int(tm2t.top_k),
            "R_size": int(tm2t.R_size),
            "diversity_times": int(tm2t.diversity_times),
            "t2m_path": str(cfg.METRIC.TM2T.t2m_path),
            "decode_mode": str(args.decode_mode),
            "temperature": float(args.temperature),
            "sample_top_k": int(args.sample_top_k),
            "sample_top_p": float(args.sample_top_p),
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
