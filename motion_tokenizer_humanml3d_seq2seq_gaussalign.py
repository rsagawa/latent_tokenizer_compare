#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HumanML3D Motion Tokenizer (Seq2Seq / Autoregressive) + optional Gaussian alignment loss
=====================================================================================

This script is a **seq2seq (encoder-decoder)** variant of a motion discrete autoencoder.

Key differences from non-AR tokenizers
-------------------------------------
- A -> B: autoregressive (AR) token generation conditioned on encoded motion (cross-attention)
- B -> A: reconstruction decoder conditioned on token embeddings (cross-attention)

Optional alignment loss (Gaussian)
----------------------------------
You can add an A-B cross-attention alignment loss inspired by "gaussian_alignment_kl":
- Encourage cross-attention (B queries -> A keys) to form a diagonal alignment in time.

Outputs (compatible with MotionGPT-style reconstruction evaluation)
------------------------------------------------------------------
- checkpoints/: model checkpoints
- tokens/<split>/<id>.txt: generated token IDs
- recon/<split>/<id>.npy: reconstructed motion (same feature space as input motions)

Usage
-----
Train:
  python motion_tokenizer_humanml3d_seq2seq_gaussalign.py train \
    --data_root /path/to/HumanML3D/new_joint_vecs \
    --save_dir runs/seq2seq_tok \
    --max_motion_len 196 \
    --patch_len 1 \
    --compression_ratio 4 \
    --vocab_size 1024 \
    --d_model 512 \
    --enc_layers 6 --dec_layers 6 --recon_layers 6 \
    --batch_size 64 --epochs 50 --lr 2e-4 \
    --gauss_align_enable --gauss_align_w 0.1

Encode tokens:
  python motion_tokenizer_humanml3d_seq2seq_gaussalign.py encode \
    --data_root /path/to/HumanML3D/new_joint_vecs \
    --ckpt runs/seq2seq_tok/checkpoints/last.pt \
    --out_dir runs/seq2seq_tok/tokens

Reconstruct:
  python motion_tokenizer_humanml3d_seq2seq_gaussalign.py reconstruct \
    --data_root /path/to/HumanML3D/new_joint_vecs \
    --ckpt runs/seq2seq_tok/checkpoints/last.pt \
    --out_dir runs/seq2seq_tok/recon

Notes
-----
- This is a research-oriented baseline. AR token generation inside training is slower than
  parallel tokenizers. Start with small vocab/length settings.
- The alignment loss uses only cross-attention weights (decoder->encoder); it does not
  require cached KV (past_key_values).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _resolve_layer_span(num_layers: int, apply: str) -> Tuple[int, int]:
    """
    Return [L1, L2) selected by mode:
      - all : all layers
      - mid : middle one-third layers
      - last: only last layer
    """
    L = int(num_layers)
    if L <= 0:
        return 0, 0
    mode = str(apply).lower().strip()
    if mode == "all":
        return 0, L
    if mode == "last":
        return L - 1, L
    if mode == "mid":
        l1 = L // 3
        l2 = (2 * L + 2) // 3  # ceil(2L/3)
        if l2 <= l1:
            l2 = min(L, l1 + 1)
        return l1, l2
    raise ValueError(f"Unknown apply mode: {apply} (expected 'all', 'mid', or 'last')")


def _layer_selected(layer_idx: int, num_layers: int, apply: str) -> bool:
    l1, l2 = _resolve_layer_span(num_layers, apply)
    return bool(l1 <= int(layer_idx) < l2)


def _merge_attn_apply(modes: List[str]) -> str:
    mm = [str(m).lower().strip() for m in modes if m is not None]
    if len(mm) == 0:
        return "all"
    uniq = set(mm)
    if "all" in uniq:
        return "all"
    if len(uniq) == 1:
        return next(iter(uniq))
    # need multiple disjoint subsets (e.g. mid + last) -> collect all
    return "all"


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/target: [B,T,D]
    mask: [B,T] bool (True = valid)
    """
    assert pred.shape == target.shape
    assert mask.dim() == 2 and mask.shape[0] == pred.shape[0] and mask.shape[1] == pred.shape[1]
    m = mask.to(pred.dtype).unsqueeze(-1)  # [B,T,1]
    num = (m * (pred - target).pow(2)).sum()
    den = m.sum().clamp_min(eps)
    return num / den


def masked_temporal_diff_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MSE on frame-to-frame differences (first-order temporal derivative).
    pred/target: [B,T,D]
    mask: [B,T] bool (True = valid)
    """
    assert pred.shape == target.shape
    assert mask.dim() == 2 and mask.shape[0] == pred.shape[0] and mask.shape[1] == pred.shape[1]
    if pred.shape[1] <= 1:
        return pred.new_zeros(())
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    pair_mask = mask[:, 1:] & mask[:, :-1]
    return masked_mse(pred_diff, target_diff, pair_mask, eps=eps)


def masked_norm_match_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Match ||valid * pred|| and ||valid * target|| on valid frames.
    pred/target: [B,T,D]
    mask: [B,T] bool (True = valid)
    """
    assert pred.shape == target.shape
    assert mask.dim() == 2 and mask.shape[0] == pred.shape[0] and mask.shape[1] == pred.shape[1]
    valid = mask.to(pred.dtype)
    valid_u = valid.unsqueeze(-1)
    pred_norm = (valid_u * pred).norm(dim=-1)
    target_norm = (valid_u * target).detach().norm(dim=-1)
    num = ((pred_norm - target_norm).abs() * valid).sum()
    den = valid.sum().clamp_min(eps)
    return num / den


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# Token statistics (no training)
# -----------------------------

class TokenStatsAggregator:
    """Aggregate simple discrete-token statistics over a dataset split.

    This is used for validation logging (no gradients):
      - mean pmax over tokens
      - mean token entropy
      - marginal entropy / effective vocab size (Hill numbers)
      - unique ID count
      - self-transition rate
      - mean latent length
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.counts = torch.zeros(self.vocab_size, dtype=torch.long)
        self.n_tokens = 0
        self.pmax_sum = 0.0
        self.ent_sum = 0.0
        self.self_tr = 0
        self.n_pairs = 0
        self.latent_len_sum = 0
        self.n_seqs = 0

    @torch.no_grad()
    def update(self, token_logits: torch.Tensor, token_ids: torch.Tensor, latent_mask: torch.Tensor) -> None:
        """
        token_logits: [B,T,V]
        token_ids:    [B,T]
        latent_mask:  [B,T] bool (True=valid)
        """
        if token_logits is None or token_ids is None or latent_mask is None:
            return
        B, T, V = token_logits.shape
        lm = latent_mask

        # token counts (argmax IDs)
        ids_flat = token_ids[lm].detach().to(torch.long).cpu()
        if ids_flat.numel() > 0:
            bc = torch.bincount(ids_flat, minlength=self.vocab_size)
            self.counts += bc
            self.n_tokens += int(ids_flat.numel())

        # per-token entropy / pmax (from logits)
        logits_flat = token_logits[lm].detach().to(torch.float32)  # [N,V]
        if logits_flat.numel() > 0:
            p = F.softmax(logits_flat, dim=-1)
            pmax = p.max(dim=-1).values
            ent = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)
            self.pmax_sum += float(pmax.sum().cpu())
            self.ent_sum += float(ent.sum().cpu())

        # self-transition rate (within valid range)
        if T >= 2:
            valid_pairs = lm[:, 1:] & lm[:, :-1]
            if bool(valid_pairs.any().item()):
                same = (token_ids[:, 1:] == token_ids[:, :-1]) & valid_pairs
                self.self_tr += int(same.sum().detach().cpu())
                self.n_pairs += int(valid_pairs.sum().detach().cpu())

        # mean latent length
        self.latent_len_sum += int(lm.sum(dim=1).detach().cpu().sum())
        self.n_seqs += int(B)

    def finalize(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.n_tokens > 0:
            out["pmax"] = float(self.pmax_sum / max(self.n_tokens, 1))
            out["token_entropy"] = float(self.ent_sum / max(self.n_tokens, 1))

            p = self.counts.to(torch.float32)
            p = p / p.sum().clamp_min(1.0)
            pm = p.clamp_min(1e-12)
            marg_entropy = float(-(pm * pm.log()).sum().item())
            out["marg_entropy"] = marg_entropy
            out["eff_num_h1"] = float(math.exp(marg_entropy))
            out["hill2"] = float(1.0 / float((p * p).sum().clamp_min(1e-12).item()))
            out["unique_ids"] = float((self.counts > 0).sum().item())
        else:
            out["pmax"] = 0.0
            out["token_entropy"] = 0.0
            out["marg_entropy"] = 0.0
            out["eff_num_h1"] = 0.0
            out["hill2"] = 0.0
            out["unique_ids"] = 0.0

        out["self_transition_rate"] = float(self.self_tr / self.n_pairs) if self.n_pairs > 0 else 0.0
        out["mean_latent_len"] = float(self.latent_len_sum / self.n_seqs) if self.n_seqs > 0 else 0.0
        return out


# -----------------------------
# Data (HumanML3D "new_joint_vecs" style)
# -----------------------------


def _read_split_ids(list_path: Path) -> List[str]:
    ids: List[str] = []
    try:
        lines = list_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        lines = list_path.read_text(errors="ignore").splitlines()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        s = s.split()[0]
        if s.endswith(".npy"):
            s = s[:-4]
        ids.append(s)
    return ids


def list_motion_files(data_root: Path, split: str) -> List[Path]:
    """
    Try multiple common HumanML3D layouts.

    Layout A (split subdirs):
      data_root/
        train/*.npy
        test/*.npy

    Layout B (MotionGPT-like):
      data_root/
        new_joint_vecs/*.npy
        train.txt / test.txt  (IDs without extension)

    Layout C (data_root == new_joint_vecs):
      data_root/*.npy
      ../train.txt / ../test.txt  (or data_root/train.txt)
    """
    data_root = Path(data_root)
    split = str(split)

    # A) split subdir
    split_dir = data_root / split
    if split_dir.exists():
        files = sorted(split_dir.glob("*.npy"))
        if len(files) > 0:
            return files

    # B) data_root contains new_joint_vecs
    njv = data_root / "new_joint_vecs"
    if njv.exists():
        list_path = data_root / f"{split}.txt"
        if list_path.exists():
            ids = _read_split_ids(list_path)
            files = [njv / f"{i}.npy" for i in ids if (njv / f"{i}.npy").exists()]
            if len(files) > 0:
                return files
        # some datasets place split lists inside new_joint_vecs
        list_path2 = njv / f"{split}.txt"
        if list_path2.exists():
            ids = _read_split_ids(list_path2)
            files = [njv / f"{i}.npy" for i in ids if (njv / f"{i}.npy").exists()]
            if len(files) > 0:
                return files
        files = sorted(njv.glob("*.npy"))
        if len(files) > 0:
            return files

    # C) data_root itself is new_joint_vecs (or flat npy dir)
    files_here = sorted(data_root.glob("*.npy"))
    if len(files_here) > 0:
        # list in same dir or parent dir
        for lp in [data_root / f"{split}.txt", data_root.parent / f"{split}.txt"]:
            if lp.exists():
                ids = _read_split_ids(lp)
                files = [data_root / f"{i}.npy" for i in ids if (data_root / f"{i}.npy").exists()]
                if len(files) > 0:
                    return files
        return files_here

    raise FileNotFoundError(
        f"Could not find motion files for split='{split}'. "
        f"Tried: {data_root}/{split}/*.npy, {data_root}/new_joint_vecs/*.npy (+ split txt), "
        f"{data_root}/*.npy (+ split txt in dir or parent)."
    )




class HumanML3DMotionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        max_motion_len: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        random_crop: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = str(split)
        self.files = list_motion_files(self.data_root, self.split)
        self.max_motion_len = int(max_motion_len)
        self.random_crop = bool(random_crop)

        # Load one file to infer feature dim
        x0 = np.load(self.files[0])
        if x0.ndim != 2:
            raise ValueError(f"Expected [T,D] motion array, got shape {x0.shape} in {self.files[0]}")
        self.feat_dim = int(x0.shape[1])

        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        x = np.load(path).astype(np.float32)  # [T,D]
        T, D = x.shape
        maxT = self.max_motion_len

        if T >= maxT:
            if self.random_crop and self.split == "train":
                s = np.random.randint(0, T - maxT + 1)
            else:
                s = 0
            x_crop = x[s : s + maxT]
            mask = np.ones((maxT,), dtype=np.bool_)
        else:
            x_crop = np.zeros((maxT, D), dtype=np.float32)
            x_crop[:T] = x
            mask = np.zeros((maxT,), dtype=np.bool_)
            mask[:T] = True

        if self.mean is not None and self.std is not None:
            x_norm = (x_crop - self.mean) / (self.std + 1e-8)
        else:
            x_norm = x_crop

        return {
            "motion": torch.from_numpy(x_norm),  # [T,D]
            "mask": torch.from_numpy(mask),      # [T]
            "id": torch.tensor([idx], dtype=torch.long),
            "path": str(path),
        }


def compute_mean_std(files: List[Path], max_files: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over frames. (Approximate if max_files < len(files)).
    """
    if max_files > 0 and len(files) > max_files:
        files = random.sample(files, max_files)

    sum_x = None
    sum_x2 = None
    n = 0

    for p in files:
        x = np.load(p).astype(np.float64)  # [T,D]
        if x.ndim != 2:
            continue
        if sum_x is None:
            sum_x = x.sum(axis=0)
            sum_x2 = (x * x).sum(axis=0)
        else:
            sum_x += x.sum(axis=0)
            sum_x2 += (x * x).sum(axis=0)
        n += x.shape[0]

    if sum_x is None or sum_x2 is None or n == 0:
        raise RuntimeError("Failed to compute mean/std (no valid files).")

    mean = sum_x / n
    var = sum_x2 / n - mean * mean
    var = np.maximum(var, 1e-10)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# -----------------------------
# Positional encoding
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        x: [B,L,D]
        """
        L = x.size(1)
        return x + self.pe[offset : offset + L].unsqueeze(0).to(x.dtype)


# -----------------------------
# Patchify helpers
# -----------------------------

def patchify(x: torch.Tensor, mask: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x:    [B,T,D]
    mask: [B,T] bool
    Returns:
      x_pad:    [B,T_pad,D]
      mask_pad: [B,T_pad]
      patches:  [B,N,P*D]
      patch_mask:[B,N] bool
    """
    B, T, D = x.shape
    P = int(patch_len)
    if P <= 0:
        raise ValueError("patch_len must be >= 1")

    T_pad = int(math.ceil(T / P) * P)
    if T_pad != T:
        pad = T_pad - T
        x_pad = torch.cat([x, torch.zeros((B, pad, D), device=x.device, dtype=x.dtype)], dim=1)
        mask_pad = torch.cat([mask, torch.zeros((B, pad), device=mask.device, dtype=torch.bool)], dim=1)
    else:
        x_pad = x
        mask_pad = mask

    N = T_pad // P
    patches = x_pad.view(B, N, P * D)
    patch_mask = mask_pad.view(B, N, P).any(dim=-1)  # True if any frame in the patch is valid
    return x_pad, mask_pad, patches, patch_mask


def unpatchify(patches: torch.Tensor, patch_len: int, feat_dim: int) -> torch.Tensor:
    """
    patches: [B,N,P*D]
    Returns x: [B,N*P,D]
    """
    B, N, PD = patches.shape
    P = int(patch_len)
    D = int(feat_dim)
    if PD != P * D:
        raise ValueError(f"Patch dim mismatch: got {PD}, expected {P*D}")
    return patches.view(B, N * P, D)


# -----------------------------
# Transformer building blocks with attention weight access
# -----------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,                 # [B,L,D]
        memory: torch.Tensor,            # [B,S,D]
        *,
        causal: bool,
        memory_key_padding_mask: Optional[torch.Tensor],  # [B,S] True=pad
        need_cross_attn: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        L = x.size(1)
        attn_mask = None
        if causal:
            # True means "masked" for MultiheadAttention (bool mask)
            attn_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=x.device), diagonal=1)

        # self-attn
        y, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop(y))

        # cross-attn
        y, w = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_cross_attn,
            average_attn_weights=False,
        )
        x = self.norm2(x + self.drop(y))

        # ffn
        y = self.lin2(self.drop(self.act(self.lin1(x))))
        x = self.norm3(x + self.drop(y))
        return x, w  # w: [B,H,L,S] if requested


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# -----------------------------
# Gaussian alignment loss (cleaned for seq2seq cross-attention)
# -----------------------------

class GaussianAlignmentKL(nn.Module):
    """
    Encourage decoder->encoder cross-attention to align diagonally.

    attention_layers: list of [B,H,T,S]
      - T: number of latent tokens (decoder steps)
      - S: number of motion patches (encoder length)

    src_keep_mask: [B,S] bool (True=valid key)
    tgt_keep_mask: [B,T] bool (True=valid latent step)
    """
    def __init__(
        self,
        sigma: float = 0.2,
        lambda_band: float = 1e-1,
        m_target: float = 0.90,
        band_k: float = 1.0,
        lambda_off: float = 3e-2,
        beta: float = 0.0,
        com_target: float = 0.35,
        lambda_com: float = 0.0,
        apply: str = "all",   # "all" | "mid" | "last"
        weak_weight: float = 0.3,  # not used in "last"
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.lambda_band = float(lambda_band)
        self.m_target = float(m_target)
        self.band_k = float(band_k)
        self.lambda_off = float(lambda_off)
        self.beta = float(beta)
        self.com_target = float(com_target)
        self.lambda_com = float(lambda_com)
        self.apply = str(apply)
        self.weak_weight = float(weak_weight)

    def forward(
        self,
        attention_layers: List[torch.Tensor],
        *,
        src_keep_mask: Optional[torch.Tensor] = None,
        tgt_keep_mask: Optional[torch.Tensor] = None,
        sigma_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Mask-aware diagonal alignment for cross-attention.

        Notes on variable lengths:
        - In this codebase, attention tensors are padded to a fixed [T,S] over the batch,
          while `tgt_keep_mask/src_keep_mask` indicate the *effective* lengths per sample.
        - We map target steps to key steps via per-sample length ratio and define
          Gaussian distance in **absolute key-step units** (sigma is a step count),
          so sigma no longer depends on normalized [0,1] coordinates.
        """
        if attention_layers is None or len(attention_layers) == 0:
            device = src_keep_mask.device if src_keep_mask is not None else "cpu"
            return torch.zeros((), device=device)

        # choose layer weights
        L = len(attention_layers)
        first = None
        for a in attention_layers:
            if a is not None:
                first = a
                break
        if first is None:
            return torch.zeros((), device=attention_layers[0].device if attention_layers[0] is not None else "cpu")

        alpha = torch.zeros((L,), dtype=torch.float32, device=first.device)
        l1, l2 = _resolve_layer_span(L, self.apply)
        alpha[l1:l2] = 1.0
        denom = alpha.sum().clamp_min(1e-12)

        # infer sizes
        B, H, T, S = first.shape
        device = first.device
        eps = 1e-12

        if T <= 0 or S <= 0:
            return torch.zeros((), device=device)

        # masks (True = keep / valid)
        km = None
        km4 = None
        if src_keep_mask is not None:
            km = src_keep_mask.to(device=device, dtype=torch.bool)
            if km.dim() != 2 or km.size(0) != B or km.size(1) != S:
                raise ValueError(f"src_keep_mask must be [B,S]={B,S}, got {tuple(km.shape)}")
            # avoid all-masked per sample
            bad = ~km.any(dim=1, keepdim=True)
            if bad.any():
                km = km.clone()
                km[bad.expand_as(km)] = True
            km4 = km[:, None, None, :]  # [B,1,1,S]

        tm = None
        tm3 = None
        if tgt_keep_mask is not None:
            tm = tgt_keep_mask.to(device=device, dtype=torch.bool)
            if tm.dim() != 2 or tm.size(0) != B or tm.size(1) != T:
                raise ValueError(f"tgt_keep_mask must be [B,T]={B,T}, got {tuple(tm.shape)}")
            tm3 = tm[:, None, :]  # [B,1,T]

        # Build per-sample diagonal in key-step units (sigma is absolute step count)
        t_idx = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T, 1).expand(B, 1, T, 1)
        s_idx = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, 1, S).expand(B, 1, 1, S)
        if tm is not None:
            t_eff = tm.sum(dim=1).clamp_min(1).to(torch.float32)  # [B]
        else:
            t_eff = torch.full((B,), float(T), device=device, dtype=torch.float32)
        if km is not None:
            s_eff = km.sum(dim=1).clamp_min(1).to(torch.float32)  # [B]
        else:
            s_eff = torch.full((B,), float(S), device=device, dtype=torch.float32)
        den_t = (t_eff - 1.0).clamp_min(1.0).view(B, 1, 1, 1)
        den_s = (s_eff - 1.0).clamp_min(1.0).view(B, 1, 1, 1)
        j_center = t_idx * (den_s / den_t)  # expected key position on diagonal
        dist = s_idx - j_center

        # Gaussian log prior (row-normalized over S); sigma is key-step count
        sigma = max(self.sigma * float(sigma_scale), 1e-6)
        logp = -(dist * dist) / (2.0 * sigma * sigma)  # [B,1,T,S]

        # mask keys if provided, then renormalize over S
        if km4 is not None:
            logp = logp.masked_fill(~km4, -1.0e9)
        logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
        logp = logp.detach()
        pri = logp.exp().detach()

        # band mask (same shape as logp)
        band_radius = self.band_k * sigma
        bmf = (torch.abs(dist) <= band_radius).to(dtype=torch.float32)
        if km4 is not None:
            bmf = bmf * km4.to(dtype=bmf.dtype)
        j_norm = s_idx / den_s

        # reduction helper over [B,H,T]
        def masked_mean(x_bht: torch.Tensor) -> torch.Tensor:
            if tm3 is None:
                return x_bht.mean()
            m = tm3.expand(B, x_bht.size(1), T).to(dtype=x_bht.dtype)
            num = (x_bht * m).sum()
            den = m.sum().clamp_min(1.0)
            return num / den

        kl_total = 0.0
        band_total = 0.0
        off_total = 0.0
        com_total = 0.0

        for li, A in enumerate(attention_layers):
            if A is None:
                continue
            w_layer = alpha[li]
            if float(w_layer.item()) == 0.0:
                continue

            # w: [B,H,T,S]
            w = A.to(dtype=torch.float32)

            # enforce src mask then renormalize
            if km4 is not None:
                w = w * km4.to(dtype=w.dtype)
            w = w / (w.sum(dim=-1, keepdim=True) + eps)

            # KL terms (broadcast logp/pri along head dim)
            kl_w_pi = (w.clamp_min(eps).log() - logp).mul(w).sum(dim=-1)  # [B,H,T]
            kl_pi_w = (pri * (logp - w.clamp_min(eps).log())).sum(dim=-1)  # [B,1,T]
            if kl_pi_w.size(1) == 1 and w.size(1) != 1:
                kl_pi_w = kl_pi_w.expand(B, w.size(1), T)
            kl = self.beta * kl_pi_w + (1.0 - self.beta) * kl_w_pi
            kl = masked_mean(kl)

            # band mass hinge
            m_band = (w * bmf).sum(dim=-1)  # [B,H,T]
            band_hinge = torch.relu(self.m_target - m_band)
            band_hinge = masked_mean(band_hinge)

            # off-band mass
            mass_off = (1.0 - m_band)
            mass_off = masked_mean(mass_off)

            # COM (within key window) -- optional
            if self.lambda_com > 0.0:
                com = (w * j_norm).sum(dim=-1)  # [B,H,T], normalized key coordinate
                com_left = torch.relu(self.com_target - com)
                com_left = masked_mean(com_left)
            else:
                com_left = torch.zeros((), device=device)

            kl_total = kl_total + w_layer * kl
            band_total = band_total + w_layer * band_hinge
            off_total = off_total + w_layer * mass_off
            com_total = com_total + w_layer * com_left

        kl_avg = kl_total / denom
        band_avg = band_total / denom
        off_avg = off_total / denom
        com_avg = com_total / denom

        loss = kl_avg
        loss = loss + self.lambda_band * band_avg
        if self.lambda_off > 0.0:
            loss = loss + self.lambda_off * off_avg
        if self.lambda_com > 0.0:
            loss = loss + self.lambda_com * com_avg
        return loss

    def compute_band_mass(
        self,
        attention_layers: List[torch.Tensor],
        *,
        src_keep_mask: Optional[torch.Tensor] = None,
        tgt_keep_mask: Optional[torch.Tensor] = None,
        sigma_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute mean in-band attention mass using the same masks/band definition as the loss."""
        if attention_layers is None or len(attention_layers) == 0:
            device = src_keep_mask.device if src_keep_mask is not None else "cpu"
            return torch.zeros((), device=device)

        L = len(attention_layers)
        first = None
        for a in attention_layers:
            if a is not None:
                first = a
                break
        if first is None:
            return torch.zeros((), device=attention_layers[0].device if attention_layers[0] is not None else "cpu")

        alpha = torch.zeros((L,), dtype=torch.float32, device=first.device)
        l1, l2 = _resolve_layer_span(L, self.apply)
        alpha[l1:l2] = 1.0
        denom = alpha.sum().clamp_min(1e-12)

        B, _, T, S = first.shape
        device = first.device
        eps = 1e-12
        if T <= 0 or S <= 0:
            return torch.zeros((), device=device)

        km = None
        km4 = None
        if src_keep_mask is not None:
            km = src_keep_mask.to(device=device, dtype=torch.bool)
            if km.dim() != 2 or km.size(0) != B or km.size(1) != S:
                raise ValueError(f"src_keep_mask must be [B,S]={B,S}, got {tuple(km.shape)}")
            bad = ~km.any(dim=1, keepdim=True)
            if bad.any():
                km = km.clone()
                km[bad.expand_as(km)] = True
            km4 = km[:, None, None, :]

        tm = None
        tm3 = None
        if tgt_keep_mask is not None:
            tm = tgt_keep_mask.to(device=device, dtype=torch.bool)
            if tm.dim() != 2 or tm.size(0) != B or tm.size(1) != T:
                raise ValueError(f"tgt_keep_mask must be [B,T]={B,T}, got {tuple(tm.shape)}")
            tm3 = tm[:, None, :]

        t_idx = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T, 1).expand(B, 1, T, 1)
        s_idx = torch.arange(S, device=device, dtype=torch.float32).view(1, 1, 1, S).expand(B, 1, 1, S)
        if tm is not None:
            t_eff = tm.sum(dim=1).clamp_min(1).to(torch.float32)
        else:
            t_eff = torch.full((B,), float(T), device=device, dtype=torch.float32)
        if km is not None:
            s_eff = km.sum(dim=1).clamp_min(1).to(torch.float32)
        else:
            s_eff = torch.full((B,), float(S), device=device, dtype=torch.float32)
        den_t = (t_eff - 1.0).clamp_min(1.0).view(B, 1, 1, 1)
        den_s = (s_eff - 1.0).clamp_min(1.0).view(B, 1, 1, 1)
        j_center = t_idx * (den_s / den_t)
        dist = s_idx - j_center

        sigma = max(self.sigma * float(sigma_scale), 1e-6)
        band_radius = self.band_k * sigma
        bmf = (torch.abs(dist) <= band_radius).to(dtype=torch.float32)
        if km4 is not None:
            bmf = bmf * km4.to(dtype=bmf.dtype)

        def masked_mean(x_bht: torch.Tensor) -> torch.Tensor:
            if tm3 is None:
                return x_bht.mean()
            m = tm3.expand(B, x_bht.size(1), T).to(dtype=x_bht.dtype)
            num = (x_bht * m).sum()
            den = m.sum().clamp_min(1.0)
            return num / den

        band_total = 0.0
        for li, A in enumerate(attention_layers):
            if A is None:
                continue
            w_layer = alpha[li]
            if float(w_layer.item()) == 0.0:
                continue
            w = A.to(dtype=torch.float32)
            if km4 is not None:
                w = w * km4.to(dtype=w.dtype)
            w = w / (w.sum(dim=-1, keepdim=True) + eps)
            m_band = (w * bmf).sum(dim=-1)
            band_total = band_total + w_layer * masked_mean(m_band)

        return band_total / denom


# -----------------------------
# Soft pointer loss (optional)
# -----------------------------

class SoftPointerLoss(nn.Module):
    """Soft pointer reconstruction loss between a *source* sequence A and a *target* sequence B."""

    def __init__(self, D: int, d_proj: int = 128, use_bridge: bool = True) -> None:
        super().__init__()
        self.lnA = nn.LayerNorm(D, elementwise_affine=False)
        self.lnB = nn.LayerNorm(D, elementwise_affine=False)
        self.projA = nn.Linear(D, d_proj, bias=False)
        self.projB = nn.Linear(D, d_proj, bias=False)
        nn.init.orthogonal_(self.projA.weight)
        nn.init.orthogonal_(self.projB.weight)

        self.bridge_P2L = nn.Linear(d_proj, d_proj, bias=False) if use_bridge else nn.Identity()
        self.bridge_L2P = nn.Linear(d_proj, d_proj, bias=False) if use_bridge else nn.Identity()
        if use_bridge:
            if self.bridge_P2L.weight.shape[0] == self.bridge_P2L.weight.shape[1]:
                nn.init.eye_(self.bridge_P2L.weight)
            if self.bridge_L2P.weight.shape[0] == self.bridge_L2P.weight.shape[1]:
                nn.init.eye_(self.bridge_L2P.weight)

    def forward(
        self,
        attention_layers: List[torch.Tensor],
        src_hidden: torch.Tensor,
        tgt_hidden_layers: List[torch.Tensor],
        *,
        src_keep_mask: Optional[torch.Tensor] = None,
        tgt_keep_mask: Optional[torch.Tensor] = None,
        L1: int = 0,
        L2: Optional[int] = None,
        tau: float = 3.0,
        head_topk: Optional[int] = None,
        detach_w: bool = False,
        lambda_cos: float = 0.0,
        eps: float = 1e-12,
        direction: str = "p2l",  # "p2l" or "l2p"
    ) -> Tuple[torch.Tensor, Dict[str, List[Dict[str, float]]]]:
        if attention_layers is None or len(attention_layers) == 0:
            return torch.zeros((), device=src_hidden.device), {"layers": []}

        L = len(attention_layers)
        if L2 is None:
            L2 = L
        L1 = int(L1)
        L2 = int(L2)
        if not (0 <= L1 < L2 <= L):
            raise ValueError(f"layer range error: L1={L1}, L2={L2}, num_layers={L}")

        if src_hidden.dim() != 3:
            raise ValueError(f"src_hidden must be [B,S,D], got {tuple(src_hidden.shape)}")
        Bsz, S, D = src_hidden.shape

        if src_keep_mask is not None:
            km = src_keep_mask.to(device=src_hidden.device, dtype=torch.bool)
            if km.dim() != 2 or km.size(0) != Bsz or km.size(1) != S:
                raise ValueError(f"src_keep_mask must be [B,S]={Bsz,S}, got {tuple(km.shape)}")
            bad = ~km.any(dim=1, keepdim=True)
            if bad.any():
                km = km.clone()
                km[bad.expand_as(km)] = True
        else:
            km = None

        total_loss = 0.0
        denom = 0.0
        logs: List[Dict[str, float]] = []

        if direction.lower() == "p2l":
            zA_full = self.projA(self.lnA(src_hidden))
            bridge = self.bridge_P2L
            proj_tgt = lambda h: self.projB(self.lnB(h))
        elif direction.lower() == "l2p":
            zA_full = self.projB(self.lnB(src_hidden))
            bridge = self.bridge_L2P
            proj_tgt = lambda h: self.projA(self.lnA(h))
        else:
            raise ValueError(f"Unknown direction: {direction} (expected 'p2l' or 'l2p')")

        for l in range(L1, L2):
            A = attention_layers[l]
            hB = tgt_hidden_layers[l] if tgt_hidden_layers is not None and len(tgt_hidden_layers) > l else None
            if A is None or hB is None:
                continue

            if A.dim() != 4:
                raise ValueError(f"attention_layers[{l}] must be [B,H,T,S], got {tuple(A.shape)}")
            if hB.dim() != 3:
                raise ValueError(f"tgt_hidden_layers[{l}] must be [B,T,D], got {tuple(hB.shape)}")

            B2, Hh, T, S2 = A.shape
            if B2 != Bsz or S2 != S:
                raise ValueError(
                    f"shape mismatch at layer {l}: attn [B,H,T,S]={tuple(A.shape)} vs src [B,S,D]={tuple(src_hidden.shape)}"
                )

            if tgt_keep_mask is not None:
                tm = tgt_keep_mask.to(device=hB.device, dtype=torch.bool)
                if tm.dim() != 2 or tm.size(0) != Bsz or tm.size(1) != T:
                    raise ValueError(f"tgt_keep_mask must be [B,T]={Bsz,T}, got {tuple(tm.shape)}")
            else:
                tm = None

            w = A.to(torch.float32).clamp_min(eps)
            if detach_w:
                w = w.detach()
            if tau != 1.0:
                w = w ** (1.0 / max(float(tau), 1e-6))

            if km is not None:
                w = w * km[:, None, None, :].to(dtype=w.dtype)
            w = w / (w.sum(dim=-1, keepdim=True) + eps)

            if head_topk is not None and int(head_topk) > 0 and int(head_topk) < w.size(1):
                score = w.max(dim=-1).values.mean(dim=(0, 2))
                top_idx = torch.topk(score, k=int(head_topk), dim=0).indices
                w = w[:, top_idx]

            zB = proj_tgt(hB)

            w_mean = w.mean(dim=1)
            zA_tilde = torch.bmm(w_mean, zA_full)
            pred = bridge(zA_tilde)

            diff = (pred - zB).pow(2).mean(dim=-1)
            if tm is not None:
                m = tm.to(dtype=diff.dtype)
                mse = (diff * m).sum() / m.sum().clamp_min(1.0)
            else:
                mse = diff.mean()

            if lambda_cos > 0.0:
                pred_n = F.normalize(pred, dim=-1)
                zB_n = F.normalize(zB, dim=-1)
                cosv = (pred_n * zB_n).sum(dim=-1)
                if tm is not None:
                    m = tm.to(dtype=cosv.dtype)
                    cos_loss = 1.0 - (cosv * m).sum() / m.sum().clamp_min(1.0)
                else:
                    cos_loss = 1.0 - cosv.mean()
            else:
                cos_loss = pred.new_zeros(())

            loss_l = mse + float(lambda_cos) * cos_loss
            total_loss = total_loss + loss_l
            denom += 1.0

            logs.append({
                "layer": float(l),
                "mse": float(mse.detach().cpu().item()),
                "cos": float(cos_loss.detach().cpu().item()),
                "attn_max": float(w.max().detach().cpu().item()),
            })

        if denom == 0.0:
            return torch.zeros((), device=src_hidden.device), {"layers": []}

        total_loss = total_loss / denom
        return total_loss, {"layers": logs}


# -----------------------------
# Token regularizers (optional; similar idea to L_possharp/L_marg/L_info_nce)
# -----------------------------

class ProposalConstraintsSoftmax(nn.Module):
    """
    Constraints on token logits to avoid collapse:
      - L_possharp: encourage per-position sharpness (entropy <= H_cap)
      - L_marg: encourage marginal usage not to collapse (toward uniform, via EMA prior mixing)
      - L_info_nce: encourage diversity across positions using normalized probability vectors
    """
    def __init__(
        self,
        vocab_size: int,
        H_cap: float = 3.0,
        possharp_temp: float = 0.3,
        tau_ent: float = 0.7,
        tau_marg: float = 1.3,
        ema_decay: float = 0.99,
        info_nce_temperature: float = 0.2,
        bank_size: int = 4096,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.H_cap = float(H_cap)
        self.possharp_temp = float(possharp_temp)
        if self.possharp_temp <= 0.0:
            raise ValueError(f"possharp_temp must be > 0, got {self.possharp_temp}")
        self.tau_ent = float(tau_ent)
        self.tau_marg = float(tau_marg)
        self.ema_decay = float(ema_decay)
        self.info_nce_temperature = float(info_nce_temperature)

        self.register_buffer("ema_prior", torch.ones(self.vocab_size, dtype=torch.float32) / self.vocab_size)

        # a very simple "bank" (FIFO) for InfoNCE
        self.bank_size = int(bank_size)
        self.register_buffer("bank", torch.zeros(self.bank_size, self.vocab_size, dtype=torch.float32))
        self.register_buffer("bank_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("bank_full", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _update_ema(self, p_marg: torch.Tensor) -> None:
        self.ema_prior.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * p_marg)

    @torch.no_grad()
    def _bank_enqueue(self, z: torch.Tensor) -> None:
        """
        z: [N,V] normalized vectors
        """
        N = z.size(0)
        if N <= 0:
            return
        if N >= self.bank_size:
            self.bank.copy_(z[-self.bank_size:])
            self.bank_ptr.fill_(0)
            self.bank_full.fill_(True)
            return
        ptr = int(self.bank_ptr.item())
        end = ptr + N
        if end <= self.bank_size:
            self.bank[ptr:end].copy_(z)
        else:
            k1 = self.bank_size - ptr
            self.bank[ptr:].copy_(z[:k1])
            self.bank[: end - self.bank_size].copy_(z[k1:])
        self.bank_ptr.fill_(end % self.bank_size)
        if end >= self.bank_size:
            self.bank_full.fill_(True)

    def _bank_get(self) -> Optional[torch.Tensor]:
        if bool(self.bank_full.item()):
            return self.bank
        ptr = int(self.bank_ptr.item())
        if ptr <= 0:
            return None
        return self.bank[:ptr]

    def forward(
        self,
        logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        *,
        update_state: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        logits: [B,T,V]
        valid_mask: [B,T] bool (optional)
        update_state: if False, do NOT update EMA prior / InfoNCE bank (useful for validation).
        returns: (L_possharp, L_marg, L_info_nce)
        """
        assert logits.dim() == 3
        B, T, V = logits.shape
        if V != self.vocab_size:
            raise ValueError(f"Vocab mismatch: logits V={V}, expected {self.vocab_size}")

        eps = 1e-12
        p = F.softmax(logits / max(self.tau_ent, eps), dim=-1)  # [B,T,V]
        q = F.softmax(logits / max(self.tau_marg, eps), dim=-1)  # [B,T,V]

        if valid_mask is not None:
            m = valid_mask.to(dtype=p.dtype).unsqueeze(-1)  # [B,T,1]
            denom = m.sum().clamp_min(1.0)
            # entropy per position
            H = -(p * (p.clamp_min(eps).log())).sum(dim=-1)  # [B,T]
            L_pos = torch.log1p(torch.exp((H - self.H_cap) / self.possharp_temp)) * self.possharp_temp
            L_possharp = (L_pos * valid_mask.to(dtype=L_pos.dtype)).sum() / denom.squeeze(-1)

            # marginal
            p_marg = (p * m).sum(dim=(0, 1)).clamp_min(eps)
            p_marg = p_marg / p_marg.sum()
        else:
            H = -(p * (p.clamp_min(eps).log())).sum(dim=-1)
            L_possharp = (torch.log1p(torch.exp((H - self.H_cap) / self.possharp_temp)) * self.possharp_temp).mean()
            p_marg = p.mean(dim=(0, 1)).clamp_min(eps)
            p_marg = p_marg / p_marg.sum()

        # mix with EMA prior, then push toward uniform
        if update_state:
            with torch.no_grad():
                self._update_ema(p_marg)
        alpha = 0.5
        p_marg_add = (1.0 - alpha) * p_marg + alpha * self.ema_prior.detach()
        p_marg_add = p_marg_add / p_marg_add.sum()
        U = torch.ones_like(p_marg_add) / p_marg_add.numel()
        L_marg = F.kl_div(p_marg_add.log(), U, reduction="sum")

        # InfoNCE
        # build vectors z (centered + L2 normalized)
        z = q - q.mean(dim=-1, keepdim=True)
        z = F.normalize(z, dim=-1)
        if valid_mask is not None:
            z = z[valid_mask]  # [N,V]
        else:
            z = z.reshape(B * T, V)

        bank = self._bank_get()
        if z.size(0) == 0:
            # No valid positions in this chunk/batch.
            L_info = torch.zeros((), device=logits.device)
        elif bank is None or bank.numel() == 0:
            # warmup: no negatives
            L_info = torch.zeros((), device=logits.device)
        else:
            # Use an immutable snapshot to avoid autograd version bumps when the FIFO bank
            # is updated later in this same forward pass.
            bank_snapshot = bank.detach().clone()
            # cosine sim
            sim = (z @ bank_snapshot.t()) / max(self.info_nce_temperature, 1e-6)  # [N,bank]
            # positives: use self-sim to mean vector (cheap proxy)
            pos = (z * z).sum(dim=-1, keepdim=True) / max(self.info_nce_temperature, 1e-6)  # [N,1]
            logits_nce = torch.cat([pos, sim], dim=1)  # [N, 1+bank]
            labels = torch.zeros((z.size(0),), dtype=torch.long, device=logits.device)
            L_info = F.cross_entropy(logits_nce, labels)

        if update_state:
            with torch.no_grad():
                if z.numel() > 0:
                    self._bank_enqueue(z.detach().clamp(-1, 1))

        return L_possharp, L_marg, L_info


# -----------------------------
# Seq2Seq (AR) tokenizer model
# -----------------------------

class ARTokenDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.query_token = nn.Parameter(torch.zeros(self.d_model))
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=max_len)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(self.d_model)
        self.to_logits = nn.Linear(self.d_model, self.vocab_size)

    def _topk_mask(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        v, _ = torch.topk(logits, top_k, dim=-1)
        thr = v[..., -1].unsqueeze(-1)
        return logits.masked_fill(logits < thr, -1.0e9)

    def generate(
        self,
        memory: torch.Tensor,                        # [B,S,D]
        memory_key_padding_mask: Optional[torch.Tensor],  # [B,S] True=pad
        T: int,
        *,
        tau: float,
        scale: float,
        hard: bool,
        top_k: int,
        return_attn: bool,
        return_hidden: bool = False,
        deterministic: bool,
        embed_mode: str = "sample",
        attn_apply: str = "all",   # "all" | "mid" | "last" (only affects which layers return weights)
        past_token_embs: Optional[torch.Tensor] = None,  # [B,Tpast,D]
        pos_offset: int = 0,
        detach_past_on_return: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Returns:
          token_ids:   [B,T]
          token_logits:[B,T,V]
          token_embs:  [B,T,D]
          attn_layers: list[n_layers] of [B,H,T,S] (cross-attn, last query row per step stacked)
          hidden_layers: list[n_layers] of [B,T,D] (per-layer hidden for the generated token positions)

        embed_mode:
          - "sample":  (default) argmax if deterministic else (straight-through) Gumbel-Softmax
          - "softmax": do NOT discretize; use softmax(logits/tau) as weights and take expected embedding
        """
        B = memory.size(0)
        device = memory.device
        T = int(T)
        if T <= 0:
            raise ValueError("T must be >= 1")

        # previous token embeddings (without positional enc); grows to [B,t,D]
        prev = past_token_embs
        if prev is not None:
            prev = prev.to(device=device, dtype=memory.dtype)
            if prev.dim() != 3 or prev.size(0) != B or prev.size(2) != self.d_model:
                raise ValueError(f"past_token_embs must be [B,Tpast,D], got {tuple(prev.shape)}")

        out_ids: List[torch.Tensor] = []
        out_logits: List[torch.Tensor] = []
        out_embs: List[torch.Tensor] = []

        attn_rows_per_layer: List[List[torch.Tensor]] = [[] for _ in range(len(self.layers))]
        hid_rows_per_layer: Optional[List[List[torch.Tensor]]] = None
        if return_hidden:
            hid_rows_per_layer = [[] for _ in range(len(self.layers))]
        embed_mode = str(embed_mode).lower().strip()

        for t in range(T):
            # build input: [prev tokens..., query]
            q = self.query_token.view(1, 1, self.d_model).expand(B, 1, self.d_model)
            if prev is None:
                x = q  # [B,1,D]
            else:
                x = torch.cat([prev, q], dim=1)  # [B,t+1,D]

            # add positional encoding
            if prev is None:
                x = self.pos_enc(x, offset=int(pos_offset))
            else:
                x = self.pos_enc(x, offset=0)

            # decoder layers (causal self-attn)
            for li, layer in enumerate(self.layers):
                x, w = layer(
                    x, memory,
                    causal=True,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_cross_attn=bool(return_attn and _layer_selected(li, len(self.layers), attn_apply)),
                )
                if return_attn and w is not None:
                    # w: [B,H,L,S] -> take last query row (position t)
                    attn_rows_per_layer[li].append(w[:, :, -1, :])
                if hid_rows_per_layer is not None:
                    hid_rows_per_layer[li].append(x[:, -1, :])

            h_last = self.ln_out(x[:, -1, :])  # [B,D]
            logits = self.to_logits(h_last)    # [B,V]
            logits = self._topk_mask(logits, top_k)

            if embed_mode == "softmax":
                # deterministic, non-discretized latent embedding
                probs = F.softmax(logits / max(float(tau), 1e-6), dim=-1)  # [B,V]
                ids = probs.argmax(dim=-1)
                emb = probs @ self.token_embed.weight  # [B,D]
            elif embed_mode == "sample":
                if deterministic:
                    ids = logits.argmax(dim=-1)
                    onehot = F.one_hot(ids, num_classes=self.vocab_size).to(dtype=logits.dtype)
                else:
                    onehot = F.gumbel_softmax(logits, tau=max(float(tau), 1e-6), hard=hard, dim=-1)

                    # g = -torch.empty_like(logits).exponential_().log() * scale  # exponential_でExp(λ=1) 分布の乱数
                    # onehot = ((logits + g) / tau).softmax(-1)           # soft ∝ pτ

                    ids = onehot.argmax(dim=-1)
                emb = onehot @ self.token_embed.weight  # [B,D]
            else:
                raise ValueError(f"Unknown embed_mode: {embed_mode} (expected 'sample' or 'softmax')")

            out_ids.append(ids)
            out_logits.append(logits)
            out_embs.append(emb)

            # append to prev
            emb_in = emb.unsqueeze(1)  # [B,1,D]
            prev = emb_in if prev is None else torch.cat([prev, emb_in], dim=1)
            prev = prev.detach()

        token_ids = torch.stack(out_ids, dim=1)         # [B,T]
        token_logits = torch.stack(out_logits, dim=1)   # [B,T,V]
        token_embs = torch.stack(out_embs, dim=1)       # [B,T,D]

        attn_layers: List[torch.Tensor] = []
        if return_attn:
            for li in range(len(self.layers)):
                if len(attn_rows_per_layer[li]) == 0:
                    attn_layers.append(None)
                else:
                    attn_layers.append(torch.stack(attn_rows_per_layer[li], dim=2))  # [B,H,T,S]

        hidden_layers: List[torch.Tensor] = []
        if hid_rows_per_layer is not None:
            for li in range(len(self.layers)):
                if len(hid_rows_per_layer[li]) == 0:
                    hidden_layers.append(None)
                else:
                    hidden_layers.append(torch.stack(hid_rows_per_layer[li], dim=1))  # [B,T,D]

        past_out = prev
        if detach_past_on_return and past_out is not None:
            past_out = past_out.detach()
        return token_ids, token_logits, token_embs, attn_layers, hidden_layers, past_out


class ReconDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.start_token = nn.Parameter(torch.zeros(self.d_model))
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(self.d_model)

    def forward(
        self,
        memory: torch.Tensor,                              # [B,Tb,D]
        memory_key_padding_mask: Optional[torch.Tensor],    # [B,Tb] True=pad
        out_len: int,
        *,
        return_attn: bool = False,
        return_hidden: bool = False,
        attn_apply: str = "all",   # "all" | "mid" | "last"
        past_inputs: Optional[Any] = None,  # legacy tensor or cache dict
        pos_offset: int = 0,
        detach_past_on_return: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], Any]:
        """
        Returns:
          y: [B,out_len,D]
          cross_attn_layers: list of per-layer cross-attention weights [B,H,out_len,Tb] (or empty if return_attn=False)
        """
        B = memory.size(0)
        L = int(out_len)

        # past_inputs accepts:
        #   - legacy tensor [B,1+Lpast,D]
        #   - cache dict {"token_inputs":[B,1+Lpast,D], "layer_inputs": list[n_layers] of [B,1+Lpast,D]}
        layer_input_cache: Optional[List[torch.Tensor]] = None
        use_abs_offset = True
        if past_inputs is None:
            prev = self.start_token.view(1, 1, self.d_model).expand(B, 1, self.d_model)  # [B,1,D]
        elif isinstance(past_inputs, dict):
            if "token_inputs" not in past_inputs:
                raise ValueError("past_inputs dict must contain 'token_inputs'.")
            prev = past_inputs["token_inputs"].to(device=memory.device, dtype=memory.dtype)
            layer_input_cache = past_inputs.get("layer_inputs", None)
            use_abs_offset = False
        else:
            prev = past_inputs.to(device=memory.device, dtype=memory.dtype)
            use_abs_offset = False
        if prev.dim() != 3 or prev.size(0) != B or prev.size(2) != self.d_model:
            raise ValueError(f"past_inputs/token_inputs must be [B,1+Lpast,D], got {tuple(prev.shape)}")
        if layer_input_cache is not None:
            if len(layer_input_cache) != len(self.layers):
                raise ValueError("past_inputs['layer_inputs'] length mismatch with num decoder layers.")
            for li, st in enumerate(layer_input_cache):
                if st.dim() != 3 or st.size(0) != B or st.size(2) != self.d_model:
                    raise ValueError(f"layer_inputs[{li}] must be [B,L,D], got {tuple(st.shape)}")

        hid_rows: List[torch.Tensor] = []
        attn_rows_per_layer: List[List[torch.Tensor]] = [[] for _ in range(len(self.layers))]
        hid_rows_per_layer: Optional[List[List[torch.Tensor]]] = None
        if return_hidden:
            hid_rows_per_layer = [[] for _ in range(len(self.layers))]
        last_layer_inputs_full: Optional[List[torch.Tensor]] = None

        for _t in range(L):
            # Fast path: use layer-wise cached inputs and compute only current token.
            if layer_input_cache is not None:
                pos_base = int(pos_offset) if use_abs_offset else 0
                cur_pos = pos_base + (prev.size(1) - 1)
                x = self.pos_enc(prev[:, -1:, :], offset=int(cur_pos))  # [B,1,D]
                next_layer_cache: List[torch.Tensor] = []
                for li, layer in enumerate(self.layers):
                    need_attn = bool(return_attn and _layer_selected(li, len(self.layers), attn_apply))
                    x_in = x
                    past_in = layer_input_cache[li]
                    kv = torch.cat([past_in, x_in], dim=1)
                    y, _ = layer.self_attn(
                        x_in, kv, kv,
                        attn_mask=None,
                        need_weights=False,
                    )
                    x = layer.norm1(x_in + layer.drop(y))
                    y, w_cross = layer.cross_attn(
                        x, memory, memory,
                        key_padding_mask=memory_key_padding_mask,
                        need_weights=need_attn,
                        average_attn_weights=False,
                    )
                    x = layer.norm2(x + layer.drop(y))
                    y = layer.lin2(layer.drop(layer.act(layer.lin1(x))))
                    x = layer.norm3(x + layer.drop(y))
                    if need_attn and w_cross is not None:
                        attn_rows_per_layer[li].append(w_cross[:, :, 0, :])
                    if hid_rows_per_layer is not None:
                        hid_rows_per_layer[li].append(x[:, 0, :])
                    next_layer_cache.append(torch.cat([past_in, x_in], dim=1))
                layer_input_cache = next_layer_cache
                h_last = self.ln_out(x[:, 0, :])  # [B,D]
            else:
                # Fallback: legacy full recomputation path (used on first chunk).
                pos_base = int(pos_offset) if use_abs_offset else 0
                x = self.pos_enc(prev, offset=pos_base)  # [B,t+1,D]
                if _t == (L - 1):
                    last_layer_inputs_full = [x.new_zeros(x.shape) for _ in range(len(self.layers))]
                for li, layer in enumerate(self.layers):
                    need_attn = bool(return_attn and _layer_selected(li, len(self.layers), attn_apply))
                    if _t == (L - 1):
                        assert last_layer_inputs_full is not None
                        last_layer_inputs_full[li] = x
                    x, w_cross = layer(
                        x, memory,
                        causal=True,
                        memory_key_padding_mask=memory_key_padding_mask,
                        need_cross_attn=need_attn,
                    )
                    if need_attn and w_cross is not None:
                        attn_rows_per_layer[li].append(w_cross[:, :, -1, :])
                    if hid_rows_per_layer is not None:
                        hid_rows_per_layer[li].append(x[:, -1, :])
                h_last = self.ln_out(x[:, -1, :])  # [B,D]

            hid_rows.append(h_last)
            prev = torch.cat([prev, h_last.unsqueeze(1)], dim=1)

        y = torch.stack(hid_rows, dim=1)  # [B,L,D]
        if layer_input_cache is None:
            layer_input_cache = last_layer_inputs_full

        cross_attn_layers: List[Optional[torch.Tensor]] = []
        if return_attn:
            for li in range(len(self.layers)):
                if len(attn_rows_per_layer[li]) == 0:
                    cross_attn_layers.append(None)
                else:
                    cross_attn_layers.append(torch.stack(attn_rows_per_layer[li], dim=2))  # [B,H,L,Tb]
        hidden_layers: List[Optional[torch.Tensor]] = []
        if hid_rows_per_layer is not None:
            for li in range(len(self.layers)):
                if len(hid_rows_per_layer[li]) == 0:
                    hidden_layers.append(None)
                else:
                    hidden_layers.append(torch.stack(hid_rows_per_layer[li], dim=1))  # [B,L,D]
        past_out: Dict[str, Any] = {"token_inputs": prev, "layer_inputs": layer_input_cache}
        if detach_past_on_return:
            past_out["token_inputs"] = past_out["token_inputs"].detach()
            if past_out["layer_inputs"] is not None:
                past_out["layer_inputs"] = [st.detach() for st in past_out["layer_inputs"]]
        return y, cross_attn_layers, hidden_layers, past_out


class MotionSeq2SeqARAE(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        patch_len: int,
        vocab_size: int,
        compression_ratio: float,
        latent_len_max: int,
        d_model: int,
        n_heads: int,
        enc_layers: int,
        dec_layers: int,
        recon_layers: int,
        d_ff: int,
        dropout: float,
        max_motion_len: int,
        gauss_cfg: Optional[Dict] = None,
        softptr_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.patch_len = int(patch_len)
        self.vocab_size = int(vocab_size)
        self.compression_ratio = float(compression_ratio)
        self.latent_len_max = int(latent_len_max)
        self.d_model = int(d_model)
        self.max_motion_len = int(max_motion_len)

        # patch projection
        self.in_proj = nn.Linear(self.patch_len * self.feat_dim, self.d_model)
        self.enc_pos = SinusoidalPositionalEncoding(self.d_model, max_len=8192)
        self.encoder = TransformerEncoder(self.d_model, n_heads, enc_layers, d_ff, dropout)

        # A -> B (AR)
        self.token_decoder = ARTokenDecoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=dec_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=8192,
        )

        # B token sequence encoder (for B -> A conditioning)
        self.latent_pos = SinusoidalPositionalEncoding(self.d_model, max_len=8192)
        self.latent_encoder = TransformerEncoder(self.d_model, n_heads, enc_layers, d_ff, dropout)

        # B -> A (recon)
        self.recon_decoder = ReconDecoder(
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=recon_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=8192,
        )
        self.out_proj = nn.Linear(self.d_model, self.patch_len * self.feat_dim)

        # alignment metrics/loss helper:
        # - always keep one instance for band_mass computation
        # - gauss_align loss itself is enabled only when requested
        gauss_cfg_eff = dict(gauss_cfg or {})
        self.gauss_align_metric = GaussianAlignmentKL(
            sigma=gauss_cfg_eff.get("sigma", 0.2),
            lambda_band=gauss_cfg_eff.get("lambda_band", 1e-1),
            m_target=gauss_cfg_eff.get("m_target", 0.90),
            band_k=gauss_cfg_eff.get("band_k", 1.0),
            lambda_off=gauss_cfg_eff.get("lambda_off", 3e-2),
            beta=gauss_cfg_eff.get("beta", 0.0),
            lambda_com=gauss_cfg_eff.get("lambda_com", 0.0),
            com_target=gauss_cfg_eff.get("com_target", 0.35),
            apply=gauss_cfg_eff.get("apply", "all"),
        )

        # optional alignment loss
        self.gauss_align: Optional[GaussianAlignmentKL] = None
        if gauss_cfg_eff.get("enable", False):
            self.gauss_align = self.gauss_align_metric

        self.softptr: Optional[SoftPointerLoss] = None
        self.softptr_cfg: Optional[Dict] = None
        if softptr_cfg is not None and bool(softptr_cfg.get("enable", False)):
            self.softptr = SoftPointerLoss(
                D=self.d_model,
                d_proj=int(softptr_cfg.get("d_proj", 128)),
                use_bridge=bool(softptr_cfg.get("use_bridge", True)),
            )
            self.softptr_cfg = softptr_cfg

    def encode_motion(self, x_patch: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """
        x_patch: [B,N,PD], patch_mask: [B,N] bool
        """
        h = self.in_proj(x_patch)
        h = self.enc_pos(h)
        src_key_padding_mask = ~patch_mask  # True=pad
        mem = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        return mem

    def compute_latent_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        """
        patch_mask: [B,N] bool
        returns latent_mask: [B,Tb] bool
        """
        B, N = patch_mask.shape
        Tb = int(self.latent_len_max)
        ratio = max(self.compression_ratio, 1e-6)
        valid_patches = patch_mask.sum(dim=1)  # [B]
        valid_latent = torch.ceil(valid_patches.to(torch.float32) / ratio).to(torch.long)
        valid_latent = valid_latent.clamp(min=1, max=Tb)
        t = torch.arange(Tb, device=patch_mask.device).unsqueeze(0).expand(B, Tb)
        latent_mask = t < valid_latent.unsqueeze(1)
        return latent_mask

    def encode_tokens(self, token_embs: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
        """
        token_embs: [B,Tb,D], latent_mask: [B,Tb] bool
        """
        h = self.latent_pos(token_embs)
        src_key_padding_mask = ~latent_mask  # True=pad
        mem = self.latent_encoder(h, src_key_padding_mask=src_key_padding_mask)
        return mem

    def forward(
        self,
        x: torch.Tensor,          # [B,T,D] normalized
        mask: torch.Tensor,       # [B,T] bool
        *,
        tau: float,
        scale: float,
        hard: bool,
        top_k: int,
        deterministic_tokens: bool,
        latent_embed_mode: str = "sample",
        return_attn: bool,
        chunk_size: int = 32,
        detach_chunk_pkv: bool = True,
        chunk_logit_consistency_enable: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          x_hat: [B,T_pad,D] normalized
          mask_pad: [B,T_pad] bool
          token_ids: [B,Tb]
          token_logits: [B,Tb,V]
          latent_mask: [B,Tb] bool
          patch_mask: [B,N] bool
          gauss_align_loss: scalar (if enabled and return_attn)
          softptr_loss: scalar (if enabled; training only)
        """
        x_pad, mask_pad, x_patch, patch_mask = patchify(x, mask, self.patch_len)
        B, N, PD = x_patch.shape

        # encoder (A)
        memA = self.encode_motion(x_patch, patch_mask)  # [B,N,D]
        src_key_padding_mask = ~patch_mask

        # latent length (max) and masks
        Tb = int(self.latent_len_max)
        latent_mask = self.compute_latent_mask(patch_mask)  # [B,Tb]
        memA_keep_mask = patch_mask  # True=valid keys

        # A -> B (AR tokens)
        need_softptr = (self.softptr is not None) and self.training
        need_attn_tok = bool(return_attn) or bool(need_softptr)
        gauss_apply = str(self.gauss_align_metric.apply).lower().strip()
        softptr_apply = str((self.softptr_cfg or {}).get("apply", "all")).lower().strip()
        attn_apply_modes: List[str] = []
        if bool(return_attn):
            attn_apply_modes.append(gauss_apply)
        if bool(need_softptr):
            attn_apply_modes.append(softptr_apply)
        attn_apply = _merge_attn_apply(attn_apply_modes)

        ch = int(chunk_size)
        if ch <= 0:
            ch = int(Tb)

        tok_ids_chunks: List[torch.Tensor] = []
        tok_logits_chunks: List[torch.Tensor] = []
        tok_emb_chunks: List[torch.Tensor] = []
        tok_attn_rows_per_layer: Optional[List[List[torch.Tensor]]] = None
        tok_hid_rows_per_layer: Optional[List[List[torch.Tensor]]] = None
        if need_attn_tok:
            tok_attn_rows_per_layer = [[] for _ in range(len(self.token_decoder.layers))]
        if need_softptr:
            tok_hid_rows_per_layer = [[] for _ in range(len(self.token_decoder.layers))]
        token_past: Optional[torch.Tensor] = None
        loss_chunk_logit_cons = torch.zeros((), device=x.device)
        n_chunk_logit_cons = 0

        for t0 in range(0, Tb, ch):
            tlen = min(ch, Tb - t0)
            ids_c, logits_c, embs_c, attn_c, hid_c, token_past = self.token_decoder.generate(
                memA,
                src_key_padding_mask,
                tlen,
                tau=tau,
                scale=scale,
                hard=hard,
                top_k=top_k,
                return_attn=need_attn_tok,
                return_hidden=need_softptr,
                deterministic=deterministic_tokens,
                embed_mode=latent_embed_mode,
                attn_apply=attn_apply,
                past_token_embs=token_past,
                pos_offset=t0,
                detach_past_on_return=detach_chunk_pkv,
            )
            tok_ids_chunks.append(ids_c)
            tok_logits_chunks.append(logits_c)
            tok_emb_chunks.append(embs_c)

            if tok_attn_rows_per_layer is not None:
                for li in range(len(tok_attn_rows_per_layer)):
                    if attn_c[li] is not None:
                        tok_attn_rows_per_layer[li].append(attn_c[li])
            if tok_hid_rows_per_layer is not None:
                for li in range(len(tok_hid_rows_per_layer)):
                    if hid_c[li] is not None:
                        tok_hid_rows_per_layer[li].append(hid_c[li])

            if chunk_logit_consistency_enable:
                with torch.no_grad():
                    _, logits_np, _, _, _, _ = self.token_decoder.generate(
                        memA,
                        src_key_padding_mask,
                        tlen,
                        tau=tau,
                        scale=scale,
                        hard=hard,
                        top_k=top_k,
                        return_attn=False,
                        return_hidden=False,
                        deterministic=True,
                        embed_mode=latent_embed_mode,
                        attn_apply=attn_apply,
                        past_token_embs=None,
                        pos_offset=t0,
                        detach_past_on_return=False,
                    )
                vm = latent_mask[:, t0 : t0 + tlen].to(logits_c.dtype).unsqueeze(-1)  # [B,t,1]
                den = vm.sum().clamp_min(1.0) * float(self.vocab_size)
                loss_chunk_logit_cons = loss_chunk_logit_cons + (((logits_c - logits_np) ** 2) * vm).sum() / den
                n_chunk_logit_cons += 1

        token_ids = torch.cat(tok_ids_chunks, dim=1)
        token_logits = torch.cat(tok_logits_chunks, dim=1)
        token_embs = torch.cat(tok_emb_chunks, dim=1)

        attn_layers: List[Optional[torch.Tensor]] = []
        if need_attn_tok:
            assert tok_attn_rows_per_layer is not None
            for li in range(len(tok_attn_rows_per_layer)):
                if len(tok_attn_rows_per_layer[li]) == 0:
                    attn_layers.append(None)
                else:
                    attn_layers.append(torch.cat(tok_attn_rows_per_layer[li], dim=2))
        hidden_layers: List[Optional[torch.Tensor]] = []
        if need_softptr:
            assert tok_hid_rows_per_layer is not None
            for li in range(len(tok_hid_rows_per_layer)):
                if len(tok_hid_rows_per_layer[li]) == 0:
                    hidden_layers.append(None)
                else:
                    hidden_layers.append(torch.cat(tok_hid_rows_per_layer[li], dim=1))
        if n_chunk_logit_cons > 0:
            loss_chunk_logit_cons = loss_chunk_logit_cons / float(n_chunk_logit_cons)

        # B -> A (reconstruct patches)
        memB = self.encode_tokens(token_embs, latent_mask)  # [B,Tb,D]
        memB_key_padding_mask = ~latent_mask  # True=pad
        sp_cfg = self.softptr_cfg or {}
        softptr_l2p_enable = bool(sp_cfg.get("l2p_enable", True))
        need_softptr_l2p = bool(need_softptr and softptr_l2p_enable)
        need_attn_recon = bool(return_attn) or bool(need_softptr_l2p)
        rec_chunks: List[torch.Tensor] = []
        rec_attn_rows_per_layer: Optional[List[List[torch.Tensor]]] = None
        rec_hid_rows_per_layer: Optional[List[List[torch.Tensor]]] = None
        if need_attn_recon:
            rec_attn_rows_per_layer = [[] for _ in range(len(self.recon_decoder.layers))]
        if need_softptr_l2p:
            rec_hid_rows_per_layer = [[] for _ in range(len(self.recon_decoder.layers))]
        recon_past: Optional[torch.Tensor] = None
        for n0 in range(0, N, ch):
            nlen = min(ch, N - n0)
            h_c, attn_c, hid_c, recon_past = self.recon_decoder(
                memB,
                memB_key_padding_mask,
                out_len=nlen,
                return_attn=need_attn_recon,
                return_hidden=need_softptr_l2p,
                attn_apply=attn_apply,
                past_inputs=recon_past,
                pos_offset=n0,
                detach_past_on_return=detach_chunk_pkv,
            )
            rec_chunks.append(h_c)
            if rec_attn_rows_per_layer is not None:
                for li in range(len(rec_attn_rows_per_layer)):
                    if attn_c[li] is not None:
                        rec_attn_rows_per_layer[li].append(attn_c[li])
            if rec_hid_rows_per_layer is not None:
                for li in range(len(rec_hid_rows_per_layer)):
                    if hid_c[li] is not None:
                        rec_hid_rows_per_layer[li].append(hid_c[li])

        h_out = torch.cat(rec_chunks, dim=1)
        attn_layers_recon: List[Optional[torch.Tensor]] = []
        if need_attn_recon:
            assert rec_attn_rows_per_layer is not None
            for li in range(len(rec_attn_rows_per_layer)):
                if len(rec_attn_rows_per_layer[li]) == 0:
                    attn_layers_recon.append(None)
                else:
                    attn_layers_recon.append(torch.cat(rec_attn_rows_per_layer[li], dim=2))
        hidden_layers_recon: List[Optional[torch.Tensor]] = []
        if need_softptr_l2p:
            assert rec_hid_rows_per_layer is not None
            for li in range(len(rec_hid_rows_per_layer)):
                if len(rec_hid_rows_per_layer[li]) == 0:
                    hidden_layers_recon.append(None)
                else:
                    hidden_layers_recon.append(torch.cat(rec_hid_rows_per_layer[li], dim=1))
        x_patch_hat = self.out_proj(h_out)  # [B,N,PD]
        x_hat = unpatchify(x_patch_hat, self.patch_len, self.feat_dim)  # [B,T_pad,D]

        out: Dict[str, torch.Tensor] = {
            "x_hat": x_hat,
            "mask_pad": mask_pad,
            "token_ids": token_ids,
            "token_logits": token_logits,
            "latent_mask": latent_mask,
            "patch_mask": patch_mask,
            "chunk_logit_consistency_loss": loss_chunk_logit_cons,
        }

        if return_attn:
            l2p_sigma_scale = 1.0 / max(self.compression_ratio, 1e-6)
            # A -> B (token decoder): (B queries) -> (A keys)
            attn_list_p2l = attn_layers
            if self.gauss_align is None:
                loss_p2l = torch.zeros((), device=x.device)
            else:
                loss_p2l = self.gauss_align(
                    attn_list_p2l,
                    src_keep_mask=memA_keep_mask,   # keys = A(patches)
                    tgt_keep_mask=latent_mask,      # queries = B(tokens)
                )

            # B -> A (reconstruction decoder): (A queries) -> (B keys)
            attn_list_l2p = attn_layers_recon
            if self.gauss_align is None:
                loss_l2p = torch.zeros((), device=x.device)
            else:
                loss_l2p = self.gauss_align(
                    attn_list_l2p,
                    src_keep_mask=latent_mask,      # keys = B(tokens)
                    tgt_keep_mask=patch_mask,       # queries = A(patches)
                    sigma_scale=l2p_sigma_scale,
                )

            # expose both directions + a combined scalar (keeps old training loop simple)
            out["gauss_align_loss_p2l"] = loss_p2l
            out["gauss_align_loss_l2p"] = loss_l2p
            out["gauss_align_loss"] = 0.5 * (loss_p2l + loss_l2p)
            out["band_mass_p2l"] = self.gauss_align_metric.compute_band_mass(
                attn_list_p2l,
                src_keep_mask=memA_keep_mask,
                tgt_keep_mask=latent_mask,
            )
            out["band_mass_l2p"] = self.gauss_align_metric.compute_band_mass(
                attn_list_l2p,
                src_keep_mask=latent_mask,
                tgt_keep_mask=patch_mask,
                sigma_scale=l2p_sigma_scale,
            )
        else:
            z = torch.zeros((), device=x.device)
            out["gauss_align_loss_p2l"] = z
            out["gauss_align_loss_l2p"] = z
            out["gauss_align_loss"] = z
            out["band_mass_p2l"] = z
            out["band_mass_l2p"] = z

        if self.softptr is not None and need_softptr:
            sp_apply = str(sp_cfg.get("apply", "all")).lower().strip()
            if sp_apply == "all":
                L1 = int(sp_cfg.get("L1", 0))
                L2 = int(sp_cfg.get("L2", -1))
                if L2 <= 0:
                    L2 = len(attn_layers)
                L2 = min(L2, len(attn_layers))
                L1 = max(0, min(L1, max(L2 - 1, 0)))
            else:
                L1, L2 = _resolve_layer_span(len(attn_layers), sp_apply)

            head_topk = int(sp_cfg.get("head_topk", 0))
            head_topk_opt = None if head_topk <= 0 else head_topk

            tau_sp = float(sp_cfg.get("tau", 3.0))
            detach_w = bool(sp_cfg.get("detach_w", False))
            lambda_cos = float(sp_cfg.get("lambda_cos", 0.0))

            loss_sp_p2l, _logs = self.softptr(
                attn_layers,
                memA,
                hidden_layers,
                src_keep_mask=memA_keep_mask,
                tgt_keep_mask=latent_mask,
                L1=L1,
                L2=L2,
                tau=tau_sp,
                head_topk=head_topk_opt,
                detach_w=detach_w,
                lambda_cos=lambda_cos,
                direction="p2l",
            )
            if need_softptr_l2p:
                if sp_apply == "all":
                    L2_recon = min(L2, len(attn_layers_recon))
                    L1_recon = max(0, min(L1, max(L2_recon - 1, 0)))
                else:
                    L1_recon, L2_recon = _resolve_layer_span(len(attn_layers_recon), sp_apply)
                if L2_recon > L1_recon:
                    loss_sp_l2p, _ = self.softptr(
                        attn_layers_recon,
                        memB,
                        hidden_layers_recon,
                        src_keep_mask=latent_mask,
                        tgt_keep_mask=patch_mask,
                        L1=L1_recon,
                        L2=L2_recon,
                        tau=tau_sp,
                        head_topk=head_topk_opt,
                        detach_w=detach_w,
                        lambda_cos=lambda_cos,
                        direction="l2p",
                    )
                else:
                    loss_sp_l2p = torch.zeros((), device=x.device)
            else:
                loss_sp_l2p = torch.zeros((), device=x.device)

            out["softptr_loss_p2l"] = loss_sp_p2l
            out["softptr_loss_l2p"] = loss_sp_l2p
            out["softptr_loss"] = 0.5 * (loss_sp_p2l + loss_sp_l2p) if need_softptr_l2p else loss_sp_p2l
        else:
            out["softptr_loss"] = torch.zeros((), device=x.device)
            out["softptr_loss_p2l"] = torch.zeros((), device=x.device)
            out["softptr_loss_l2p"] = torch.zeros((), device=x.device)

        return out


# -----------------------------
# Train / Encode / Reconstruct
# -----------------------------

@dataclass
class TrainConfig:
    data_root: str
    save_dir: str
    max_motion_len: int = 196
    patch_len: int = 1
    compression_ratio: float = 4.0
    latent_len: int = 0  # 0 => derive from (T_pad/patch_len)/compression_ratio

    vocab_size: int = 1024
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    enc_layers: int = 6
    dec_layers: int = 6
    recon_layers: int = 6

    # AR sampling
    gumbel_tau: float = 0.5
    gumbel_scale: float = 1.0
    gumbel_hard: bool = True
    top_k: int = 0
    latent_embed_mode: str = "sample"  # sample|softmax (softmax = no discretization for B embeddings)
    chunk_size: int = 32
    detach_chunk_pkv: bool = True
    chunk_logit_consistency_enable: bool = False
    chunk_logit_consistency_w: float = 0.0

    # losses
    rec_w: float = 1.0
    token_ce_w: float = 0.0   # CE(logits, sampled_ids) as a "self-sharpening" term
    norm_match_w: float = 0.0
    temporal_diff_w: float = 0.0

    # proposal constraints (optional)
    proposal_enable: bool = False
    possharp_w: float = 0.01
    proposal_h_cap: float = 3.0
    proposal_possharp_temp: float = 0.3
    proposal_tau_ent: float = 0.7
    proposal_tau_marg: float = 1.3
    proposal_ema_decay: float = 0.99
    proposal_info_nce_temperature: float = 0.2
    proposal_bank_size: int = 4096
    marg_w: float = 0.01
    info_w: float = 0.01

    # gaussian alignment (optional)
    gauss_align_enable: bool = False
    gauss_align_w: float = 0.1
    gauss_sigma: float = 0.2
    gauss_lambda_band: float = 1e-1
    gauss_m_target: float = 0.90
    gauss_band_k: float = 1.0
    gauss_lambda_off: float = 3e-2
    gauss_beta: float = 0.0
    gauss_apply: str = "all"  # all|mid|last

    # soft pointer loss (optional; uses cross-attn + per-layer decoder hiddens)
    softptr_enable: bool = False
    softptr_w: float = 0.0
    softptr_d_proj: int = 128
    softptr_use_bridge: bool = True
    softptr_L1: int = 0
    softptr_L2: int = -1  # <=0 means "all layers"
    softptr_tau: float = 3.0
    softptr_head_topk: int = 0  # <=0 means "all heads"
    softptr_detach_w: bool = False
    softptr_lambda_cos: float = 0.0
    softptr_l2p_enable: bool = True
    softptr_apply: str = "all"  # all|mid|last (when not "all", overrides L1/L2)

    # optim
    batch_size: int = 64
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 4
    train_log_interval: int = 50

    seed: int = 0
    device: str = "cuda"
    pretrain_ckpt: str = ""
    pretrain_strict: bool = False


def _load_pretrained_model(model: nn.Module, ckpt_path: Path, strict: bool = False) -> None:
    obj: Any = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    # Backward compatibility: ReconDecoder query_token -> start_token.
    old_k = "recon_decoder.query_token"
    new_k = "recon_decoder.start_token"
    if old_k in state and new_k not in state:
        state[new_k] = state.pop(old_k)

    if strict:
        model.load_state_dict(state, strict=True)
        print(f"[pretrain] loaded (strict=True): {ckpt_path}", flush=True)
        return

    incompatible = model.load_state_dict(state, strict=False)
    miss = list(incompatible.missing_keys)
    unexp = list(incompatible.unexpected_keys)
    print(
        f"[pretrain] loaded (strict=False): {ckpt_path} "
        f"(missing={len(miss)}, unexpected={len(unexp)})",
        flush=True,
    )
    if len(miss) > 0:
        print(f"[pretrain] missing keys (first 20): {miss[:20]}", flush=True)
    if len(unexp) > 0:
        print(f"[pretrain] unexpected keys (first 20): {unexp[:20]}", flush=True)


def build_model_and_stats(cfg: TrainConfig) -> Tuple[MotionSeq2SeqARAE, np.ndarray, np.ndarray]:
    data_root = Path(cfg.data_root)
    train_files = list_motion_files(data_root, "train")
    mean, std = compute_mean_std(train_files)

    # infer feature dim
    x0 = np.load(train_files[0])
    feat_dim = int(x0.shape[1])

    # derive latent_len_max
    T = int(cfg.max_motion_len)
    P = int(cfg.patch_len)
    T_pad = int(math.ceil(T / P) * P)
    N = T_pad // P
    if cfg.latent_len > 0:
        Tb = int(cfg.latent_len)
    else:
        Tb = int(math.ceil(N / max(cfg.compression_ratio, 1e-6)))

    gauss_cfg = dict(
        enable=cfg.gauss_align_enable,
        sigma=cfg.gauss_sigma,
        lambda_band=cfg.gauss_lambda_band,
        m_target=cfg.gauss_m_target,
        band_k=cfg.gauss_band_k,
        lambda_off=cfg.gauss_lambda_off,
        beta=cfg.gauss_beta,
        apply=cfg.gauss_apply,
    )
    softptr_cfg = dict(
        enable=cfg.softptr_enable,
        d_proj=cfg.softptr_d_proj,
        use_bridge=cfg.softptr_use_bridge,
        L1=cfg.softptr_L1,
        L2=cfg.softptr_L2,
        tau=cfg.softptr_tau,
        head_topk=cfg.softptr_head_topk,
        detach_w=cfg.softptr_detach_w,
        lambda_cos=cfg.softptr_lambda_cos,
        l2p_enable=cfg.softptr_l2p_enable,
        apply=cfg.softptr_apply,
    )

    model = MotionSeq2SeqARAE(
        feat_dim=feat_dim,
        patch_len=cfg.patch_len,
        vocab_size=cfg.vocab_size,
        compression_ratio=cfg.compression_ratio,
        latent_len_max=Tb,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        recon_layers=cfg.recon_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_motion_len=cfg.max_motion_len,
        gauss_cfg=gauss_cfg,
        softptr_cfg=softptr_cfg,
    )
    return model, mean, std


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    save_dir = Path(cfg.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_log_{run_id}.jsonl"

    model, mean, std = build_model_and_stats(cfg)
    model.to(device)
    if str(cfg.pretrain_ckpt).strip():
        _load_pretrained_model(model, Path(cfg.pretrain_ckpt), strict=bool(cfg.pretrain_strict))

    # save normalization
    np.save(save_dir / "mean.npy", mean)
    np.save(save_dir / "std.npy", std)
    save_json(save_dir / "train_config.json", asdict(cfg))

    # datasets
    ds_train = HumanML3DMotionDataset(Path(cfg.data_root), "train", cfg.max_motion_len, mean=mean, std=std, random_crop=True)
    ds_test = HumanML3DMotionDataset(Path(cfg.data_root), "test", cfg.max_motion_len, mean=mean, std=std, random_crop=False)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    proposal = None
    if cfg.proposal_enable:
        proposal = ProposalConstraintsSoftmax(
            vocab_size=cfg.vocab_size,
            H_cap=cfg.proposal_h_cap,
            possharp_temp=cfg.proposal_possharp_temp,
            tau_ent=cfg.proposal_tau_ent,
            tau_marg=cfg.proposal_tau_marg,
            ema_decay=cfg.proposal_ema_decay,
            info_nce_temperature=cfg.proposal_info_nce_temperature,
            bank_size=cfg.proposal_bank_size,
        ).to(device)

    step = 0
    best = float("inf")
    train_tok_stats = TokenStatsAggregator(cfg.vocab_size)
    detach_chunk_pkv_train = bool(cfg.detach_chunk_pkv)
    if not detach_chunk_pkv_train:
        print(
            "[train] forcing chunk cache detach for per-chunk optimizer.step() to avoid cross-chunk autograd reuse.",
            flush=True,
        )
        detach_chunk_pkv_train = True

    def _encode_chunk_incremental(
        enc: TransformerEncoder,
        x_new: torch.Tensor,              # [B,Lnew,D]
        keep_new: torch.Tensor,           # [B,Lnew] bool
        past_layer_states: Optional[List[torch.Tensor]],  # list[n_layers] of [B,Lpast,D]
        keep_past: Optional[torch.Tensor],                # [B,Lpast] bool
        *,
        detach_past_on_return: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Incremental encoder forward without re-encoding past chunks.
        Returns:
          y_new: [B,Lnew,D]              (current chunk output)
          full_states: list[n_layers]    ([B,Lpast+Lnew,D] per layer)
          keep_full: [B,Lpast+Lnew] bool
        """
        layers = enc.encoder.layers
        x = x_new
        keep_full = keep_new if keep_past is None else torch.cat([keep_past, keep_new], dim=1)
        key_padding_mask = ~keep_full

        full_states: List[torch.Tensor] = []
        for li, layer in enumerate(layers):
            if not bool(getattr(layer, "norm_first", False)):
                raise RuntimeError("Incremental encoder path requires norm_first=True.")

            past_l = None if past_layer_states is None else past_layer_states[li]
            q = layer.norm1(x)
            if past_l is None:
                kv = q
            else:
                kv = torch.cat([layer.norm1(past_l), q], dim=1)

            y, _ = layer.self_attn(
                q, kv, kv,
                attn_mask=None,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + layer.dropout1(y)

            y = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
            x = x + layer.dropout2(y)

            if past_l is None:
                full_l = x
            else:
                full_l = torch.cat([past_l, x], dim=1)
            if detach_past_on_return:
                full_states.append(full_l.detach())
            else:
                full_states.append(full_l)

        return x, full_states, keep_full

    def _build_chunk_plans(
        n_a: int,
        n_b: int,
        a_chunk_size: int,
        compression_ratio: float,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Return list of (a_start, a_len, b_start, b_len).
        A chunk length = a_chunk_size
        B chunk length = round(A_len / compression_ratio)
        """
        plans: List[Tuple[int, int, int, int]] = []
        a_cursor = 0
        b_cursor = 0
        while a_cursor < n_a:
            a_len = min(a_chunk_size, n_a - a_cursor)
            b_nom = max(1, int(round(float(a_len) / max(float(compression_ratio), 1e-6))))
            b_len = min(b_nom, max(n_b - b_cursor, 0))
            plans.append((a_cursor, a_len, b_cursor, b_len))
            a_cursor += a_len
            b_cursor += b_len
        return plans

    def run_eval() -> Dict[str, float]:
        model.eval()

        n_batches = 0
        sum_loss = 0.0
        sum_rec = 0.0
        sum_ce = 0.0
        sum_norm_match = 0.0
        sum_temporal_diff = 0.0
        sum_align = 0.0
        sum_align_p2l = 0.0
        sum_align_l2p = 0.0
        sum_softptr = 0.0
        sum_softptr_p2l = 0.0
        sum_softptr_l2p = 0.0
        sum_chunk_logit_cons = 0.0
        sum_band_mass_p2l = 0.0
        sum_band_mass_l2p = 0.0
        sum_Lpos = 0.0
        sum_Lmarg = 0.0
        sum_Linfo = 0.0
        sum_xhat_norm = 0.0
        sum_xpad_norm = 0.0
        sum_norm_gap = 0.0
        tok_stats = TokenStatsAggregator(cfg.vocab_size)

        with torch.no_grad():
            for batch in dl_test:
                x = batch["motion"].to(device)
                m = batch["mask"].to(device)
                out = model(
                    x, m,
                    tau=cfg.gumbel_tau,
                    scale=cfg.gumbel_scale,
                    hard=cfg.gumbel_hard,
                    top_k=cfg.top_k,
                    deterministic_tokens=False,  # match train-time token generation
                    latent_embed_mode=cfg.latent_embed_mode,
                    return_attn=True,
                    chunk_size=cfg.chunk_size,
                    detach_chunk_pkv=cfg.detach_chunk_pkv,
                    chunk_logit_consistency_enable=cfg.chunk_logit_consistency_enable,
                )

                # reconstruction
                x_pad = patchify(x, m, cfg.patch_len)[0]
                loss_rec = masked_mse(out["x_hat"], x_pad, out["mask_pad"])
                loss_norm_match = masked_norm_match_loss(out["x_hat"], x_pad, out["mask_pad"])
                loss_temporal_diff = masked_temporal_diff_mse(out["x_hat"], x_pad, out["mask_pad"])
                # xhat_norm = out["x_hat"].norm(dim=-1).mean()
                # xpad_norm = x_pad.norm(dim=-1).mean()
                # norm_gap = (xhat_norm - xpad_norm).abs()

                eps = 1e-8
                valid = out["mask_pad"]                 # [B, T_pad] bool
                valid = valid.to(x_pad.dtype).unsqueeze(-1)  # [B,T,1]
                den = valid.sum().clamp_min(eps)

                num = (valid * (x_pad)).detach().norm(dim=-1).sum()
                xpad_norm = num / den
                num = (valid * (out["x_hat"])).detach().norm(dim=-1).sum()
                xhat_norm = num / den
                norm_gap = (xhat_norm - xpad_norm).abs()

                # token CE (optional)
                if cfg.token_ce_w > 0.0:
                    lm = out["latent_mask"]
                    logits = out["token_logits"][lm]
                    ids = out["token_ids"][lm]
                    loss_ce = F.cross_entropy(logits, ids) if logits.numel() > 0 else torch.zeros((), device=device)
                else:
                    loss_ce = torch.zeros((), device=device)

                # gaussian alignment (optional)
                if cfg.gauss_align_enable:
                    loss_align = out["gauss_align_loss"]
                    loss_align_p2l = out.get("gauss_align_loss_p2l", torch.zeros((), device=device))
                    loss_align_l2p = out.get("gauss_align_loss_l2p", torch.zeros((), device=device))
                else:
                    loss_align = torch.zeros((), device=device)
                    loss_align_p2l = torch.zeros((), device=device)
                    loss_align_l2p = torch.zeros((), device=device)

                loss_softptr = out["softptr_loss"] if cfg.softptr_enable else torch.zeros((), device=device)
                loss_softptr_p2l = out.get("softptr_loss_p2l", torch.zeros((), device=device))
                loss_softptr_l2p = out.get("softptr_loss_l2p", torch.zeros((), device=device))
                loss_chunk_logit_cons = out.get("chunk_logit_consistency_loss", torch.zeros((), device=device))
                band_mass_p2l = out.get("band_mass_p2l", torch.zeros((), device=device))
                band_mass_l2p = out.get("band_mass_l2p", torch.zeros((), device=device))

                # proposal constraints (optional) -- do NOT update internal EMA/bank in validation
                if proposal is not None:
                    L_pos, L_marg, L_info = proposal(out["token_logits"], valid_mask=out["latent_mask"], update_state=False)
                else:
                    L_pos = torch.zeros((), device=device)
                    L_marg = torch.zeros((), device=device)
                    L_info = torch.zeros((), device=device)

                loss = cfg.rec_w * loss_rec
                loss = loss + cfg.token_ce_w * loss_ce
                loss = loss + cfg.norm_match_w * loss_norm_match
                loss = loss + cfg.temporal_diff_w * loss_temporal_diff
                loss = loss + cfg.gauss_align_w * loss_align
                loss = loss + cfg.softptr_w * loss_softptr
                loss = loss + cfg.chunk_logit_consistency_w * loss_chunk_logit_cons
                loss = loss + cfg.possharp_w * L_pos + cfg.marg_w * L_marg + cfg.info_w * L_info

                n_batches += 1
                sum_loss += float(loss.item())
                sum_rec += float(loss_rec.item())
                sum_ce += float(loss_ce.item())
                sum_norm_match += float(loss_norm_match.item())
                sum_temporal_diff += float(loss_temporal_diff.item())
                sum_align += float(loss_align.item())
                sum_align_p2l += float(loss_align_p2l.item())
                sum_align_l2p += float(loss_align_l2p.item())
                sum_softptr += float(loss_softptr.item())
                sum_softptr_p2l += float(loss_softptr_p2l.item())
                sum_softptr_l2p += float(loss_softptr_l2p.item())
                sum_chunk_logit_cons += float(loss_chunk_logit_cons.item())
                sum_band_mass_p2l += float(band_mass_p2l.item())
                sum_band_mass_l2p += float(band_mass_l2p.item())
                sum_Lpos += float(L_pos.item())
                sum_Lmarg += float(L_marg.item())
                sum_Linfo += float(L_info.item())
                sum_xhat_norm += float(xhat_norm.item())
                sum_xpad_norm += float(xpad_norm.item())
                sum_norm_gap += float(norm_gap.item())

                # token stats
                tok_stats.update(out["token_logits"], out["token_ids"], out["latent_mask"])

        model.train()
        denom = max(n_batches, 1)
        stats = {
            "loss": sum_loss / denom,
            "loss_rec": sum_rec / denom,
            "loss_ce": sum_ce / denom,
            "loss_norm_match": sum_norm_match / denom,
            "loss_temporal_diff": sum_temporal_diff / denom,
            "loss_align": sum_align / denom,
            "loss_align_p2l": sum_align_p2l / denom,
            "loss_align_l2p": sum_align_l2p / denom,
            "loss_softptr": sum_softptr / denom,
            "loss_softptr_p2l": sum_softptr_p2l / denom,
            "loss_softptr_l2p": sum_softptr_l2p / denom,
            "loss_chunk_logit_consistency": sum_chunk_logit_cons / denom,
            "band_mass_p2l": sum_band_mass_p2l / denom,
            "band_mass_l2p": sum_band_mass_l2p / denom,
            "L_pos": sum_Lpos / denom,
            "L_marg": sum_Lmarg / denom,
            "L_info": sum_Linfo / denom,
            "xhat_norm": sum_xhat_norm / denom,
            "xpad_norm": sum_xpad_norm / denom,
            "xhat_xpad_norm_gap": sum_norm_gap / denom,
        }
        id_stats = tok_stats.finalize()
        for k, v in id_stats.items():
            stats[f"id_{k}"] = v
        return stats

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for batch in dl_train:
            x = batch["motion"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            # ================================
            # Phase 0) Prepare A/B chunk plan
            # ================================
            x_pad, mask_pad, x_patch, patch_mask = patchify(x, m, cfg.patch_len)
            B, N, _ = x_patch.shape
            Tb = int(model.latent_len_max)
            ch = int(cfg.chunk_size)
            if ch <= 0:
                ch = int(max(Tb, N))

            src_key_padding_mask = ~patch_mask
            latent_mask = model.compute_latent_mask(patch_mask)

            need_softptr = bool(model.softptr is not None and cfg.softptr_enable)
            gauss_apply = str(model.gauss_align_metric.apply).lower().strip()
            softptr_apply = str((model.softptr_cfg or {}).get("apply", "all")).lower().strip()
            attn_apply_modes: List[str] = []
            if cfg.gauss_align_enable:
                attn_apply_modes.append(gauss_apply)
            if need_softptr:
                attn_apply_modes.append(softptr_apply)
            attn_apply = _merge_attn_apply(attn_apply_modes)
            need_attn_tok = bool(cfg.gauss_align_enable or need_softptr)

            chunk_plans = _build_chunk_plans(
                n_a=N,
                n_b=Tb,
                a_chunk_size=ch,
                compression_ratio=cfg.compression_ratio,
            )

            # ========================================
            # Phase 1) Initialize cross-chunk caches
            # ========================================
            token_past: Optional[torch.Tensor] = None
            recon_past: Optional[torch.Tensor] = None
            memA_cache: Optional[torch.Tensor] = None
            maskA_cache: Optional[torch.Tensor] = None
            memB_cache: Optional[torch.Tensor] = None
            maskB_cache: Optional[torch.Tensor] = None
            encA_layer_cache: Optional[List[torch.Tensor]] = None
            encB_layer_cache: Optional[List[torch.Tensor]] = None
            tok_ids_chunks: List[torch.Tensor] = []
            tok_logits_chunks: List[torch.Tensor] = []
            tok_emb_chunks: List[torch.Tensor] = []
            rec_chunks: List[torch.Tensor] = []
            rec_loss_ce = 0.0
            rec_loss_norm_match = 0.0
            rec_loss_temporal_diff = 0.0
            rec_loss_align_p2l = 0.0
            rec_loss_align_l2p = 0.0
            rec_loss_softptr_p2l = 0.0
            rec_loss_softptr_l2p = 0.0
            rec_loss_chunk_cons = 0.0
            rec_loss_rec = 0.0
            rec_L_pos = 0.0
            rec_L_marg = 0.0
            rec_L_info = 0.0
            rec_band_mass_p2l = 0.0
            rec_band_mass_l2p = 0.0
            n_tok_chunks = 0
            n_chunk_cons = 0
            n_rec_chunks = 0

            sp_cfg = model.softptr_cfg or {}
            need_softptr_l2p = bool(need_softptr and bool(sp_cfg.get("l2p_enable", True)))
            need_attn_recon = bool(cfg.gauss_align_enable or need_softptr_l2p)
            for a_start, a_len, t_start, t_len in chunk_plans:
                # ======================================
                # Phase 2) A -> B for current chunk pair
                # ======================================
                a_chunk = x_patch[:, a_start : a_start + a_len, :]
                a_mask_chunk = patch_mask[:, a_start : a_start + a_len]
                f0 = a_start * cfg.patch_len
                f1 = (a_start + a_len) * cfg.patch_len
                x_target_chunk = x_pad[:, f0:f1, :]
                mask_target_chunk = mask_pad[:, f0:f1]

                h_a = model.in_proj(a_chunk)
                h_a = model.enc_pos(h_a, offset=int(a_start))
                memA_chunk, encA_full, memA_mask = _encode_chunk_incremental(
                    model.encoder,
                    h_a,
                    a_mask_chunk,
                    encA_layer_cache,
                    maskA_cache,
                    detach_past_on_return=detach_chunk_pkv_train,
                )
                memA_keys = encA_full[-1]

                loss_ce_c = torch.zeros((), device=device)
                loss_align_p2l_c = torch.zeros((), device=device)
                loss_softptr_p2l_c = torch.zeros((), device=device)
                loss_chunk_cons_c = torch.zeros((), device=device)
                L_pos_c = torch.zeros((), device=device)
                L_marg_c = torch.zeros((), device=device)
                L_info_c = torch.zeros((), device=device)
                band_mass_p2l_c = torch.zeros((), device=device)
                lm_chunk = latent_mask[:, 0:0]
                embs_c: Optional[torch.Tensor] = None
                memB_chunk: Optional[torch.Tensor] = None

                opt.zero_grad(set_to_none=True)
                if t_len > 0:
                    lm_chunk = latent_mask[:, t_start : t_start + t_len]
                    token_past_in = token_past
                    ids_c, logits_c, embs_c, attn_c, hid_c, token_past = model.token_decoder.generate(
                        memA_keys,
                        ~memA_mask,
                        t_len,
                        tau=cfg.gumbel_tau,
                        scale=cfg.gumbel_scale,
                        hard=cfg.gumbel_hard,
                        top_k=cfg.top_k,
                        return_attn=need_attn_tok,
                        return_hidden=need_softptr,
                        deterministic=False,
                        embed_mode=cfg.latent_embed_mode,
                        attn_apply=attn_apply,
                        past_token_embs=token_past_in,
                        pos_offset=t_start,
                        detach_past_on_return=detach_chunk_pkv_train,
                    )
                    tok_ids_chunks.append(ids_c.detach())
                    tok_logits_chunks.append(logits_c.detach())
                    tok_emb_chunks.append(embs_c.detach())
                    train_tok_stats.update(logits_c.detach(), ids_c.detach(), lm_chunk.detach())
                    n_tok_chunks += 1

                    if cfg.token_ce_w > 0.0:
                        logits_v = logits_c[lm_chunk]
                        ids_v = ids_c[lm_chunk]
                        loss_ce_c = F.cross_entropy(logits_v, ids_v) if logits_v.numel() > 0 else torch.zeros((), device=device)

                    attn_local_p2l: List[Optional[torch.Tensor]] = []
                    for ai in attn_c:
                        if ai is None:
                            attn_local_p2l.append(None)
                        else:
                            attn_local_p2l.append(ai[:, :, :, -a_len:])

                    if cfg.gauss_align_enable:
                        loss_align_p2l_c = model.gauss_align_metric(
                            attn_local_p2l,
                            src_keep_mask=a_mask_chunk,
                            tgt_keep_mask=lm_chunk,
                        )
                        band_mass_p2l_c = model.gauss_align_metric.compute_band_mass(
                            attn_local_p2l,
                            src_keep_mask=a_mask_chunk,
                            tgt_keep_mask=lm_chunk,
                        )

                    if need_softptr:
                        sp_apply = str(sp_cfg.get("apply", "all")).lower().strip()
                        if sp_apply == "all":
                            L1 = int(sp_cfg.get("L1", 0))
                            L2 = int(sp_cfg.get("L2", -1))
                            if L2 <= 0:
                                L2 = len(attn_local_p2l)
                            L2 = min(L2, len(attn_local_p2l))
                            L1 = max(0, min(L1, max(L2 - 1, 0)))
                        else:
                            L1, L2 = _resolve_layer_span(len(attn_local_p2l), sp_apply)
                        head_topk = int(sp_cfg.get("head_topk", 0))
                        head_topk_opt = None if head_topk <= 0 else head_topk
                        tau_sp = float(sp_cfg.get("tau", 3.0))
                        detach_w = bool(sp_cfg.get("detach_w", False))
                        lambda_cos = float(sp_cfg.get("lambda_cos", 0.0))
                        if L2 > L1:
                            loss_softptr_p2l_c, _ = model.softptr(
                                attn_local_p2l,
                                memA_chunk,
                                hid_c,
                                src_keep_mask=a_mask_chunk,
                                tgt_keep_mask=lm_chunk,
                                L1=L1,
                                L2=L2,
                                tau=tau_sp,
                                head_topk=head_topk_opt,
                                detach_w=detach_w,
                                lambda_cos=lambda_cos,
                                direction="p2l",
                            )

                    if cfg.chunk_logit_consistency_enable:
                        with torch.no_grad():
                            _, logits_np, _, _, _, _ = model.token_decoder.generate(
                                memA_keys,
                                ~memA_mask,
                                t_len,
                                tau=cfg.gumbel_tau,
                                scale=cfg.gumbel_scale,
                                hard=cfg.gumbel_hard,
                                top_k=cfg.top_k,
                                return_attn=False,
                                return_hidden=False,
                                deterministic=True,
                                embed_mode=cfg.latent_embed_mode,
                                attn_apply=attn_apply,
                                past_token_embs=token_past_in,
                                pos_offset=t_start,
                                detach_past_on_return=False,
                            )
                        vm = lm_chunk.to(logits_c.dtype).unsqueeze(-1)
                        den = vm.sum().clamp_min(1.0) * float(cfg.vocab_size)
                        loss_chunk_cons_c = (((logits_c - logits_np) ** 2) * vm).sum() / den
                        n_chunk_cons += 1

                    if proposal is not None:
                        L_pos_c, L_marg_c, L_info_c = proposal(logits_c, valid_mask=lm_chunk)

                    h_b = model.latent_pos(embs_c, offset=int(t_start))
                    memB_chunk, encB_full, memB_mask_new = _encode_chunk_incremental(
                        model.latent_encoder,
                        h_b,
                        lm_chunk,
                        encB_layer_cache,
                        maskB_cache,
                        detach_past_on_return=detach_chunk_pkv_train,
                    )

                if memB_chunk is not None:
                    memB_keys = encB_full[-1]
                    memB_mask = memB_mask_new
                else:
                    memB_keys = memB_cache
                    memB_mask = maskB_cache

                if memB_keys is None or memB_mask is None:
                    memA_cache = memA_keys.detach()
                    maskA_cache = memA_mask
                    encA_layer_cache = encA_full
                    continue

                # ======================================
                # Phase 3) B -> A for current chunk pair
                # ======================================
                h_c, attn_c_recon, hid_c_recon, recon_past = model.recon_decoder(
                    memB_keys,
                    ~memB_mask,
                    out_len=a_len,
                    return_attn=need_attn_recon,
                    return_hidden=need_softptr_l2p,
                    attn_apply=attn_apply,
                    past_inputs=recon_past,
                    pos_offset=a_start,
                    detach_past_on_return=detach_chunk_pkv_train,
                )
                x_patch_hat_c = model.out_proj(h_c)
                x_hat_chunk = unpatchify(x_patch_hat_c, model.patch_len, model.feat_dim)
                rec_chunks.append(x_hat_chunk.detach())
                n_rec_chunks += 1
                loss_rec_c = masked_mse(x_hat_chunk, x_target_chunk, mask_target_chunk)
                loss_norm_match_c = masked_norm_match_loss(x_hat_chunk, x_target_chunk, mask_target_chunk)
                loss_temporal_diff_c = masked_temporal_diff_mse(x_hat_chunk, x_target_chunk, mask_target_chunk)
                loss_align_l2p_c = torch.zeros((), device=device)
                loss_softptr_l2p_c = torch.zeros((), device=device)
                band_mass_l2p_c = torch.zeros((), device=device)

                attn_local_l2p: List[Optional[torch.Tensor]] = []
                for ai in attn_c_recon:
                    if ai is None or t_len <= 0:
                        attn_local_l2p.append(None)
                    else:
                        attn_local_l2p.append(ai[:, :, :, -t_len:])

                if cfg.gauss_align_enable and t_len > 0:
                    l2p_sigma_scale = 1.0 / max(model.compression_ratio, 1e-6)
                    loss_align_l2p_c = model.gauss_align_metric(
                        attn_local_l2p,
                        src_keep_mask=lm_chunk,
                        tgt_keep_mask=a_mask_chunk,
                        sigma_scale=l2p_sigma_scale,
                    )
                    band_mass_l2p_c = model.gauss_align_metric.compute_band_mass(
                        attn_local_l2p,
                        src_keep_mask=lm_chunk,
                        tgt_keep_mask=a_mask_chunk,
                        sigma_scale=l2p_sigma_scale,
                    )

                if need_softptr_l2p and t_len > 0 and memB_chunk is not None:
                    sp_apply = str(sp_cfg.get("apply", "all")).lower().strip()
                    if sp_apply == "all":
                        L1 = int(sp_cfg.get("L1", 0))
                        L2 = int(sp_cfg.get("L2", -1))
                        if L2 <= 0:
                            L2 = len(attn_local_l2p)
                        L2 = min(L2, len(attn_local_l2p))
                        L1 = max(0, min(L1, max(L2 - 1, 0)))
                    else:
                        L1, L2 = _resolve_layer_span(len(attn_local_l2p), sp_apply)
                    head_topk = int(sp_cfg.get("head_topk", 0))
                    head_topk_opt = None if head_topk <= 0 else head_topk
                    tau_sp = float(sp_cfg.get("tau", 3.0))
                    detach_w = bool(sp_cfg.get("detach_w", False))
                    lambda_cos = float(sp_cfg.get("lambda_cos", 0.0))
                    if L2 > L1:
                        loss_softptr_l2p_c, _ = model.softptr(
                            attn_local_l2p,
                            memB_chunk,
                            hid_c_recon,
                            src_keep_mask=lm_chunk,
                            tgt_keep_mask=a_mask_chunk,
                            L1=L1,
                            L2=L2,
                            tau=tau_sp,
                            head_topk=head_topk_opt,
                            detach_w=detach_w,
                            lambda_cos=lambda_cos,
                            direction="l2p",
                        )

                # ============================================
                # Phase 4) Chunk loss + single backward/step
                # ============================================
                loss_align_c = 0.5 * (loss_align_p2l_c + loss_align_l2p_c)
                loss_softptr_c = 0.5 * (loss_softptr_p2l_c + loss_softptr_l2p_c)
                loss_chunk = cfg.rec_w * loss_rec_c
                loss_chunk = loss_chunk + cfg.token_ce_w * loss_ce_c
                loss_chunk = loss_chunk + cfg.norm_match_w * loss_norm_match_c
                loss_chunk = loss_chunk + cfg.temporal_diff_w * loss_temporal_diff_c
                loss_chunk = loss_chunk + cfg.gauss_align_w * loss_align_c
                loss_chunk = loss_chunk + cfg.softptr_w * loss_softptr_c
                loss_chunk = loss_chunk + cfg.chunk_logit_consistency_w * loss_chunk_cons_c
                loss_chunk = loss_chunk + cfg.possharp_w * L_pos_c + cfg.marg_w * L_marg_c + cfg.info_w * L_info_c

                if loss_chunk.requires_grad:
                    loss_chunk.backward()
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    step += 1

                rec_loss_rec += float(loss_rec_c.detach().item())
                rec_loss_ce += float(loss_ce_c.detach().item())
                rec_loss_norm_match += float(loss_norm_match_c.detach().item())
                rec_loss_temporal_diff += float(loss_temporal_diff_c.detach().item())
                rec_loss_align_p2l += float(loss_align_p2l_c.detach().item())
                rec_loss_align_l2p += float(loss_align_l2p_c.detach().item())
                rec_loss_softptr_p2l += float(loss_softptr_p2l_c.detach().item())
                rec_loss_softptr_l2p += float(loss_softptr_l2p_c.detach().item())
                rec_loss_chunk_cons += float(loss_chunk_cons_c.detach().item())
                rec_L_pos += float(L_pos_c.detach().item())
                rec_L_marg += float(L_marg_c.detach().item())
                rec_L_info += float(L_info_c.detach().item())
                rec_band_mass_p2l += float(band_mass_p2l_c.detach().item())
                rec_band_mass_l2p += float(band_mass_l2p_c.detach().item())

                memA_cache = memA_keys.detach()
                maskA_cache = memA_mask
                encA_layer_cache = encA_full
                memB_cache = memB_keys.detach()
                maskB_cache = memB_mask
                if memB_chunk is not None:
                    encB_layer_cache = encB_full

            token_ids = torch.cat(tok_ids_chunks, dim=1) if len(tok_ids_chunks) > 0 else torch.zeros((B, 0), device=device, dtype=torch.long)
            token_logits = torch.cat(tok_logits_chunks, dim=1) if len(tok_logits_chunks) > 0 else torch.zeros((B, 0, cfg.vocab_size), device=device)
            token_embs = torch.cat(tok_emb_chunks, dim=1) if len(tok_emb_chunks) > 0 else torch.zeros((B, 0, model.d_model), device=device)

            out = {
                "x_hat": torch.cat(rec_chunks, dim=1),
                "mask_pad": mask_pad,
                "token_ids": token_ids,
                "token_logits": token_logits,
                "latent_mask": latent_mask,
                "patch_mask": patch_mask,
            }

            # batch-level logging values (chunk averages)
            n_tok = max(n_tok_chunks, 1)
            n_rec = max(n_rec_chunks, 1)
            loss_rec = torch.tensor(rec_loss_rec / n_rec, device=device)
            loss_ce = torch.tensor(rec_loss_ce / n_tok, device=device)
            loss_norm_match = torch.tensor(rec_loss_norm_match / n_rec, device=device)
            loss_temporal_diff = torch.tensor(rec_loss_temporal_diff / n_rec, device=device)
            loss_align_p2l = torch.tensor(rec_loss_align_p2l / n_tok, device=device)
            loss_align_l2p = torch.tensor(rec_loss_align_l2p / n_rec, device=device)
            loss_align = 0.5 * (loss_align_p2l + loss_align_l2p)
            loss_softptr_p2l = torch.tensor(rec_loss_softptr_p2l / n_tok, device=device)
            loss_softptr_l2p = torch.tensor(rec_loss_softptr_l2p / n_rec, device=device)
            loss_softptr = 0.5 * (loss_softptr_p2l + loss_softptr_l2p)
            loss_chunk_logit_cons = torch.tensor(rec_loss_chunk_cons / max(n_chunk_cons, 1), device=device)
            band_mass_p2l = torch.tensor(rec_band_mass_p2l / n_tok, device=device)
            band_mass_l2p = torch.tensor(rec_band_mass_l2p / n_rec, device=device)
            L_pos = torch.tensor(rec_L_pos / n_tok, device=device)
            L_marg = torch.tensor(rec_L_marg / n_tok, device=device)
            L_info = torch.tensor(rec_L_info / n_tok, device=device)

            loss = cfg.rec_w * loss_rec
            loss = loss + cfg.token_ce_w * loss_ce
            loss = loss + cfg.norm_match_w * loss_norm_match
            loss = loss + cfg.temporal_diff_w * loss_temporal_diff
            loss = loss + cfg.gauss_align_w * loss_align
            loss = loss + cfg.softptr_w * loss_softptr
            loss = loss + cfg.chunk_logit_consistency_w * loss_chunk_logit_cons
            loss = loss + cfg.possharp_w * L_pos + cfg.marg_w * L_marg + cfg.info_w * L_info

            eps = 1e-8
            valid = out["mask_pad"].to(x_pad.dtype).unsqueeze(-1)
            den = valid.sum().clamp_min(eps)
            num = (valid * (x_pad)).detach().norm(dim=-1).sum()
            xpad_norm = num / den
            num = (valid * (out["x_hat"])).detach().norm(dim=-1).sum()
            xhat_norm = num / den
            norm_gap = (xhat_norm - xpad_norm).abs()

            log_interval = max(int(cfg.train_log_interval), 1)
            if step % log_interval == 0:
                log = {
                    "split": "train",
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "loss_rec": float(loss_rec.item()),
                    "loss_ce": float(loss_ce.item()),
                    "loss_norm_match": float(loss_norm_match.item()),
                    "loss_temporal_diff": float(loss_temporal_diff.item()),
                    "loss_align": float(loss_align.item()),
                    "loss_align_p2l": float(loss_align_p2l.item()),
                    "loss_align_l2p": float(loss_align_l2p.item()),
                    "loss_softptr": float(loss_softptr.item()),
                    "loss_softptr_p2l": float(loss_softptr_p2l.item()),
                    "loss_softptr_l2p": float(loss_softptr_l2p.item()),
                    "loss_chunk_logit_consistency": float(loss_chunk_logit_cons.item()),
                    "band_mass_p2l": float(band_mass_p2l.item()),
                    "band_mass_l2p": float(band_mass_l2p.item()),
                    "L_pos": float(L_pos.item()),
                    "L_marg": float(L_marg.item()),
                    "L_info": float(L_info.item()),
                    "xhat_norm": float(xhat_norm.item()),
                    "xpad_norm": float(xpad_norm.item()),
                    "xhat_xpad_norm_gap": float(norm_gap.item()),
                    "time": time.time() - t0,
                }
                id_stats = train_tok_stats.finalize()
                for k, v in id_stats.items():
                    log[f"id_{k}"] = v
                train_tok_stats = TokenStatsAggregator(cfg.vocab_size)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log) + "\n")
                print(log, flush=True)

        # eval per epoch
        val = run_eval()
        val_log = {"split": "val", "epoch": epoch, "step": step, "time": time.time() - t0}
        val_log.update(val)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(val_log) + "\n")
        print(val_log, flush=True)
        val_rec = float(val.get("loss_rec", float("inf")))

        # save
        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": asdict(cfg),
            "mean": mean,
            "std": std,
            "epoch": epoch,
            "step": step,
            "val_rec_mse": val_rec,
            "val_metrics": val,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_rec < best:
            best = val_rec
            torch.save(ckpt, ckpt_dir / "best.pt")


@torch.no_grad()
def load_model(ckpt_path: Path, device: torch.device) -> Tuple[MotionSeq2SeqARAE, np.ndarray, np.ndarray, Dict]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing cfg")

    # build cfg dataclass with defaults
    cfg = TrainConfig(**cfg_dict)

    model, mean, std = build_model_and_stats(cfg)
    state = ckpt["model"]
    # Backward compatibility: ReconDecoder query_token -> start_token.
    old_k = "recon_decoder.query_token"
    new_k = "recon_decoder.start_token"
    if old_k in state and new_k not in state:
        state[new_k] = state.pop(old_k)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, mean, std, cfg_dict


@torch.no_grad()
def encode_tokens(data_root: str, ckpt: str, out_dir: str, split: str, batch_size: int, num_workers: int, seed: int, latent_embed_mode: Optional[str] = None) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, cfg_dict = load_model(Path(ckpt), device)
    cfg = TrainConfig(**cfg_dict)
    embed_mode = latent_embed_mode if latent_embed_mode is not None else getattr(cfg, "latent_embed_mode", "sample")

    ds = HumanML3DMotionDataset(Path(data_root), split, cfg.max_motion_len, mean=mean, std=std, random_crop=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)

    for batch in dl:
        x = batch["motion"].to(device)
        m = batch["mask"].to(device)
        paths = batch["path"]

        out = model(
            x, m,
            tau=cfg.gumbel_tau,
            scale=cfg.gumbel_scale,
            hard=True,
            top_k=0,
            deterministic_tokens=True,
            latent_embed_mode=embed_mode,
            return_attn=False,
            chunk_size=cfg.chunk_size,
            detach_chunk_pkv=cfg.detach_chunk_pkv,
            chunk_logit_consistency_enable=False,
        )
        token_ids = out["token_ids"].cpu().numpy()
        latent_mask = out["latent_mask"].cpu().numpy()

        for i, p in enumerate(paths):
            seq_id = Path(p).stem
            ids = token_ids[i][latent_mask[i]].tolist()
            (out_root / f"{seq_id}.txt").write_text(" ".join(map(str, ids)) + "\n", encoding="utf-8")


@torch.no_grad()
def reconstruct(data_root: str, ckpt: str, out_dir: str, split: str, batch_size: int, num_workers: int, seed: int, latent_embed_mode: Optional[str] = None) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, cfg_dict = load_model(Path(ckpt), device)
    cfg = TrainConfig(**cfg_dict)
    embed_mode = latent_embed_mode if latent_embed_mode is not None else getattr(cfg, "latent_embed_mode", "sample")

    ds = HumanML3DMotionDataset(Path(data_root), split, cfg.max_motion_len, mean=mean, std=std, random_crop=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)

    for batch in dl:
        x = batch["motion"].to(device)
        m = batch["mask"].to(device)
        paths = batch["path"]

        out = model(
            x, m,
            tau=cfg.gumbel_tau,
            scale=cfg.gumbel_scale,
            hard=True,
            top_k=0,
            deterministic_tokens=True,
            latent_embed_mode=embed_mode,
            return_attn=False,
            chunk_size=cfg.chunk_size,
            detach_chunk_pkv=cfg.detach_chunk_pkv,
            chunk_logit_consistency_enable=False,
        )
        x_hat = out["x_hat"].cpu().numpy()  # normalized
        mask_pad = out["mask_pad"].cpu().numpy()

        # de-normalize
        x_hat = x_hat * (std[None, None, :] + 1e-8) + mean[None, None, :]

        for i, p in enumerate(paths):
            seq_id = Path(p).stem
            # Optionally trim to original length by mask (keep exact length)
            valid_len = int(mask_pad[i].sum())
            arr = x_hat[i, :valid_len].astype(np.float32)
            np.save(out_root / f"{seq_id}.npy", arr)


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--data_root", type=str, required=True)
    ap_tr.add_argument("--save_dir", type=str, required=True)
    ap_tr.add_argument("--max_motion_len", type=int, default=196)
    ap_tr.add_argument("--patch_len", type=int, default=1)
    ap_tr.add_argument("--compression_ratio", type=float, default=4.0)
    ap_tr.add_argument("--latent_len", type=int, default=0)

    ap_tr.add_argument("--vocab_size", type=int, default=1024)
    ap_tr.add_argument("--d_model", type=int, default=512)
    ap_tr.add_argument("--n_heads", type=int, default=8)
    ap_tr.add_argument("--d_ff", type=int, default=2048)
    ap_tr.add_argument("--dropout", type=float, default=0.1)
    ap_tr.add_argument("--enc_layers", type=int, default=6)
    ap_tr.add_argument("--dec_layers", type=int, default=6)
    ap_tr.add_argument("--recon_layers", type=int, default=6)

    ap_tr.add_argument("--gumbel_tau", type=float, default=0.5)
    ap_tr.add_argument("--gumbel_scale", type=float, default=0.2)
    g_gumbel = ap_tr.add_mutually_exclusive_group()
    g_gumbel.add_argument("--gumbel_hard", dest="gumbel_hard", action="store_true", default=True,
                          help="Use straight-through Gumbel-Softmax (hard one-hot; default)")
    g_gumbel.add_argument("--gumbel_soft", dest="gumbel_hard", action="store_false",
                          help="Use soft Gumbel-Softmax (not straight-through)")
    ap_tr.add_argument("--top_k", type=int, default=0)
    ap_tr.add_argument(
        "--latent_embed_mode", type=str, default="sample", choices=["sample", "softmax"],
        help="How to form token embeddings in A->B and feed them to B->A. sample: argmax/Gumbel; softmax: expected embedding (no discretization)."
    )
    ap_tr.add_argument("--chunk_size", type=int, default=32,
                       help="Chunk size for autoregressive decoding (A->B and B->A).")
    g_det = ap_tr.add_mutually_exclusive_group()
    g_det.add_argument("--detach_chunk_pkv", dest="detach_chunk_pkv", action="store_true", default=True,
                       help="Detach chunk cache (past states) between chunks (default).")
    g_det.add_argument("--no_detach_chunk_pkv", dest="detach_chunk_pkv", action="store_false",
                       help="Keep gradients through chunk cache.")
    ap_tr.add_argument("--chunk_logit_consistency_enable", action="store_true",
                       help="Add auxiliary chunk consistency objective on token logits.")
    ap_tr.add_argument("--chunk_logit_consistency_w", type=float, default=0.0,
                       help="Weight for chunk logit consistency loss.")

    ap_tr.add_argument("--rec_w", type=float, default=1.0)
    ap_tr.add_argument("--token_ce_w", type=float, default=0.0)
    ap_tr.add_argument("--norm_match_w", type=float, default=0.0,
                       help="Weight for auxiliary norm-matching loss on valid frames.")
    ap_tr.add_argument("--temporal_diff_w", type=float, default=0.0,
                       help="Weight for frame-to-frame difference reconstruction loss.")

    ap_tr.add_argument("--proposal_enable", action="store_true")
    ap_tr.add_argument("--possharp_w", type=float, default=0.01)
    ap_tr.add_argument("--proposal_h_cap", type=float, default=3.0)
    ap_tr.add_argument("--proposal_possharp_temp", type=float, default=0.3)
    ap_tr.add_argument("--proposal_tau_ent", type=float, default=0.7)
    ap_tr.add_argument("--proposal_tau_marg", type=float, default=1.3)
    ap_tr.add_argument("--proposal_ema_decay", type=float, default=0.99)
    ap_tr.add_argument("--proposal_info_nce_temperature", type=float, default=0.2)
    ap_tr.add_argument("--proposal_bank_size", type=int, default=4096)
    ap_tr.add_argument("--marg_w", type=float, default=0.01)
    ap_tr.add_argument("--info_w", type=float, default=0.01)

    ap_tr.add_argument("--gauss_align_enable", action="store_true")
    ap_tr.add_argument("--gauss_align_w", type=float, default=0.1)
    ap_tr.add_argument("--gauss_sigma", type=float, default=0.2)
    ap_tr.add_argument("--gauss_lambda_band", type=float, default=1e-1)
    ap_tr.add_argument("--gauss_m_target", type=float, default=0.90)
    ap_tr.add_argument("--gauss_band_k", type=float, default=1.0)
    ap_tr.add_argument("--gauss_lambda_off", type=float, default=3e-2)
    ap_tr.add_argument("--gauss_beta", type=float, default=0.0)
    ap_tr.add_argument("--gauss_apply", type=str, default="mid", choices=["all", "mid", "last"])

    ap_tr.add_argument("--softptr_enable", action="store_true")
    ap_tr.add_argument("--softptr_w", type=float, default=0.0)
    ap_tr.add_argument("--softptr_d_proj", type=int, default=128)
    g_sp = ap_tr.add_mutually_exclusive_group()
    g_sp.add_argument("--softptr_use_bridge", dest="softptr_use_bridge", action="store_true", default=True,
                      help="Use linear bridge in SoftPointerLoss (default)")
    g_sp.add_argument("--softptr_no_bridge", dest="softptr_use_bridge", action="store_false",
                      help="Disable linear bridge in SoftPointerLoss")
    ap_tr.add_argument("--softptr_L1", type=int, default=0)
    ap_tr.add_argument("--softptr_L2", type=int, default=-1, help="<=0 means use all layers")
    ap_tr.add_argument("--softptr_tau", type=float, default=3.0)
    ap_tr.add_argument("--softptr_head_topk", type=int, default=0, help="<=0 means use all heads")
    ap_tr.add_argument("--softptr_detach_w", action="store_true")
    ap_tr.add_argument("--softptr_lambda_cos", type=float, default=0.0)
    ap_tr.add_argument("--softptr_apply", type=str, default="mid", choices=["all", "mid", "last"],
                       help="Layer selection mode for SoftPointer. When mid/last is used, softptr_L1/L2 are ignored.")
    g_sp_l2p = ap_tr.add_mutually_exclusive_group()
    g_sp_l2p.add_argument("--softptr_l2p_enable", dest="softptr_l2p_enable", action="store_true", default=True)
    g_sp_l2p.add_argument("--softptr_l2p_disable", dest="softptr_l2p_enable", action="store_false")

    ap_tr.add_argument("--batch_size", type=int, default=64)
    ap_tr.add_argument("--epochs", type=int, default=50)
    ap_tr.add_argument("--lr", type=float, default=2e-4)
    ap_tr.add_argument("--weight_decay", type=float, default=0.0)
    ap_tr.add_argument("--grad_clip", type=float, default=1.0)
    ap_tr.add_argument("--num_workers", type=int, default=4)
    ap_tr.add_argument("--train_log_interval", type=int, default=50)

    ap_tr.add_argument("--seed", type=int, default=0)
    ap_tr.add_argument("--device", type=str, default="cuda")
    ap_tr.add_argument("--pretrain_ckpt", type=str, default="",
                       help="Optional checkpoint to initialize model weights before training.")
    ap_tr.add_argument("--pretrain_strict", action="store_true",
                       help="Load pretrain checkpoint with strict=True (default: strict=False).")

    # encode
    ap_en = sub.add_parser("encode")
    ap_en.add_argument("--data_root", type=str, required=True)
    ap_en.add_argument("--ckpt", type=str, required=True)
    ap_en.add_argument("--out_dir", type=str, required=True)
    ap_en.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap_en.add_argument("--batch_size", type=int, default=64)
    ap_en.add_argument("--num_workers", type=int, default=4)
    ap_en.add_argument("--seed", type=int, default=0)
    ap_en.add_argument("--latent_embed_mode", type=str, default="from_ckpt", choices=["from_ckpt", "sample", "softmax"])

    # reconstruct
    ap_rc = sub.add_parser("reconstruct")
    ap_rc.add_argument("--data_root", type=str, required=True)
    ap_rc.add_argument("--ckpt", type=str, required=True)
    ap_rc.add_argument("--out_dir", type=str, required=True)
    ap_rc.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap_rc.add_argument("--batch_size", type=int, default=64)
    ap_rc.add_argument("--num_workers", type=int, default=4)
    ap_rc.add_argument("--seed", type=int, default=0)
    ap_rc.add_argument("--latent_embed_mode", type=str, default="from_ckpt", choices=["from_ckpt", "sample", "softmax"])

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "train":
        d = vars(args)
        d.pop("cmd", None)
        cfg = TrainConfig(**d)
        train(cfg)
        return

    if args.cmd == "encode":
        encode_tokens(
            data_root=args.data_root,
            ckpt=args.ckpt,
            out_dir=args.out_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            latent_embed_mode=None if getattr(args, "latent_embed_mode", "from_ckpt") == "from_ckpt" else args.latent_embed_mode,
        )
        return

    if args.cmd == "reconstruct":
        reconstruct(
            data_root=args.data_root,
            ckpt=args.ckpt,
            out_dir=args.out_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            latent_embed_mode=None if getattr(args, "latent_embed_mode", "from_ckpt") == "from_ckpt" else args.latent_embed_mode,
        )
        return

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
