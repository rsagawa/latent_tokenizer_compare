#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline B: Discrete Sequence Autoencoder (Discrete Latent Variables) for HumanML3D
===============================================================================

This script provides a "seq2seq-style" discrete bottleneck autoencoder baseline for HumanML3D,
in the spirit of:
  - Kaiser & Bengio (2018) Discrete Autoencoders for Sequence Models
  - Kaiser et al. (2018) Fast Decoding in Sequence Models Using Discrete Latent Variables

Key idea (baseline B vs VQ-VAE baseline A)
-----------------------------------------
Instead of nearest-neighbor vector quantization, we predict a categorical distribution q(z|x)
over a discrete vocabulary for each latent step, sample/argmax discrete IDs, then reconstruct x.

Model (minimal)
---------------
Input motion features x: [T, D] (HumanML3D "new_joint_vecs", typically D=263)
  -> patchify (patch_len P) to [N, P*D]
  -> Transformer encoder (non-causal) to contextual features
  -> logits over discrete vocabulary V: q_logits [N, V]
  -> straight-through Gumbel-Softmax (train) or argmax (encode/recon) to obtain discrete IDs
  -> embedding lookup to latent embeddings [N, d_model]
  -> Transformer decoder (non-causal) to reconstruct patches
  -> unpatchify to x_hat [T, D]

Regularization
--------------
We include a KL-to-uniform term KL(q(z|x) || U) as a simple "rate" regularizer to avoid trivial usage.
This is conceptually aligned with discrete latent variable AE/VAE training, while keeping the
implementation lightweight and stable.

Optional: After training the AE, you can train an autoregressive prior on the discrete IDs
(train_prior) to mimic the "Fast Decoding..." pipeline (train AE -> train latent LM).

HumanML3D expected structure
----------------------------
<data_root>/
  new_joint_vecs/<id>.npy   # [T, D]
  train.txt / val.txt / test.txt

Outputs
-------
<save_dir>/
  config.json
  norm/mean.npy, norm/std.npy (if --normalize and not provided)
  checkpoints/ae_ckpt_last.pt, ae_ckpt_best.pt
  logs/ae_train_log.jsonl
  tokens/<split>/<id>.txt               (encode)
  recon/<split>/<id>.npy                (reconstruct, de-normalized features)
  prior_checkpoints/prior_ckpt_last.pt  (train_prior)
  logs/prior_train_log.jsonl

Examples
--------
Train AE (M2DM-like backbone defaults: 4 layers, 8 heads, d_model=512, V=8192, crop=64):
  python discrete_seq_autoencoder_humanml3d_baseline_b.py train \
    --data_root /path/to/HumanML3D \
    --save_dir  ./runs/baselineB_h3d \
    --normalize \
    --epochs 100 --batch_size 64 --lr 1e-4 \
    --max_motion_len 64 --patch_len 1 \
    --d_model 512 --n_layers 4 --n_heads 8 \
    --vocab_size 8192 \
    --kl_weight 0.1 --kl_anneal_steps 20000 \
    --tau_start 1.0 --tau_end 0.3 --tau_anneal_steps 20000

Encode to discrete IDs:
  python discrete_seq_autoencoder_humanml3d_baseline_b.py encode \
    --data_root /path/to/HumanML3D \
    --save_dir  ./runs/baselineB_h3d \
    --ckpt ./runs/baselineB_h3d/checkpoints/ae_ckpt_best.pt \
    --split test --normalize

Reconstruct:
  python discrete_seq_autoencoder_humanml3d_baseline_b.py reconstruct \
    --data_root /path/to/HumanML3D \
    --save_dir  ./runs/baselineB_h3d \
    --ckpt ./runs/baselineB_h3d/checkpoints/ae_ckpt_best.pt \
    --split test --normalize --save_input

Train AR prior on encoded IDs (optional):
  python discrete_seq_autoencoder_humanml3d_baseline_b.py train_prior \
    --tokens_dir ./runs/baselineB_h3d/tokens/train \
    --save_dir   ./runs/baselineB_h3d \
    --vocab_size 8192 \
    --d_model 512 --n_layers 4 --n_heads 8 \
    --epochs 10 --batch_size 256 --lr 3e-4

Notes
-----
- This is a baseline implementation for controlled comparisons; it is not a faithful reproduction
  of any specific paper codebase.
- With vocab_size=8192 and long sequences, memory usage can be high. Reduce --batch_size if needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Utilities
# -------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def save_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_split_ids(split_file: Path) -> List[str]:
    ids: List[str] = []
    for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Dataset (motion features)
# -------------------------

class HumanML3DMotionDataset(Dataset):
    """
    Loads motion feature sequences from HumanML3D new_joint_vecs/<id>.npy.

    Returns:
      id: str
      x : torch.FloatTensor [T, D] (padded/cropped to max_motion_len)
      mask: torch.BoolTensor [T] (True for valid frames, False for padded)
    """
    def __init__(
        self,
        data_root: Path,
        split: str,
        motion_dir: str = "new_joint_vecs",
        split_file: Optional[Path] = None,
        max_motion_len: int = 64,
        min_motion_len: int = 40,
        random_crop: bool = True,
        normalize: bool = True,
        mean_path: Optional[Path] = None,
        std_path: Optional[Path] = None,
        *,
        missing_ok: bool = True,
        max_resample_tries: int = 20,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.motion_dir = self.data_root / motion_dir
        if split_file is None:
            split_file = self.data_root / f"{split}.txt"
        self.split_file = Path(split_file)

        if not self.motion_dir.is_dir():
            raise FileNotFoundError(f"motion_dir not found: {self.motion_dir}")
        if not self.split_file.is_file():
            raise FileNotFoundError(f"split_file not found: {self.split_file}")

        ids = read_split_ids(self.split_file)
        if len(ids) == 0:
            raise RuntimeError(f"No ids found in split_file: {self.split_file}")

        self.max_motion_len = int(max_motion_len)
        self.min_motion_len = int(min_motion_len)
        self.random_crop = bool(random_crop)
        self.normalize = bool(normalize)
        self.missing_ok = bool(missing_ok)
        self.max_resample_tries = int(max_resample_tries)

        self.mean = None
        self.std = None
        if self.normalize:
            if mean_path is not None and std_path is not None and mean_path.is_file() and std_path.is_file():
                self.mean = np.load(str(mean_path)).astype(np.float32)
                self.std = np.load(str(std_path)).astype(np.float32)
                self.std = np.maximum(self.std, 1e-6)

        # Filter out missing / broken npy files so training won't crash on np.load.
        kept: List[str] = []
        missing = 0
        bad = 0
        feat_dim: Optional[int] = None

        for mid in ids:
            p = self.motion_dir / f"{mid}.npy"
            if not p.is_file():
                missing += 1
                if not self.missing_ok:
                    raise FileNotFoundError(f"Motion npy not found: {p}")
                continue
            try:
                # mmap_mode reads header cheaply for validation
                arr = np.load(str(p), mmap_mode="r")
                if getattr(arr, "ndim", None) != 2:
                    bad += 1
                    continue
                d = int(arr.shape[1])
                if feat_dim is None:
                    feat_dim = d
                elif d != feat_dim:
                    bad += 1
                    continue
                kept.append(mid)
            except Exception:
                bad += 1
                continue

        if (missing > 0 or bad > 0) and self.missing_ok:
            print(
                f"[WARN] {self.split}: skipped {missing} missing and {bad} invalid motion files "
                f"(motion_dir={self.motion_dir})",
                flush=True,
            )

        if len(kept) == 0:
            raise RuntimeError(
                f"No usable motion npy files found for split={self.split}. "
                f"motion_dir={self.motion_dir}, split_file={self.split_file}"
            )

        self.ids = kept
        self.feat_dim = int(feat_dim) if feat_dim is not None else 0
        if self.feat_dim <= 0:
            raise RuntimeError(f"Failed to infer feat_dim for split={self.split} (no valid [T,D] arrays).")

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def __len__(self) -> int:
        return len(self.ids)

    def _safe_load_motion(self, mid: str) -> Optional[np.ndarray]:
        p = self.motion_dir / f"{mid}.npy"
        try:
            arr = np.load(str(p)).astype(np.float32)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        if int(arr.shape[1]) != int(self.feat_dim):
            return None
        return arr

    def __getitem__(self, idx: int):
        # Try the requested index first; if load fails, resample a different id so training continues.
        mid = self.ids[idx]
        arr = self._safe_load_motion(mid)

        if arr is None and self.missing_ok:
            for _ in range(max(1, self.max_resample_tries)):
                mid2 = random.choice(self.ids)
                arr2 = self._safe_load_motion(mid2)
                if arr2 is not None:
                    mid, arr = mid2, arr2
                    break

        if arr is None:
            if not self.missing_ok:
                raise FileNotFoundError(f"Failed to load motion npy for id={mid}")
            # Fallback: return zeros (avoid crashing); mask all True to keep losses finite.
            arr = np.zeros((self.max_motion_len, self.feat_dim), dtype=np.float32)

        T0 = int(arr.shape[0])

        # Enforce minimum length by padding (mask False for padded frames)
        if T0 < self.min_motion_len:
            pad = self.min_motion_len - T0
            arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
            T0 = int(arr.shape[0])

        # Crop/pad to max_motion_len
        if T0 > self.max_motion_len:
            if self.random_crop:
                start = random.randint(0, T0 - self.max_motion_len)
            else:
                start = 0
            arr = arr[start:start + self.max_motion_len]
            valid_len = self.max_motion_len
        elif T0 < self.max_motion_len:
            valid_len = T0
            pad = self.max_motion_len - T0
            arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
        else:
            valid_len = self.max_motion_len

        # Normalize
        if self.normalize and self.mean is not None and self.std is not None:
            arr = (arr - self.mean) / self.std

        x = torch.from_numpy(arr)  # [T,D]
        mask = torch.zeros(self.max_motion_len, dtype=torch.bool)
        mask[:valid_len] = True
        return mid, x, mask


def collate_motion(batch):
    ids = [b[0] for b in batch]
    x = torch.stack([b[1] for b in batch], dim=0)      # [B,T,D]
    mask = torch.stack([b[2] for b in batch], dim=0)   # [B,T]
    return ids, x, mask


def compute_mean_std(
    data_root: Path,
    split: str,
    motion_dir: str,
    split_file: Optional[Path],
    max_files: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = HumanML3DMotionDataset(
        data_root=Path(data_root),
        split=split,
        motion_dir=motion_dir,
        split_file=split_file,
        random_crop=False,
        normalize=False,
        max_motion_len=10**9,  # no crop during stats
        min_motion_len=1,
        missing_ok=True,
    )
    ids = ds.ids
    if max_files > 0 and len(ids) > max_files:
        rng = random.Random(seed)
        ids = rng.sample(ids, max_files)

    sums = None
    sqs = None
    count = 0
    skipped = 0

    for mid in ids:
        p = ds.motion_dir / f"{mid}.npy"
        try:
            arr = np.load(str(p)).astype(np.float32)
        except Exception:
            skipped += 1
            continue
        if arr.ndim != 2 or int(arr.shape[1]) != int(ds.feat_dim):
            skipped += 1
            continue

        if sums is None:
            sums = arr.sum(axis=0)
            sqs = (arr ** 2).sum(axis=0)
        else:
            sums += arr.sum(axis=0)
            sqs += (arr ** 2).sum(axis=0)
        count += arr.shape[0]

    if skipped > 0:
        print(f"[WARN] compute_mean_std: skipped {skipped} files due to load/shape errors.", flush=True)

    if sums is None or sqs is None or count == 0:
        raise RuntimeError("Failed to compute mean/std (no usable data?)")

    mean = sums / count
    var = sqs / count - mean ** 2
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# -------------------------
# Model components
# -------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        return x + self.pe[:n].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class DiscreteSeqAutoencoder(nn.Module):
    """
    Transformer encoder -> categorical logits -> (ST gumbel / argmax) -> embedding -> Transformer decoder -> recon.

    Optionally tie encoder logits weight and embedding weight (like LM tied embeddings),
    reducing parameter count and aligning more closely with VQ-like parameterization.
    """
    def __init__(
        self,
        feat_dim: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        tie_logits_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)

        self.in_proj = nn.Linear(self.patch_len * self.feat_dim, self.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=8192)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ff_mult) * self.d_model,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.code_embed = nn.Embedding(self.vocab_size, self.d_model)

        self.to_logits = nn.Linear(self.d_model, self.vocab_size, bias=True)
        if tie_logits_embedding:
            # Tie weights: to_logits.weight is [V, D] same as code_embed.weight
            self.to_logits.weight = self.code_embed.weight

        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ff_mult) * self.d_model,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(n_layers))

        self.out_proj = nn.Linear(self.d_model, self.patch_len * self.feat_dim)

    def patchify(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        x: [B,T,D], mask: [B,T] bool
        returns:
          x_patch: [B,N,PD]
          patch_mask: [B,N] bool
          x_pad: [B,T_pad,D]
          mask_pad: [B,T_pad] bool
          T_pad: int
        """
        B, T, D = x.shape
        P = self.patch_len
        pad = (P - (T % P)) % P
        if pad > 0:
            x = torch.cat([x, torch.zeros(B, pad, D, device=x.device, dtype=x.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros(B, pad, device=x.device, dtype=torch.bool)], dim=1)
        T_pad = x.size(1)
        N = T_pad // P
        x_patch = x.view(B, N, P * D)
        patch_mask = mask.view(B, N, P).any(dim=2)
        return x_patch, patch_mask, x, mask, T_pad

    def unpatchify(self, y_patch: torch.Tensor, T_pad: int) -> torch.Tensor:
        B, N, PD = y_patch.shape
        P = self.patch_len
        D = self.feat_dim
        y = y_patch.view(B, N, P, D).reshape(B, N * P, D)
        if y.size(1) != T_pad:
            y = y[:, :T_pad, :]
        return y

    def encode_logits(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        x_patch, patch_mask, x_pad, mask_pad, T_pad = self.patchify(x, mask)
        h = self.in_proj(x_patch)
        h = self.pos_enc(h)
        key_padding = ~patch_mask
        h = self.encoder(h, src_key_padding_mask=key_padding)
        logits = self.to_logits(h)  # [B,N,V]
        return logits, patch_mask, x_pad, mask_pad, T_pad

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        mode: str,
        tau: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        mode:
          - "train": use straight-through Gumbel-Softmax sampling
          - "argmax": use argmax codes (deterministic)
          - "soft": use soft expected embedding (no discrete IDs)

        returns:
          x_hat: [B,T_pad,D]
          logits: [B,N,V]
          indices: [B,N] (argmax indices, useful for stats)
          losses: dict (empty here; computed outside)
          stats: dict (entropy/perplexity/usage)
        """
        logits, patch_mask, x_pad, mask_pad, T_pad = self.encode_logits(x, mask)  # logits [B,N,V]
        B, N, V = logits.shape

        # Posterior distribution q
        q = F.softmax(logits, dim=-1)  # [B,N,V]

        # Discretization / embedding
        if mode == "train":
            # ST Gumbel-Softmax: returns [B,N,V]
            z = F.gumbel_softmax(logits, tau=max(float(tau), 1e-6), hard=bool(hard), dim=-1)
            # embedding as expected / one-hot times embedding matrix
            z_emb = torch.einsum("bnv,vd->bnd", z, self.code_embed.weight)
        elif mode == "soft":
            z_emb = torch.einsum("bnv,vd->bnd", q, self.code_embed.weight)
        elif mode == "argmax":
            idx = torch.argmax(logits, dim=-1)  # [B,N]
            z_emb = self.code_embed(idx)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        idx_argmax = torch.argmax(logits, dim=-1)  # [B,N]

        # Decode
        z_emb = self.pos_enc(z_emb)
        key_padding = ~patch_mask
        y = self.decoder(z_emb, src_key_padding_mask=key_padding)
        y_patch = self.out_proj(y)  # [B,N,PD]
        x_hat = self.unpatchify(y_patch, T_pad)

        # Stats (mask-aware)
        with torch.no_grad():
            eps = 1e-12
            # Per-position entropy
            ent = -(q * (q.clamp_min(eps).log())).sum(dim=-1)  # [B,N]
            ent_mean = ent[patch_mask].mean() if patch_mask.any() else ent.mean()

            # Marginal entropy over (B,N) valid positions
            q_flat = q[patch_mask] if patch_mask.any() else q.reshape(-1, V)
            p_marg = q_flat.mean(dim=0).clamp_min(eps)
            p_marg = p_marg / p_marg.sum()
            marg_ent = -(p_marg * p_marg.log()).sum()
            marg_ppl = torch.exp(marg_ent)

            # Usage from argmax
            idx_flat = idx_argmax[patch_mask] if patch_mask.any() else idx_argmax.reshape(-1)
            hist = torch.bincount(idx_flat, minlength=V).float()
            usage_frac = (hist > 0).float().mean()

            stats = {
                "pos_entropy": float(ent_mean.cpu().item()),
                "marg_entropy": float(marg_ent.cpu().item()),
                "marg_perplexity": float(marg_ppl.cpu().item()),
                "usage_frac_batch": float(usage_frac.cpu().item()),
                "num_latent_tokens": float(N),
            }

        return x_hat, logits, idx_argmax, {}, stats


def masked_mse(x_hat: torch.Tensor, x_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x_hat, x_true: [B,T,D]
    mask: [B,T] bool (valid frames)
    """
    mse = (x_hat - x_true).pow(2).mean(dim=-1)  # [B,T]
    return mse[mask].mean()


def kl_to_uniform(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    KL(q || U) where q=softmax(logits).
    logits: [B,N,V]
    mask: [B,N] bool (valid latent positions)
    """
    eps = 1e-12
    q = F.softmax(logits, dim=-1)
    logq = (q.clamp_min(eps)).log()
    V = logits.size(-1)
    kl = (q * (logq + math.log(V))).sum(dim=-1)  # [B,N]
    if mask.any():
        return kl[mask].mean()
    return kl.mean()


def anneal_linear(step: int, start: float, end: float, steps: int) -> float:
    if steps <= 0:
        return float(end)
    t = min(max(step, 0), steps) / float(steps)
    return float(start + (end - start) * t)


# -------------------------
# Training
# -------------------------

@dataclass
class AETrainConfig:
    data_root: str
    save_dir: str
    motion_dir: str = "new_joint_vecs"

    # model
    patch_len: int = 1
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    vocab_size: int = 8192
    dropout: float = 0.1
    ff_mult: int = 4
    tie_logits_embedding: bool = True

    # data
    max_motion_len: int = 64
    min_motion_len: int = 40
    normalize: bool = True
    mean_path: str = ""
    std_path: str = ""

    # loss
    kl_weight: float = 0.1
    kl_anneal_steps: int = 20000
    tau_start: float = 1.0
    tau_end: float = 0.3
    tau_anneal_steps: int = 20000

    # optimization
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # runtime
    num_workers: int = 4
    device: str = ""
    seed: int = 1234
    log_every: int = 50
    eval_every: int = 1
    save_every: int = 1
    amp: bool = False


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, scaler, step: int, epoch: int, cfg: AETrainConfig, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "cfg": asdict(cfg),
        "mean": None if mean is None else mean.astype(np.float32),
        "std": None if std is None else std.astype(np.float32),
    }
    torch.save(ckpt, str(path))


def load_checkpoint(path: Path, model: nn.Module, device: torch.device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


@torch.no_grad()
def evaluate_ae(model: DiscreteSeqAutoencoder, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    agg = {"recon_mse": 0.0, "kl_u": 0.0, "pos_entropy": 0.0, "marg_perplexity": 0.0, "usage_frac_batch": 0.0, "count": 0}
    for ids, x, mask in loader:
        x = x.to(device)
        mask = mask.to(device)
        x_hat, logits, _, _, stats = model(x, mask, mode="argmax")
        # latent mask
        # compute patch mask by reusing patchify logic (cheap)
        _, patch_mask, x_pad, mask_pad, _ = model.patchify(x, mask)
        recon = masked_mse(x_hat, x_pad, mask_pad)
        kl_u = kl_to_uniform(logits, patch_mask)

        bs = x.size(0)
        agg["recon_mse"] += float(recon.cpu().item()) * bs
        agg["kl_u"] += float(kl_u.cpu().item()) * bs
        agg["pos_entropy"] += float(stats["pos_entropy"]) * bs
        agg["marg_perplexity"] += float(stats["marg_perplexity"]) * bs
        agg["usage_frac_batch"] += float(stats["usage_frac_batch"]) * bs
        agg["count"] += bs

    c = max(1, int(agg["count"]))
    return {k: v / c for k, v in agg.items() if k != "count"}


def train_ae(cfg: AETrainConfig) -> None:
    seed_everything(cfg.seed)

    save_dir = Path(cfg.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_path = save_dir / "logs" / "ae_train_log.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device_str = cfg.device.strip()
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    data_root = Path(cfg.data_root)
    mean_path = Path(cfg.mean_path) if cfg.mean_path else None
    std_path = Path(cfg.std_path) if cfg.std_path else None

    ds_train = HumanML3DMotionDataset(
        data_root=data_root,
        split="train",
        motion_dir=cfg.motion_dir,
        max_motion_len=cfg.max_motion_len,
        min_motion_len=cfg.min_motion_len,
        random_crop=True,
        normalize=cfg.normalize,
        mean_path=mean_path,
        std_path=std_path,
    )
    ds_val = HumanML3DMotionDataset(
        data_root=data_root,
        split="val",
        motion_dir=cfg.motion_dir,
        max_motion_len=cfg.max_motion_len,
        min_motion_len=cfg.min_motion_len,
        random_crop=False,
        normalize=cfg.normalize,
        mean_path=mean_path,
        std_path=std_path,
    )

    # Norm
    mean = ds_train.mean
    std = ds_train.std
    if cfg.normalize and (mean is None or std is None):
        print("[INFO] Computing mean/std from TRAIN split ...", flush=True)
        mean, std = compute_mean_std(
            data_root=data_root,
            split="train",
            motion_dir=cfg.motion_dir,
            split_file=None,
            max_files=0,
            seed=cfg.seed,
        )
        ds_train.set_norm(mean, std)
        ds_val.set_norm(mean, std)
        (save_dir / "norm").mkdir(parents=True, exist_ok=True)
        np.save(str(save_dir / "norm" / "mean.npy"), mean)
        np.save(str(save_dir / "norm" / "std.npy"), std)
        print(f"[INFO] Saved mean/std to: {save_dir / 'norm'}", flush=True)

    model = DiscreteSeqAutoencoder(
        feat_dim=ds_train.feat_dim,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        vocab_size=cfg.vocab_size,
        dropout=cfg.dropout,
        ff_mult=cfg.ff_mult,
        tie_logits_embedding=cfg.tie_logits_embedding,
    ).to(device)

    print(f"[INFO] AE params: {count_parameters(model):,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = bool(cfg.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_motion)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_motion)

    # Save config
    (save_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    best_val = float("inf")
    step = 0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        for it, (ids, x, mask) in enumerate(dl_train, start=1):
            x = x.to(device)
            mask = mask.to(device)

            # anneal schedules
            tau = anneal_linear(step, cfg.tau_start, cfg.tau_end, cfg.tau_anneal_steps)
            kl_w = anneal_linear(step, 0.0, cfg.kl_weight, cfg.kl_anneal_steps)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                x_hat, logits, _, _, stats = model(x, mask, mode="train", tau=tau, hard=True)
                # Need padded x/mask for recon loss
                _, patch_mask, x_pad, mask_pad, _ = model.patchify(x, mask)
                recon = masked_mse(x_hat, x_pad, mask_pad)
                kl_u = kl_to_uniform(logits, patch_mask)
                loss = recon + kl_w * kl_u

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            step += 1

            if step % cfg.log_every == 0:
                row = {
                    "time": time.time(),
                    "epoch": epoch,
                    "iter": it,
                    "step": step,
                    "lr": float(optim.param_groups[0]["lr"]),
                    "loss_total": float(loss.detach().cpu().item()),
                    "loss_recon": float(recon.detach().cpu().item()),
                    "loss_kl_u": float(kl_u.detach().cpu().item()),
                    "kl_weight": float(kl_w),
                    "tau": float(tau),
                    **stats,
                }
                save_jsonl(log_path, row)
                print(
                    f"[train] ep {epoch:03d} it {it:04d} step {step:07d} "
                    f"loss={row['loss_total']:.4f} recon={row['loss_recon']:.4f} kl={row['loss_kl_u']:.4f} "
                    f"klw={row['kl_weight']:.3f} tau={row['tau']:.3f} "
                    f"ppl={row['marg_perplexity']:.1f} usage={row['usage_frac_batch']:.3f}",
                    flush=True,
                )

        # Eval
        if epoch % cfg.eval_every == 0:
            val = evaluate_ae(model, dl_val, device)
            row = {"time": time.time(), "epoch": epoch, "step": step, "split": "val", **val}
            save_jsonl(log_path, row)
            print(
                f"[val] ep {epoch:03d} recon_mse={val['recon_mse']:.6f} kl_u={val['kl_u']:.6f} "
                f"ppl={val['marg_perplexity']:.1f} usage={val['usage_frac_batch']:.3f} "
                f"elapsed={human_time(time.time()-t_start)}",
                flush=True,
            )
            if val["recon_mse"] < best_val:
                best_val = val["recon_mse"]
                save_checkpoint(ckpt_dir / "ae_ckpt_best.pt", model, optim, scaler, step, epoch, cfg, mean, std)

        if epoch % cfg.save_every == 0:
            save_checkpoint(ckpt_dir / "ae_ckpt_last.pt", model, optim, scaler, step, epoch, cfg, mean, std)

        print(f"[epoch] {epoch:03d} finished in {human_time(time.time()-t0)}", flush=True)

    print(f"[done] AE training finished. best_val_recon_mse={best_val:.6f}", flush=True)


# -------------------------
# Encode / Reconstruct
# -------------------------

@torch.no_grad()
def run_encode(
    data_root: Path,
    split: str,
    motion_dir: str,
    ckpt_path: Path,
    save_dir: Path,
    out_tokens_dir: Optional[Path],
    batch_size: int,
    num_workers: int,
    device: str,
    normalize: bool,
) -> None:
    device_t = torch.device(device)
    cfg_path = save_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found in save_dir: {save_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    mean = std = None
    if normalize:
        mean_p = save_dir / "norm" / "mean.npy"
        std_p = save_dir / "norm" / "std.npy"
        if mean_p.is_file() and std_p.is_file():
            mean = np.load(str(mean_p)).astype(np.float32)
            std = np.load(str(std_p)).astype(np.float32)

    ds = HumanML3DMotionDataset(
        data_root=data_root,
        split=split,
        motion_dir=motion_dir,
        max_motion_len=int(cfg["max_motion_len"]),
        min_motion_len=int(cfg["min_motion_len"]),
        random_crop=False,
        normalize=normalize,
    )
    if normalize and mean is not None and std is not None:
        ds.set_norm(mean, std)

    model = DiscreteSeqAutoencoder(
        feat_dim=ds.feat_dim,
        patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        vocab_size=int(cfg["vocab_size"]),
        dropout=float(cfg["dropout"]),
        ff_mult=int(cfg["ff_mult"]),
        tie_logits_embedding=bool(cfg.get("tie_logits_embedding", True)),
    ).to(device_t)

    _ = load_checkpoint(ckpt_path, model, device_t)
    model.eval()

    print(f"[INFO] AE params: {count_parameters(model):,}", flush=True)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_motion)

    if out_tokens_dir is None:
        out_tokens_dir = save_dir / "tokens"
    out_tokens_dir = out_tokens_dir / split
    out_tokens_dir.mkdir(parents=True, exist_ok=True)

    for ids, x, mask in dl:
        x = x.to(device_t)
        mask = mask.to(device_t)
        _, logits, idx, _, _ = model(x, mask, mode="argmax")
        # Save per sequence (idx includes padded positions; we remove using patch_mask)
        x_patch, patch_mask, _, _, _ = model.patchify(x, mask)
        idx_np = idx.detach().cpu().numpy()
        pm_np = patch_mask.detach().cpu().numpy()
        for b, mid in enumerate(ids):
            seq = idx_np[b][pm_np[b]].tolist()
            (out_tokens_dir / f"{mid}.txt").write_text(" ".join(map(str, seq)) + "\n", encoding="utf-8")

    print(f"[done] tokens saved to: {out_tokens_dir}", flush=True)


@torch.no_grad()
def run_reconstruct(
    data_root: Path,
    split: str,
    motion_dir: str,
    ckpt_path: Path,
    save_dir: Path,
    out_recon_dir: Optional[Path],
    batch_size: int,
    num_workers: int,
    device: str,
    normalize: bool,
    save_input: bool,
) -> None:
    device_t = torch.device(device)
    cfg_path = save_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found in save_dir: {save_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    mean = std = None
    if normalize:
        mean_p = save_dir / "norm" / "mean.npy"
        std_p = save_dir / "norm" / "std.npy"
        if mean_p.is_file() and std_p.is_file():
            mean = np.load(str(mean_p)).astype(np.float32)
            std = np.load(str(std_p)).astype(np.float32)

    ds = HumanML3DMotionDataset(
        data_root=data_root,
        split=split,
        motion_dir=motion_dir,
        max_motion_len=int(cfg["max_motion_len"]),
        min_motion_len=int(cfg["min_motion_len"]),
        random_crop=False,
        normalize=normalize,
    )
    if normalize and mean is not None and std is not None:
        ds.set_norm(mean, std)

    model = DiscreteSeqAutoencoder(
        feat_dim=ds.feat_dim,
        patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        vocab_size=int(cfg["vocab_size"]),
        dropout=float(cfg["dropout"]),
        ff_mult=int(cfg["ff_mult"]),
        tie_logits_embedding=bool(cfg.get("tie_logits_embedding", True)),
    ).to(device_t)
    _ = load_checkpoint(ckpt_path, model, device_t)
    model.eval()

    print(f"[INFO] AE params: {count_parameters(model):,}", flush=True)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_motion)

    if out_recon_dir is None:
        out_recon_dir = save_dir / "recon"
    out_recon_dir = out_recon_dir / split
    out_recon_dir.mkdir(parents=True, exist_ok=True)
    if save_input:
        (out_recon_dir / "_input").mkdir(parents=True, exist_ok=True)

    for ids, x, mask in dl:
        x = x.to(device_t)
        mask = mask.to(device_t)
        x_hat, _, _, _, _ = model(x, mask, mode="argmax")

        # de-normalize if required
        x_hat_np = x_hat.detach().cpu().numpy()
        # also need x_pad/mask_pad for saving aligned length
        _, _, x_pad, mask_pad, _ = model.patchify(x, mask)
        x_pad_np = x_pad.detach().cpu().numpy()
        mask_pad_np = mask_pad.detach().cpu().numpy()

        if normalize and mean is not None and std is not None:
            x_hat_np = x_hat_np * std[None, None, :] + mean[None, None, :]
            x_pad_np = x_pad_np * std[None, None, :] + mean[None, None, :]

        for b, mid in enumerate(ids):
            # Save only valid frames
            valid = mask_pad_np[b].astype(bool)
            np.save(str(out_recon_dir / f"{mid}.npy"), x_hat_np[b][valid].astype(np.float32))
            if save_input:
                np.save(str(out_recon_dir / "_input" / f"{mid}.npy"), x_pad_np[b][valid].astype(np.float32))

    print(f"[done] recon saved to: {out_recon_dir}", flush=True)


# -------------------------
# Optional: Train AR prior on discrete IDs
# -------------------------

class TokenDataset(Dataset):
    """
    Loads token sequences from a directory of <id>.txt (space-separated ints).
    """
    def __init__(self, tokens_dir: Path, max_len: int = 0):
        super().__init__()
        self.tokens_dir = Path(tokens_dir)
        if not self.tokens_dir.is_dir():
            raise FileNotFoundError(f"tokens_dir not found: {self.tokens_dir}")
        self.files = sorted(self.tokens_dir.glob("*.txt"))
        if not self.files:
            raise RuntimeError(f"No .txt token files found in: {self.tokens_dir}")
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        mid = path.stem
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        toks = [int(x) for x in text.split()] if text else []
        if self.max_len > 0:
            toks = toks[: self.max_len]
        return mid, torch.tensor(toks, dtype=torch.long)


def collate_tokens(batch):
    ids = [b[0] for b in batch]
    seqs = [b[1] for b in batch]
    max_len = max(int(s.numel()) for s in seqs)
    # Pad with -1
    x = torch.full((len(seqs), max_len), -1, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        if n > 0:
            x[i, :n] = s
            mask[i, :n] = True
    return ids, x, mask


class LatentARPrior(nn.Module):
    """
    Causal Transformer LM over token IDs.
    Predicts token at position t given previous tokens.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.bos_id = self.vocab_size  # extra token
        self.embed = nn.Embedding(self.vocab_size + 1, self.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=8192)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ff_mult) * self.d_model,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.to_logits = nn.Linear(self.d_model, self.vocab_size, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B,L] token ids (0..V-1), padded with -1
        mask: [B,L] bool (True for valid)
        returns:
          logits: [B,L,V] for predicting each token position (including first from BOS)
        """
        B, L = x.shape
        # Build input with BOS shift-right: inp[0]=BOS, inp[t]=x[t-1]
        inp = torch.full((B, L), self.bos_id, dtype=torch.long, device=x.device)
        if L > 1:
            inp[:, 1:] = torch.where(mask[:, :-1], x[:, :-1], torch.full_like(x[:, :-1], self.bos_id))

        h = self.embed(inp)
        h = self.pos_enc(h)

        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        # Padding mask: True for pad positions
        key_padding = ~mask

        h = self.tr(h, mask=causal, src_key_padding_mask=key_padding)
        logits = self.to_logits(h)
        return logits


@dataclass
class PriorTrainConfig:
    tokens_dir: str
    save_dir: str
    vocab_size: int = 8192
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    ff_mult: int = 4
    max_len: int = 0

    epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    num_workers: int = 4
    device: str = ""
    seed: int = 1234
    log_every: int = 50
    save_every: int = 1
    amp: bool = False


def train_prior(cfg: PriorTrainConfig) -> None:
    seed_everything(cfg.seed)
    save_dir = Path(cfg.save_dir)
    ckpt_dir = save_dir / "prior_checkpoints"
    log_path = save_dir / "logs" / "prior_train_log.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device_str = cfg.device.strip()
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    ds = TokenDataset(Path(cfg.tokens_dir), max_len=int(cfg.max_len))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_tokens)

    model = LatentARPrior(
        vocab_size=int(cfg.vocab_size),
        d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads),
        n_layers=int(cfg.n_layers),
        dropout=float(cfg.dropout),
        ff_mult=int(cfg.ff_mult),
    ).to(device)

    print(f"[INFO] Prior params: {count_parameters(model):,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = bool(cfg.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        for it, (ids, x, mask) in enumerate(dl, start=1):
            x = x.to(device)       # [B,L], -1 padded
            mask = mask.to(device) # [B,L]

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x, mask)  # [B,L,V]
                # CE only on valid positions
                V = logits.size(-1)
                tgt = torch.where(mask, x, torch.zeros_like(x))  # dummy targets for pad
                loss_flat = F.cross_entropy(logits.view(-1, V), tgt.view(-1), reduction="none")
                loss = (loss_flat.view_as(mask)[mask]).mean()

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            step += 1
            if step % cfg.log_every == 0:
                row = {
                    "time": time.time(),
                    "epoch": epoch,
                    "iter": it,
                    "step": step,
                    "lr": float(optim.param_groups[0]["lr"]),
                    "loss_nll": float(loss.detach().cpu().item()),
                }
                save_jsonl(log_path, row)
                print(
                    f"[prior] ep {epoch:03d} it {it:04d} step {step:07d} nll={row['loss_nll']:.4f} "
                    f"elapsed={human_time(time.time()-t_start)}",
                    flush=True,
                )

        if epoch % cfg.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "cfg": asdict(cfg),
            }
            torch.save(ckpt, str(ckpt_dir / "prior_ckpt_last.pt"))

        print(f"[epoch] prior {epoch:03d} finished in {human_time(time.time()-t0)}", flush=True)

    print("[done] prior training finished.", flush=True)


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    # train AE
    ap_t = sub.add_parser("train")
    ap_t.add_argument("--data_root", type=str, required=True)
    ap_t.add_argument("--save_dir", type=str, required=True)
    ap_t.add_argument("--motion_dir", type=str, default="new_joint_vecs")

    ap_t.add_argument("--patch_len", type=int, default=1)
    ap_t.add_argument("--d_model", type=int, default=512)
    ap_t.add_argument("--n_heads", type=int, default=8)
    ap_t.add_argument("--n_layers", type=int, default=4)
    ap_t.add_argument("--vocab_size", type=int, default=8192)
    ap_t.add_argument("--dropout", type=float, default=0.1)
    ap_t.add_argument("--ff_mult", type=int, default=4)
    ap_t.add_argument("--tie_logits_embedding", action="store_true", help="Tie logits head weights to code embedding (default: enabled).")
    ap_t.add_argument("--no_tie_logits_embedding", action="store_true", help="Disable tied weights.")

    ap_t.add_argument("--max_motion_len", type=int, default=64)
    ap_t.add_argument("--min_motion_len", type=int, default=40)
    ap_t.add_argument("--normalize", action="store_true")
    ap_t.add_argument("--mean_path", type=str, default="")
    ap_t.add_argument("--std_path", type=str, default="")

    ap_t.add_argument("--kl_weight", type=float, default=0.1)
    ap_t.add_argument("--kl_anneal_steps", type=int, default=20000)
    ap_t.add_argument("--tau_start", type=float, default=1.0)
    ap_t.add_argument("--tau_end", type=float, default=0.3)
    ap_t.add_argument("--tau_anneal_steps", type=int, default=20000)

    ap_t.add_argument("--epochs", type=int, default=100)
    ap_t.add_argument("--batch_size", type=int, default=64)
    ap_t.add_argument("--lr", type=float, default=1e-4)
    ap_t.add_argument("--weight_decay", type=float, default=1e-4)
    ap_t.add_argument("--grad_clip", type=float, default=1.0)

    ap_t.add_argument("--num_workers", type=int, default=4)
    ap_t.add_argument("--device", type=str, default="")
    ap_t.add_argument("--seed", type=int, default=1234)
    ap_t.add_argument("--log_every", type=int, default=50)
    ap_t.add_argument("--eval_every", type=int, default=1)
    ap_t.add_argument("--save_every", type=int, default=1)
    ap_t.add_argument("--amp", action="store_true")

    # encode
    ap_e = sub.add_parser("encode")
    ap_e.add_argument("--data_root", type=str, required=True)
    ap_e.add_argument("--save_dir", type=str, required=True)
    ap_e.add_argument("--ckpt", type=str, required=True)
    ap_e.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap_e.add_argument("--motion_dir", type=str, default="new_joint_vecs")
    ap_e.add_argument("--out_tokens_dir", type=str, default="")
    ap_e.add_argument("--batch_size", type=int, default=128)
    ap_e.add_argument("--num_workers", type=int, default=4)
    ap_e.add_argument("--device", type=str, default="")
    ap_e.add_argument("--normalize", action="store_true")

    # reconstruct
    ap_r = sub.add_parser("reconstruct")
    ap_r.add_argument("--data_root", type=str, required=True)
    ap_r.add_argument("--save_dir", type=str, required=True)
    ap_r.add_argument("--ckpt", type=str, required=True)
    ap_r.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap_r.add_argument("--motion_dir", type=str, default="new_joint_vecs")
    ap_r.add_argument("--out_recon_dir", type=str, default="")
    ap_r.add_argument("--batch_size", type=int, default=128)
    ap_r.add_argument("--num_workers", type=int, default=4)
    ap_r.add_argument("--device", type=str, default="")
    ap_r.add_argument("--normalize", action="store_true")
    ap_r.add_argument("--save_input", action="store_true")

    # train prior
    ap_p = sub.add_parser("train_prior")
    ap_p.add_argument("--tokens_dir", type=str, required=True, help="Directory containing *.txt token sequences (e.g., <save_dir>/tokens/train)")
    ap_p.add_argument("--save_dir", type=str, required=True)
    ap_p.add_argument("--vocab_size", type=int, default=8192)
    ap_p.add_argument("--d_model", type=int, default=512)
    ap_p.add_argument("--n_heads", type=int, default=8)
    ap_p.add_argument("--n_layers", type=int, default=4)
    ap_p.add_argument("--dropout", type=float, default=0.1)
    ap_p.add_argument("--ff_mult", type=int, default=4)
    ap_p.add_argument("--max_len", type=int, default=0)

    ap_p.add_argument("--epochs", type=int, default=10)
    ap_p.add_argument("--batch_size", type=int, default=256)
    ap_p.add_argument("--lr", type=float, default=3e-4)
    ap_p.add_argument("--weight_decay", type=float, default=1e-4)
    ap_p.add_argument("--grad_clip", type=float, default=1.0)

    ap_p.add_argument("--num_workers", type=int, default=4)
    ap_p.add_argument("--device", type=str, default="")
    ap_p.add_argument("--seed", type=int, default=1234)
    ap_p.add_argument("--log_every", type=int, default=50)
    ap_p.add_argument("--save_every", type=int, default=1)
    ap_p.add_argument("--amp", action="store_true")

    args = ap.parse_args()

    if args.command == "train":
        tie = True
        if args.no_tie_logits_embedding:
            tie = False
        if args.tie_logits_embedding:
            tie = True

        cfg = AETrainConfig(
            data_root=args.data_root,
            save_dir=args.save_dir,
            motion_dir=args.motion_dir,
            patch_len=args.patch_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            vocab_size=args.vocab_size,
            dropout=args.dropout,
            ff_mult=args.ff_mult,
            tie_logits_embedding=tie,
            max_motion_len=args.max_motion_len,
            min_motion_len=args.min_motion_len,
            normalize=bool(args.normalize),
            mean_path=args.mean_path,
            std_path=args.std_path,
            kl_weight=args.kl_weight,
            kl_anneal_steps=args.kl_anneal_steps,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_anneal_steps=args.tau_anneal_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            eval_every=args.eval_every,
            save_every=args.save_every,
            amp=bool(args.amp),
        )
        train_ae(cfg)
        return

    if args.command in ("encode", "reconstruct"):
        device = args.device.strip()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        save_dir = Path(args.save_dir)
        ckpt = Path(args.ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"ckpt not found: {ckpt}")

        if args.command == "encode":
            out_tokens_dir = Path(args.out_tokens_dir) if args.out_tokens_dir else None
            run_encode(
                data_root=Path(args.data_root),
                split=args.split,
                motion_dir=args.motion_dir,
                ckpt_path=ckpt,
                save_dir=save_dir,
                out_tokens_dir=out_tokens_dir,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
                normalize=bool(args.normalize),
            )
            return

        if args.command == "reconstruct":
            out_recon_dir = Path(args.out_recon_dir) if args.out_recon_dir else None
            run_reconstruct(
                data_root=Path(args.data_root),
                split=args.split,
                motion_dir=args.motion_dir,
                ckpt_path=ckpt,
                save_dir=save_dir,
                out_recon_dir=out_recon_dir,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
                normalize=bool(args.normalize),
                save_input=bool(args.save_input),
            )
            return

    if args.command == "train_prior":
        device = args.device.strip()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = PriorTrainConfig(
            tokens_dir=args.tokens_dir,
            save_dir=args.save_dir,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            ff_mult=args.ff_mult,
            max_len=args.max_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
            log_every=args.log_every,
            save_every=args.save_every,
            amp=bool(args.amp),
        )
        train_prior(cfg)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
