#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer VQ-VAE for HumanML3D (motion tokenizer baseline)
============================================================

Purpose
-------
A minimal, reproducible baseline for "context-aware" (Transformer-based) VQ-VAE motion tokenization
on HumanML3D. This script can:

  (1) train        : train a Transformer VQ-VAE on HumanML3D motions (new_joint_vecs/*.npy)
  (2) encode       : encode motions into discrete codebook indices (token IDs)
  (3) reconstruct  : reconstruct motions from input motions (encode->decode) and optionally save .npy

This is intended as a baseline "Transformer-VQ-VAE tokenizer" (baseline A),
not an exact reproduction of any specific paper implementation.

Assumed HumanML3D structure
---------------------------
<data_root>/
  new_joint_vecs/   # motion features, each <id>.npy is [T, D] (often D=263)
  train.txt / val.txt / test.txt   # list of <id> (no extension)

If your dataset is arranged differently, use --motion_dir or --split_file.

Outputs
-------
<save_dir>/
  checkpoints/ckpt_last.pt, ckpt_best.pt
  logs/train_log.jsonl
  tokens/<split>/<id>.txt          # when mode=encode
  recon/<split>/<id>.npy           # when mode=reconstruct (feature reconstruction)

Usage examples
--------------
Train:
  python transformer_vqvae_humanml3d.py train \
    --data_root /path/to/HumanML3D \
    --save_dir  ./runs/tvqvae_h3d \
    --epochs 100 --batch_size 64 --lr 1e-4 \
    --patch_len 1 --d_model 512 --n_layers 4 --n_heads 8 \
    --codebook_size 8192 --beta 0.25

Encode (tokenize):
  python transformer_vqvae_humanml3d.py encode \
    --data_root /path/to/HumanML3D \
    --save_dir  ./runs/tvqvae_h3d \
    --ckpt ./runs/tvqvae_h3d/checkpoints/ckpt_best.pt \
    --split test --out_tokens_dir ./runs/tvqvae_h3d/tokens

Reconstruct:
  python transformer_vqvae_humanml3d.py reconstruct \
    --data_root /path/to/HumanML3D \
    --save_dir  ./runs/tvqvae_h3d \
    --ckpt ./runs/tvqvae_h3d/checkpoints/ckpt_best.pt \
    --split test --out_recon_dir ./runs/tvqvae_h3d/recon

Notes
-----
- This operates on HumanML3D motion feature vectors (e.g., 263-D). It does not require text.
- The encoder is Transformer + patching (temporal grouping). patch_len controls compression ratio:
    N_tokens ≈ ceil(T / patch_len)
- For a fair comparison with other tokenizers, keep patch_len, codebook_size, and evaluation protocol consistent.
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
from typing import Dict, List, Optional, Tuple

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


def read_split_ids(split_file: Path) -> List[str]:
    ids: List[str] = []
    for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids


def save_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def human_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


# -------------------------
# Dataset
# -------------------------

class HumanML3DMotionDataset(Dataset):
    """
    Loads motion feature sequences from HumanML3D new_joint_vecs/<id>.npy.

    Returns:
      id: str
      x : torch.FloatTensor [T, D]
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

        self.ids = read_split_ids(self.split_file)
        if len(self.ids) == 0:
            raise RuntimeError(f"No ids found in split_file: {self.split_file}")

        # Filter out missing motion files upfront so the training/encoding loops don't crash.
        # (Files can be missing due to dataset download issues, split mismatch, etc.)
        orig_n = len(self.ids)
        kept: List[str] = []
        missing: List[str] = []
        for mid in self.ids:
            p = self.motion_dir / f"{mid}.npy"
            if not p.is_file():
                missing.append(mid)
                continue
            kept.append(mid)

        if missing:
            print(
                f"[WARN] {len(missing)}/{orig_n} motion .npy files are missing under {self.motion_dir}. "
                f"They will be skipped.",
                flush=True,
            )

        self.ids = kept
        if len(self.ids) == 0:
            raise RuntimeError(f"No existing motion npy found under {self.motion_dir} for split_file: {self.split_file}")

        self.max_motion_len = int(max_motion_len)
        self.min_motion_len = int(min_motion_len)
        self.random_crop = bool(random_crop)
        self.normalize = bool(normalize)

        # How many times to resample another index if a file is missing/corrupted at access time.
        self._max_resample_attempts = 10

        # Load mean/std if provided; else keep None and compute externally if needed
        self.mean = None
        self.std = None
        if self.normalize:
            if mean_path is not None and std_path is not None and mean_path.is_file() and std_path.is_file():
                self.mean = np.load(str(mean_path)).astype(np.float32)
                self.std = np.load(str(std_path)).astype(np.float32)
                self.std = np.maximum(self.std, 1e-6)
            else:
                # user may compute and provide later
                self.mean = None
                self.std = None

        # Infer feature dim from first valid sample
        self.feat_dim = self._infer_feat_dim()

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)


    def _infer_feat_dim(self) -> int:
        """Infer feature dimension D by scanning for the first readable [T,D] motion file."""
        for mid in self.ids:
            path = self.motion_dir / f"{mid}.npy"
            try:
                arr = np.load(str(path))
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"[WARN] Failed to load motion file {path}: {e}. Skipping.", flush=True)
                continue

            if arr.ndim != 2:
                print(f"[WARN] Expected motion npy as [T,D], got {arr.shape} in {path}. Skipping.", flush=True)
                continue
            return int(arr.shape[1])

        raise RuntimeError(f"Could not infer feature dim: no readable [T,D] npy found in {self.motion_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_raw(self, idx: int) -> np.ndarray:
        mid = self.ids[idx]
        path = self.motion_dir / f"{mid}.npy"
        # np.load can raise FileNotFoundError / OSError (corrupted file) etc.
        arr = np.load(str(path)).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected [T,D] motion, got {arr.shape} in {path}")
        return arr

    def __getitem__(self, idx: int):
        # Robust loading: if a referenced .npy is missing/corrupted, resample another id
        # instead of stopping the whole training run.
        x = None
        mid = None
        for _ in range(self._max_resample_attempts):
            mid = self.ids[idx]
            try:
                x = self._load_raw(idx)  # [T,D]
                break
            except FileNotFoundError:
                idx = random.randint(0, len(self.ids) - 1)
                continue
            except Exception as e:
                print(f"[WARN] Failed to load motion file for id={mid}: {e}. Resampling.", flush=True)
                idx = random.randint(0, len(self.ids) - 1)
                continue

        if x is None or mid is None:
            # Fallback (should be extremely rare): return an all-zero sample so the dataloader keeps moving.
            mid = self.ids[0]
            x = np.zeros((self.max_motion_len, self.feat_dim), dtype=np.float32)
        T = int(x.shape[0])

        if T < self.min_motion_len:
            # pad up to min len
            pad = self.min_motion_len - T
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
            T = int(x.shape[0])

        # Crop/pad to max_motion_len for training stability
        if T > self.max_motion_len:
            if self.random_crop:
                start = random.randint(0, T - self.max_motion_len)
            else:
                start = 0
            x = x[start:start + self.max_motion_len]
        elif T < self.max_motion_len:
            pad = self.max_motion_len - T
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")

        # Normalize
        if self.normalize and self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        x_t = torch.from_numpy(x)  # [T,D]
        return mid, x_t


def collate_batch(batch):
    ids = [b[0] for b in batch]
    xs = [b[1] for b in batch]
    # All are same length due to dataset padding/cropping
    x = torch.stack(xs, dim=0)  # [B,T,D]
    # mask: valid where any non-zero? (Since we padded with zeros)
    # For normalization, zeros may be meaningful; so use a separate mask:
    # Here we assume the dataset has fixed length max_motion_len and real length unknown in this loader.
    # For training this baseline, we treat all frames as valid.
    mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool)
    return ids, x, mask


def compute_mean_std(
    data_root: Path,
    split: str,
    motion_dir: str,
    split_file: Optional[Path],
    max_files: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over all frames in the split (optionally subsample files).
    """
    ds = HumanML3DMotionDataset(
        data_root=Path(data_root),
        split=split,
        motion_dir=motion_dir,
        split_file=split_file,
        random_crop=False,
        normalize=False,
    )
    ids = ds.ids
    if max_files > 0 and len(ids) > max_files:
        rng = random.Random(seed)
        ids = rng.sample(ids, max_files)

    sums = None
    sqs = None
    count = 0

    missing = 0
    bad = 0

    for mid in ids:
        p = ds.motion_dir / f"{mid}.npy"
        try:
            arr = np.load(str(p)).astype(np.float32)  # [T,D]
        except FileNotFoundError:
            missing += 1
            continue
        except Exception:
            bad += 1
            continue

        if arr.ndim != 2:
            bad += 1
            continue

        if sums is None:
            sums = arr.sum(axis=0)
            sqs = (arr ** 2).sum(axis=0)
        else:
            sums += arr.sum(axis=0)
            sqs += (arr ** 2).sum(axis=0)
        count += arr.shape[0]

    if missing > 0:
        print(f"[WARN] compute_mean_std: skipped {missing} missing motion files.", flush=True)
    if bad > 0:
        print(f"[WARN] compute_mean_std: skipped {bad} unreadable/corrupted motion files.", flush=True)

    if sums is None or sqs is None or count == 0:
        raise RuntimeError("Failed to compute mean/std (no data?)")

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
        """
        x: [B, N, D]
        """
        n = x.size(1)
        return x + self.pe[:n].unsqueeze(0).to(x.dtype)


class VectorQuantizer(nn.Module):
    """
    VQ-VAE quantizer (nearest neighbor in a learnable codebook) with straight-through estimator.

    Options:
      - use_l2_norm: use L2-normalized vectors for codebook lookup (Euclidean distance on unit sphere).
      - orth_reg_weight: weight for codebook orthogonality regularization (approximate).
      - orth_reg_samples: number of code vectors sampled to compute orth loss each step
                          (<=0 or >=K means full K, which can be expensive).
    """
    def __init__(
        self,
        codebook_size: int,
        d_model: int,
        beta: float = 0.25,
        use_l2_norm: bool = True,
        orth_reg_weight: float = 0.0,
        orth_reg_samples: int = 1024,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.d_model = int(d_model)
        self.beta = float(beta)
        self.use_l2_norm = bool(use_l2_norm)
        self.orth_reg_weight = float(orth_reg_weight)
        self.orth_reg_samples = int(orth_reg_samples)

        self.codebook = nn.Embedding(self.codebook_size, self.d_model)
        self.codebook.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def _orth_loss(self) -> torch.Tensor:
        """Approximate orthogonality regularization on the codebook."""
        if self.orth_reg_weight <= 0.0:
            return torch.zeros((), device=self.codebook.weight.device, dtype=self.codebook.weight.dtype)

        e = self.codebook.weight  # [K,D]
        e = F.normalize(e, dim=1)
        K = e.size(0)
        s = self.orth_reg_samples

        if s <= 0 or s >= K:
            e_s = e
        else:
            # Sample a subset of code vectors to keep the Gram matrix manageable.
            idx = torch.randperm(K, device=e.device)[:s]
            e_s = e.index_select(0, idx)

        gram = e_s @ e_s.T  # [S,S]
        I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return ((gram - I) ** 2).mean()

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        z_e: [B, N, D]
        returns:
          z_q_st: [B, N, D] quantized (straight-through)
          indices: [B, N] code indices
          loss_vq_total: scalar
          stats: dict (perplexity, usage, losses, etc.)
        """
        B, N, D = z_e.shape
        z = z_e.reshape(-1, D)  # [BN,D]
        e = self.codebook.weight  # [K,D]

        # Compute distance to codebook entries
        if self.use_l2_norm:
            z_n = F.normalize(z, dim=1)
            e_n = F.normalize(e, dim=1)
            # ||z_n - e_n||^2 = 2 - 2 * <z_n, e_n>
            dist = 2.0 - 2.0 * (z_n @ e_n.T)  # [BN,K]
        else:
            # Squared L2: ||z||^2 + ||e||^2 - 2 z e^T
            z2 = (z ** 2).sum(dim=1, keepdim=True)  # [BN,1]
            e2 = (e ** 2).sum(dim=1, keepdim=True).T  # [1,K]
            dist = z2 + e2 - 2.0 * (z @ e.T)  # [BN,K]

        indices = torch.argmin(dist, dim=1)  # [BN]
        z_q = self.codebook(indices).view(B, N, D)

        # VQ losses (standard)
        loss_codebook = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        loss_vq = loss_codebook + self.beta * loss_commit

        # Optional orth regularization
        loss_orth = self._orth_loss()
        loss_total = loss_vq + self.orth_reg_weight * loss_orth

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Stats
        with torch.no_grad():
            idx = indices.detach()
            hist = torch.bincount(idx, minlength=self.codebook_size).float()
            prob = hist / (hist.sum() + 1e-12)
            entropy = -(prob * (prob + 1e-12).log()).sum()
            perplexity = torch.exp(entropy)
            usage = (hist > 0).float().mean()  # fraction of active codes
            stats = {
                "vq_perplexity": float(perplexity.cpu().item()),
                "vq_entropy": float(entropy.cpu().item()),
                "vq_usage_frac": float(usage.cpu().item()),
                "vq_loss_codebook": float(loss_codebook.detach().cpu().item()),
                "vq_loss_commit": float(loss_commit.detach().cpu().item()),
                "vq_loss_orth": float(loss_orth.detach().cpu().item()),
                "vq_use_l2_norm": float(1.0 if self.use_l2_norm else 0.0),
            }

        return z_q_st, indices.view(B, N), loss_total, stats

class TransformerVQVAE(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        codebook_size: int,
        beta: float,
        use_l2_norm: bool = True,
        orth_reg_weight: float = 0.0,
        orth_reg_samples: int = 1024,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)

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

        self.quantizer = VectorQuantizer(
            codebook_size=int(codebook_size),
            d_model=self.d_model,
            beta=float(beta),
            use_l2_norm=bool(use_l2_norm),
            orth_reg_weight=float(orth_reg_weight),
            orth_reg_samples=int(orth_reg_samples),
        )

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

    def patchify(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        x: [B,T,D]  mask: [B,T] bool (True valid)
        returns:
          x_patch: [B,N,P*D]
          patch_mask: [B,N] bool (True valid)
          x_pad: [B,T_pad,D]
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
        return x_patch, patch_mask, x, T_pad

    def unpatchify(self, y_patch: torch.Tensor, T_pad: int) -> torch.Tensor:
        """
        y_patch: [B,N,P*D] -> [B,T_pad,D]
        """
        B, N, PD = y_patch.shape
        P = self.patch_len
        D = self.feat_dim
        assert PD == P * D
        y = y_patch.view(B, N, P, D).reshape(B, N * P, D)
        if y.size(1) != T_pad:
            y = y[:, :T_pad, :]
        return y

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        x: [B,T,D] (normalized)
        mask: [B,T] bool
        returns:
          x_hat: [B,T_pad,D]
          indices: [B,N]
          losses: dict of tensors
          stats: dict of floats
        """
        x_patch, patch_mask, x_pad, T_pad = self.patchify(x, mask)  # [B,N,PD], [B,N]
        h = self.in_proj(x_patch)  # [B,N,D]
        h = self.pos_enc(h)

        key_padding = ~patch_mask  # True for pad tokens
        z_e = self.encoder(h, src_key_padding_mask=key_padding)

        z_q, indices, loss_vq, vq_stats = self.quantizer(z_e)

        z_q = self.pos_enc(z_q)
        y = self.decoder(z_q, src_key_padding_mask=key_padding)
        y_patch = self.out_proj(y)  # [B,N,PD]
        x_hat = self.unpatchify(y_patch, T_pad)  # [B,T_pad,D]

        # recon loss only on valid frames
        recon_mse = (x_hat - x_pad).pow(2).mean(dim=-1)  # [B,T_pad]
        recon_loss = recon_mse[mask].mean()

        losses = {
            "recon": recon_loss,
            "vq": loss_vq,
            "total": recon_loss + loss_vq,
        }
        stats = dict(vq_stats)
        stats["recon_mse"] = float(recon_loss.detach().cpu().item())
        stats["num_tokens_per_seq"] = float(indices.size(1))
        return x_hat, indices, losses, stats


# -------------------------
# Training / Eval
# -------------------------

@dataclass
class TrainConfig:
    data_root: str
    save_dir: str
    motion_dir: str = "new_joint_vecs"
    patch_len: int = 1
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    codebook_size: int = 8192
    beta: float = 0.25
    dropout: float = 0.1
    ff_mult: int = 4

    # quantization (M2DM-style)
    use_l2_norm: bool = True
    orth_reg_weight: float = 1e-3
    orth_reg_samples: int = 1024

    # data
    max_motion_len: int = 64
    min_motion_len: int = 40
    normalize: bool = True
    mean_path: str = ""
    std_path: str = ""

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


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, step: int, epoch: int, cfg: TrainConfig, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "cfg": asdict(cfg),
        "mean": None if mean is None else mean.astype(np.float32),
        "std": None if std is None else std.astype(np.float32),
    }
    torch.save(ckpt, str(path))


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Dict:
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


@torch.no_grad()
def evaluate_recon(model: TransformerVQVAE, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    agg = {
        "recon_mse": 0.0,
        "vq_loss_codebook": 0.0,
        "vq_loss_commit": 0.0,
        "vq_entropy": 0.0,
        "vq_perplexity": 0.0,
        "vq_usage_frac": 0.0,
        "count": 0,
    }
    for ids, x, mask in loader:
        x = x.to(device)
        mask = mask.to(device)
        _, _, losses, stats = model(x, mask)
        bs = x.size(0)
        agg["recon_mse"] += stats["recon_mse"] * bs
        agg["vq_loss_codebook"] += stats["vq_loss_codebook"] * bs
        agg["vq_loss_commit"] += stats["vq_loss_commit"] * bs
        agg["vq_entropy"] += stats["vq_entropy"] * bs
        agg["vq_perplexity"] += stats["vq_perplexity"] * bs
        agg["vq_usage_frac"] += stats["vq_usage_frac"] * bs
        agg["count"] += bs

    c = max(1, int(agg["count"]))
    out = {k: (v / c) for k, v in agg.items() if k != "count"}
    return out


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)

    save_dir = Path(cfg.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    log_path = save_dir / "logs" / "train_log.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = cfg.device.strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

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

    # Compute mean/std if requested and not provided
    if cfg.normalize and (ds_train.mean is None or ds_train.std is None):
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
        # Save for reproducibility
        (save_dir / "norm").mkdir(parents=True, exist_ok=True)
        np.save(str(save_dir / "norm" / "mean.npy"), mean)
        np.save(str(save_dir / "norm" / "std.npy"), std)
        print(f"[INFO] Saved mean/std to: {save_dir / 'norm'}", flush=True)
    else:
        mean = ds_train.mean
        std = ds_train.std

    feat_dim = ds_train.feat_dim
    model = TransformerVQVAE(
        feat_dim=feat_dim,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        codebook_size=cfg.codebook_size,
        beta=cfg.beta,
        use_l2_norm=cfg.use_l2_norm,
        orth_reg_weight=cfg.orth_reg_weight,
        orth_reg_samples=cfg.orth_reg_samples,
        dropout=cfg.dropout,
        ff_mult=cfg.ff_mult,
    ).to(device_t)

    # Model parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model params: {n_params/1e6:.2f} M", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch)

    best_val = float("inf")
    step = 0
    start_time = time.time()

    # Save config
    (save_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        for it, (ids, x, mask) in enumerate(dl_train, start=1):
            x = x.to(device_t)
            mask = mask.to(device_t)

            _, _, losses, stats = model(x, mask)
            loss = losses["total"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            step += 1

            if step % cfg.log_every == 0:
                row = {
                    "time": time.time(),
                    "epoch": epoch,
                    "iter": it,
                    "step": step,
                    "lr": float(optim.param_groups[0]["lr"]),
                    "loss_total": float(losses["total"].detach().cpu().item()),
                    "loss_recon": float(losses["recon"].detach().cpu().item()),
                    "loss_vq": float(losses["vq"].detach().cpu().item()),
                    **stats,
                }
                save_jsonl(log_path, row)
                print(
                    f"[train] ep {epoch:03d} it {it:04d} step {step:07d} "
                    f"loss={row['loss_total']:.4f} recon={row['loss_recon']:.4f} vq={row['loss_vq']:.4f} "
                    f"ppl={row['vq_perplexity']:.1f} usage={row['vq_usage_frac']:.3f}",
                    flush=True,
                )

        # Eval
        if epoch % cfg.eval_every == 0:
            val = evaluate_recon(model, dl_val, device_t)
            row = {"time": time.time(), "epoch": epoch, "step": step, "split": "val", **val}
            save_jsonl(log_path, row)
            print(
                f"[val] ep {epoch:03d} recon_mse={val['recon_mse']:.6f} "
                f"ppl={val['vq_perplexity']:.1f} usage={val['vq_usage_frac']:.3f} "
                f"elapsed={human_time(time.time()-start_time)}",
                flush=True,
            )

            if val["recon_mse"] < best_val:
                best_val = val["recon_mse"]
                save_checkpoint(ckpt_dir / "ckpt_best.pt", model, optim, step, epoch, cfg, mean, std)

        # Save periodic + last
        if epoch % cfg.save_every == 0:
            save_checkpoint(ckpt_dir / "ckpt_last.pt", model, optim, step, epoch, cfg, mean, std)

        print(f"[epoch] {epoch:03d} finished in {human_time(time.time()-t0)}", flush=True)

    print(f"[done] training finished. best_val_recon_mse={best_val:.6f}", flush=True)


# -------------------------
# Encode / Reconstruct
# -------------------------

@torch.no_grad()
def run_encode(
    data_root: Path,
    split: str,
    motion_dir: str,
    ckpt_path: Path,
    out_tokens_dir: Path,
    max_motion_len: int,
    min_motion_len: int,
    normalize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    patch_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    codebook_size: int,
    beta: float,
    use_l2_norm: bool,
    orth_reg_weight: float,
    orth_reg_samples: int,
    dropout: float,
    ff_mult: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    device_t = torch.device(device)
    ds = HumanML3DMotionDataset(
        data_root=data_root,
        split=split,
        motion_dir=motion_dir,
        max_motion_len=max_motion_len,
        min_motion_len=min_motion_len,
        random_crop=False,
        normalize=normalize,
    )
    if normalize and mean is not None and std is not None:
        ds.set_norm(mean, std)

    model = TransformerVQVAE(
        feat_dim=ds.feat_dim,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        codebook_size=codebook_size,
        beta=beta,
        use_l2_norm=use_l2_norm,
        orth_reg_weight=orth_reg_weight,
        orth_reg_samples=orth_reg_samples,
        dropout=dropout,
        ff_mult=ff_mult,
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model params: {n_params/1e6:.2f} M", flush=True)
    _ = load_checkpoint(ckpt_path, model, device_t)
    model.eval()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)

    out_tokens_dir = out_tokens_dir / split
    out_tokens_dir.mkdir(parents=True, exist_ok=True)

    for ids, x, mask in dl:
        x = x.to(device_t)
        mask = mask.to(device_t)
        _, indices, _, _ = model(x, mask)  # indices: [B,N]

        indices_np = indices.detach().cpu().numpy()
        for b, mid in enumerate(ids):
            seq = indices_np[b].tolist()
            (out_tokens_dir / f"{mid}.txt").write_text(" ".join(map(str, seq)) + "\n", encoding="utf-8")

    print(f"[done] tokens saved to: {out_tokens_dir}", flush=True)


@torch.no_grad()
def run_reconstruct(
    data_root: Path,
    split: str,
    motion_dir: str,
    ckpt_path: Path,
    out_recon_dir: Path,
    max_motion_len: int,
    min_motion_len: int,
    normalize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    patch_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    codebook_size: int,
    beta: float,
    use_l2_norm: bool,
    orth_reg_weight: float,
    orth_reg_samples: int,
    dropout: float,
    ff_mult: int,
    batch_size: int,
    num_workers: int,
    device: str,
    save_input: bool,
) -> None:
    device_t = torch.device(device)
    ds = HumanML3DMotionDataset(
        data_root=data_root,
        split=split,
        motion_dir=motion_dir,
        max_motion_len=max_motion_len,
        min_motion_len=min_motion_len,
        random_crop=False,
        normalize=normalize,
    )
    if normalize and mean is not None and std is not None:
        ds.set_norm(mean, std)

    model = TransformerVQVAE(
        feat_dim=ds.feat_dim,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        codebook_size=codebook_size,
        beta=beta,
        use_l2_norm=use_l2_norm,
        orth_reg_weight=orth_reg_weight,
        orth_reg_samples=orth_reg_samples,
        dropout=dropout,
        ff_mult=ff_mult,
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model params: {n_params/1e6:.2f} M", flush=True)
    _ = load_checkpoint(ckpt_path, model, device_t)
    model.eval()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)

    out_recon_dir = out_recon_dir / split
    out_recon_dir.mkdir(parents=True, exist_ok=True)
    if save_input:
        (out_recon_dir / "_input").mkdir(parents=True, exist_ok=True)

    for ids, x, mask in dl:
        x = x.to(device_t)
        mask = mask.to(device_t)
        x_hat, _, _, _ = model(x, mask)  # [B,T,D]

        x_hat_np = x_hat.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()

        # de-normalize if requested
        if normalize and mean is not None and std is not None:
            x_hat_np = x_hat_np * std[None, None, :] + mean[None, None, :]
            x_np = x_np * std[None, None, :] + mean[None, None, :]

        for b, mid in enumerate(ids):
            np.save(str(out_recon_dir / f"{mid}.npy"), x_hat_np[b].astype(np.float32))
            if save_input:
                np.save(str(out_recon_dir / "_input" / f"{mid}.npy"), x_np[b].astype(np.float32))

    print(f"[done] recon saved to: {out_recon_dir}", flush=True)


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    # train
    ap_train = sub.add_parser("train")
    ap_train.add_argument("--data_root", type=str, required=True)
    ap_train.add_argument("--save_dir", type=str, required=True)
    ap_train.add_argument("--motion_dir", type=str, default="new_joint_vecs")
    ap_train.add_argument("--patch_len", type=int, default=1)
    ap_train.add_argument("--d_model", type=int, default=512)
    ap_train.add_argument("--n_heads", type=int, default=8)
    ap_train.add_argument("--n_layers", type=int, default=4)
    ap_train.add_argument("--codebook_size", type=int, default=8192)
    ap_train.add_argument("--beta", type=float, default=0.25)
    ap_train.add_argument("--dropout", type=float, default=0.1)
    ap_train.add_argument("--ff_mult", type=int, default=4)

    # quantization (M2DM-style)
    ap_train.add_argument("--no_vq_l2norm", action="store_true", help="Disable L2-normalized distance for VQ lookup.")
    ap_train.add_argument("--orth_reg_weight", type=float, default=1e-3, help="Weight for codebook orthogonality regularization (approx).")
    ap_train.add_argument("--orth_reg_samples", type=int, default=1024, help="Num code vectors sampled for orth reg (<=0: full; expensive).")

    ap_train.add_argument("--max_motion_len", type=int, default=64)
    ap_train.add_argument("--min_motion_len", type=int, default=40)
    ap_train.add_argument("--normalize", action="store_true")
    ap_train.add_argument("--mean_path", type=str, default="")
    ap_train.add_argument("--std_path", type=str, default="")

    ap_train.add_argument("--epochs", type=int, default=100)
    ap_train.add_argument("--batch_size", type=int, default=64)
    ap_train.add_argument("--lr", type=float, default=1e-4)
    ap_train.add_argument("--weight_decay", type=float, default=1e-4)
    ap_train.add_argument("--grad_clip", type=float, default=1.0)

    ap_train.add_argument("--num_workers", type=int, default=4)
    ap_train.add_argument("--device", type=str, default="")
    ap_train.add_argument("--seed", type=int, default=1234)
    ap_train.add_argument("--log_every", type=int, default=50)
    ap_train.add_argument("--eval_every", type=int, default=1)
    ap_train.add_argument("--save_every", type=int, default=1)

    # encode
    ap_enc = sub.add_parser("encode")
    ap_enc.add_argument("--data_root", type=str, required=True)
    ap_enc.add_argument("--save_dir", type=str, required=True, help="Training run dir (for config/norm).")
    ap_enc.add_argument("--ckpt", type=str, required=True)
    ap_enc.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap_enc.add_argument("--motion_dir", type=str, default="new_joint_vecs")
    ap_enc.add_argument("--out_tokens_dir", type=str, default="", help="Default: <save_dir>/tokens")
    ap_enc.add_argument("--batch_size", type=int, default=64)
    ap_enc.add_argument("--num_workers", type=int, default=4)
    ap_enc.add_argument("--device", type=str, default="")
    ap_enc.add_argument("--normalize", action="store_true")

    # reconstruct
    ap_rec = sub.add_parser("reconstruct")
    ap_rec.add_argument("--data_root", type=str, required=True)
    ap_rec.add_argument("--save_dir", type=str, required=True, help="Training run dir (for config/norm).")
    ap_rec.add_argument("--ckpt", type=str, required=True)
    ap_rec.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap_rec.add_argument("--motion_dir", type=str, default="new_joint_vecs")
    ap_rec.add_argument("--out_recon_dir", type=str, default="", help="Default: <save_dir>/recon")
    ap_rec.add_argument("--batch_size", type=int, default=64)
    ap_rec.add_argument("--num_workers", type=int, default=4)
    ap_rec.add_argument("--device", type=str, default="")
    ap_rec.add_argument("--normalize", action="store_true")
    ap_rec.add_argument("--save_input", action="store_true", help="Also save the (de-normalized) input features for reference.")

    args = ap.parse_args()

    if args.command == "train":
        cfg = TrainConfig(
            data_root=args.data_root,
            save_dir=args.save_dir,
            motion_dir=args.motion_dir,
            patch_len=args.patch_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            codebook_size=args.codebook_size,
            beta=args.beta,
            dropout=args.dropout,
            ff_mult=args.ff_mult,
            use_l2_norm=not bool(args.no_vq_l2norm),
            orth_reg_weight=float(args.orth_reg_weight),
            orth_reg_samples=int(args.orth_reg_samples),
            max_motion_len=args.max_motion_len,
            min_motion_len=args.min_motion_len,
            normalize=bool(args.normalize),
            mean_path=args.mean_path,
            std_path=args.std_path,
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
        )
        train(cfg)
        return

    # Shared load config from save_dir/config.json if exists (so encode/recon matches training hyperparams)
    save_dir = Path(args.save_dir)
    cfg_path = save_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"ERROR: config.json not found in save_dir: {save_dir} (run train first)")

    # Load mean/std from save_dir/norm if exists and --normalize
    mean = None
    std = None
    if args.normalize:
        mean_p = save_dir / "norm" / "mean.npy"
        std_p = save_dir / "norm" / "std.npy"
        if mean_p.is_file() and std_p.is_file():
            mean = np.load(str(mean_p)).astype(np.float32)
            std = np.load(str(std_p)).astype(np.float32)
        else:
            print("[WARN] --normalize is set but mean/std not found. Proceeding without de/normalization.", flush=True)

    device = args.device.strip()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.command == "encode":
        out_tokens_dir = Path(args.out_tokens_dir) if args.out_tokens_dir else (save_dir / "tokens")
        run_encode(
            data_root=Path(args.data_root),
            split=args.split,
            motion_dir=args.motion_dir,
            ckpt_path=Path(args.ckpt),
            out_tokens_dir=out_tokens_dir,
            max_motion_len=int(cfg["max_motion_len"]),
            min_motion_len=int(cfg["min_motion_len"]),
            normalize=bool(args.normalize),
            mean=mean,
            std=std,
            patch_len=int(cfg["patch_len"]),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            codebook_size=int(cfg["codebook_size"]),
            beta=float(cfg["beta"]),
            use_l2_norm=bool(cfg.get("use_l2_norm", True)),
            orth_reg_weight=float(cfg.get("orth_reg_weight", 0.0)),
            orth_reg_samples=int(cfg.get("orth_reg_samples", 1024)),
            dropout=float(cfg["dropout"]),
            ff_mult=int(cfg["ff_mult"]),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        return

    if args.command == "reconstruct":
        out_recon_dir = Path(args.out_recon_dir) if args.out_recon_dir else (save_dir / "recon")
        run_reconstruct(
            data_root=Path(args.data_root),
            split=args.split,
            motion_dir=args.motion_dir,
            ckpt_path=Path(args.ckpt),
            out_recon_dir=out_recon_dir,
            max_motion_len=int(cfg["max_motion_len"]),
            min_motion_len=int(cfg["min_motion_len"]),
            normalize=bool(args.normalize),
            mean=mean,
            std=std,
            patch_len=int(cfg["patch_len"]),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            codebook_size=int(cfg["codebook_size"]),
            beta=float(cfg["beta"]),
            use_l2_norm=bool(cfg.get("use_l2_norm", True)),
            orth_reg_weight=float(cfg.get("orth_reg_weight", 0.0)),
            orth_reg_samples=int(cfg.get("orth_reg_samples", 1024)),
            dropout=float(cfg["dropout"]),
            ff_mult=int(cfg["ff_mult"]),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            save_input=bool(args.save_input),
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
