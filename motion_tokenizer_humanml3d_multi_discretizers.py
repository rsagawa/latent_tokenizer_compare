#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motion_tokenizer_humanml3d_multi_discretizers.py

HumanML3D Motion Tokenizer Baselines (multiple discretization bottlenecks)
===========================================================================

This script trains a motion-feature autoencoder on HumanML3D (new_joint_vecs, 263-D)
with several *discrete* bottlenecks, following the discretization techniques
surveyed in:
  - Kaiser et al., 2018, "Fast Decoding in Sequence Models Using Discrete Latent Variables"
    (Gumbel-Softmax, VQ-VAE, Improved Semantic Hashing, Decomposed Vector Quantization).

Supported bottlenecks
---------------------
  1) gumbel   : straight-through Gumbel-Softmax over vocab_size categories.
  2) vqvae    : nearest-neighbor vector quantization (single codebook).
  3) ish      : improved semantic hashing (bit bottleneck + straight-through).
  4) dvq      : decomposed vector quantization ("sliced" or "projected" variants).

In all cases, the backbone is:
  motion feats [T, D] -> patchify -> Transformer encoder -> discrete bottleneck ->
  Transformer decoder -> reconstruct patches -> unpatchify -> [T, D]

Outputs (compatible with MotionGPT evaluation script)
-----------------------------------------------------
<save_dir>/
  checkpoints/ckpt_last.pt, ckpt_best.pt
  logs/train_log.jsonl
  norm/mean.npy, norm/std.npy                 (if --normalize)
  tokens/<split>/<id>.txt                     (encode)
  recon/<split>/<id>.npy                      (reconstruct; de-normalized)

After running `reconstruct`, compute MotionGPT metrics using:
  python eval_recon_humanml3d_motiongpt_metrics.py \
    --data_root  /path/to/HumanML3D \
    --recon_root <save_dir> \
    --split test \
    --t2m_path   /path/to/t2m_evaluators \
    --meta_dir   /path/to/MotionGPT/assets/meta

Proposal-style constraints (optional)
------------------------------------
Optionally add a regularizer inspired by the user's proposal losses:
  - per-token entropy cap ("possharp")
  - marginal distribution regularization (KL-to-uniform)
  - InfoNCE diversity on token posteriors

These are enabled via --proposal_* flags.

Notes
-----
* This is a research baseline / scaffold. It is not an exact reproduction of
  the original Tensor2Tensor implementations.
* For large vocab sizes, VQ-VAE constraints based on full softmax over codebook
  distances can be memory heavy. DVQ is usually more efficient.
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
    """Load HumanML3D motions from new_joint_vecs/<id>.npy.

    Returns:
      mid: str
      x  : torch.FloatTensor [T, D] (padded/cropped to max_motion_len)
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
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = str(split)
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

        # Filter missing motion files so training won't crash on FileNotFoundError.
        missing = 0
        self.ids: List[str] = []
        for mid in ids:
            if (self.motion_dir / f"{mid}.npy").is_file():
                self.ids.append(mid)
            else:
                missing += 1
        if missing > 0:
            print(f"[warn] {self.split}: skipped {missing} ids with missing npy under {self.motion_dir}")

        if len(self.ids) == 0:
            raise RuntimeError(f"No valid motion files found for split={self.split} under {self.motion_dir}")

        self.max_motion_len = int(max_motion_len)
        self.min_motion_len = int(min_motion_len)
        self.random_crop = bool(random_crop)
        self.normalize = bool(normalize)

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        if self.normalize and mean_path is not None and std_path is not None:
            if mean_path.is_file() and std_path.is_file():
                self.mean = np.load(str(mean_path)).astype(np.float32)
                self.std = np.maximum(np.load(str(std_path)).astype(np.float32), 1e-6)

        # Infer feature dim from the first readable motion file (do not assume ids[0] exists).
        feat_dim = None
        bad = 0
        for mid in self.ids:
            try:
                arr0 = np.load(str(self.motion_dir / f"{mid}.npy"))
            except Exception:
                bad += 1
                continue
            if getattr(arr0, "ndim", None) != 2:
                bad += 1
                continue
            feat_dim = int(arr0.shape[1])
            break
        if feat_dim is None:
            raise RuntimeError(f"Failed to infer feat_dim (all files unreadable/invalid?) under {self.motion_dir}")
        if bad > 0:
            print(f"[warn] {self.split}: encountered {bad} unreadable/invalid npy while inferring feat_dim (will be skipped during sampling).")
        self.feat_dim = int(feat_dim)

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def __len__(self) -> int:
        return len(self.ids)
    def __getitem__(self, idx: int):
        """Safe loader: if a target npy is missing/unreadable, resample another id."""
        mid0 = self.ids[idx]
        max_tries = 20
        last_mid = mid0

        for attempt in range(max_tries):
            mid = mid0 if attempt == 0 else random.choice(self.ids)
            last_mid = mid
            fpath = self.motion_dir / f"{mid}.npy"
            try:
                arr = np.load(str(fpath)).astype(np.float32)
            except FileNotFoundError:
                continue
            except Exception:
                continue
            if arr.ndim != 2 or int(arr.shape[1]) != int(self.feat_dim):
                continue

            T0 = int(arr.shape[0])
            if T0 < self.min_motion_len:
                pad = self.min_motion_len - T0
                arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
                T0 = int(arr.shape[0])

            # Crop/pad to max_motion_len
            if T0 > self.max_motion_len:
                start = random.randint(0, T0 - self.max_motion_len) if self.random_crop else 0
                arr = arr[start : start + self.max_motion_len]
                valid_len = self.max_motion_len
            elif T0 < self.max_motion_len:
                valid_len = T0
                pad = self.max_motion_len - T0
                arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant")
            else:
                valid_len = self.max_motion_len

            if self.normalize and self.mean is not None and self.std is not None:
                arr = (arr - self.mean) / self.std

            x = torch.from_numpy(arr)
            mask = torch.zeros(self.max_motion_len, dtype=torch.bool)
            mask[:valid_len] = True
            return mid, x, mask

        # Fallback: return zeros (keeps training running even if data is badly broken)
        arr = np.zeros((self.max_motion_len, self.feat_dim), dtype=np.float32)
        if self.normalize and self.mean is not None and self.std is not None:
            arr = (arr - self.mean) / self.std
        x = torch.from_numpy(arr)
        mask = torch.zeros(self.max_motion_len, dtype=torch.bool)
        return last_mid, x, mask


def collate_motion(batch):
    ids = [b[0] for b in batch]
    x = torch.stack([b[1] for b in batch], dim=0)
    mask = torch.stack([b[2] for b in batch], dim=0)
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
        max_motion_len=196,
        min_motion_len=1,
        random_crop=False,
        normalize=False,
    )
    ids = list(ds.ids)
    rng = random.Random(int(seed))
    if max_files > 0 and len(ids) > max_files:
        ids = rng.sample(ids, max_files)

    sums: Optional[np.ndarray] = None
    sqs: Optional[np.ndarray] = None
    count = 0
    skipped = 0
    for mid in ids:
        try:
            arr = np.load(str(ds.motion_dir / f"{mid}.npy")).astype(np.float32)
        except FileNotFoundError:
            skipped += 1
            continue
        except Exception:
            skipped += 1
            continue
        if arr.ndim != 2:
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
        print(f"[warn] compute_mean_std: skipped {skipped} missing/unreadable/invalid npy files")

    if sums is None or sqs is None or count == 0:
        raise RuntimeError("Failed to compute mean/std (no readable data?)")
    mean = sums / count
    var = sqs / count - mean ** 2
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# -------------------------
# Proposal-style regularizers
# -------------------------


def info_nce_self(z: torch.Tensor, *, temperature: float = 0.2) -> torch.Tensor:
    """InfoNCE that uses each vector as its own positive and all others as negatives.

    z: [M, D], assumed normalized (||z||=1)
    Returns scalar loss.
    """
    if z.ndim != 2:
        raise ValueError(f"info_nce_self expects [M,D], got {z.shape}")
    m = int(z.shape[0])
    if m <= 1:
        return z.new_tensor(0.0)
    logits = (z @ z.t()) / max(float(temperature), 1e-6)
    labels = torch.arange(m, device=z.device)
    return F.cross_entropy(logits, labels)


class ProposalConstraintsSoftmax(nn.Module):
    """Per-token entropy cap + marginal KL-to-uniform + InfoNCE diversity for softmax logits."""

    def __init__(
        self,
        vocab_size: int,
        *,
        H_cap: float = 3.0,
        tau_ent: float = 0.7,
        tau_marg: float = 1.3,
        ema_decay: float = 0.99,
        alpha_mix_ema: float = 0.5,
        infonce_temperature: float = 0.2,
        max_infonce_samples: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.H_cap = float(H_cap)
        self.tau_ent = float(tau_ent)
        self.tau_marg = float(tau_marg)
        self.ema_decay = float(ema_decay)
        self.alpha_mix_ema = float(alpha_mix_ema)
        self.infonce_temperature = float(infonce_temperature)
        self.max_infonce_samples = int(max_infonce_samples)
        ema = torch.ones(self.vocab_size, dtype=torch.float32) / float(self.vocab_size)
        self.register_buffer("ema_prior", ema, persistent=True)

    @torch.no_grad()
    def _update_ema(self, p_marg: torch.Tensor) -> None:
        p = p_marg.detach().to(self.ema_prior.dtype)
        p = p / p.sum().clamp_min(1e-12)
        self.ema_prior.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * p)
        self.ema_prior.div_(self.ema_prior.sum().clamp_min(1e-12))

    def forward(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute (L_possharp, L_marg, L_info_nce, stats)."""
        assert logits.dim() == 3, "logits must be [B,N,V]"
        B, N, V = logits.shape
        if V != self.vocab_size:
            raise ValueError(f"vocab mismatch: logits V={V} vs reg V={self.vocab_size}")
        eps = 1e-12

        # valid positions
        if mask is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=logits.device)
        else:
            valid = mask.to(torch.bool)

        p = F.softmax(logits / max(self.tau_ent, eps), dim=-1)
        q = F.softmax(logits / max(self.tau_marg, eps), dim=-1)

        # 1) per-position entropy cap
        H = -(p * (p.clamp_min(eps).log())).sum(dim=-1)  # [B,N]
        tau_soft = 0.3
        L_possharp = (torch.log1p(torch.exp((H - self.H_cap) / tau_soft)) * tau_soft)
        L_possharp = L_possharp[valid].mean() if valid.any() else L_possharp.mean()

        # 2) marginal KL-to-uniform (with EMA mixing)
        q_flat = p[valid] if valid.any() else p.reshape(-1, V)
        p_marg = q_flat.mean(dim=0).clamp_min(eps)
        p_marg = p_marg / p_marg.sum()
        p_marg_add = (1.0 - self.alpha_mix_ema) * p_marg + self.alpha_mix_ema * self.ema_prior.detach()
        p_marg_add = p_marg_add / p_marg_add.sum().clamp_min(eps)
        U = torch.ones_like(p_marg_add) / float(V)
        L_marg = F.kl_div(p_marg_add.log(), U, reduction="sum")

        with torch.no_grad():
            self._update_ema(p_marg)

        # 3) InfoNCE diversity on q (mask-aware + subsample)
        z = q - q.mean(dim=-1, keepdim=True)
        z = F.normalize(z, dim=-1)
        z_flat = z[valid] if valid.any() else z.reshape(-1, V)
        M = int(z_flat.shape[0])
        if self.max_infonce_samples > 0 and M > self.max_infonce_samples:
            idx = torch.randperm(M, device=z_flat.device)[: self.max_infonce_samples]
            z_flat = z_flat[idx]
        L_info = info_nce_self(z_flat, temperature=self.infonce_temperature)

        stats = {
            "H_mean": float((H[valid].mean() if valid.any() else H.mean()).detach().cpu().item()),
            "p_marg_max": float(p_marg.max().detach().cpu().item()),
            "p_marg_min": float(p_marg.min().detach().cpu().item()),
            "valid_tokens": int(valid.sum().detach().cpu().item()),
        }
        return L_possharp, L_marg, L_info, stats


class ProposalConstraintsBernoulli(nn.Module):
    """Bitwise version of proposal constraints for improved semantic hashing.

    p_bit is probability of bit=1: [B,N,Bits].
    """

    def __init__(
        self,
        num_bits: int,
        *,
        H_cap_bit: float = 0.3,
        ema_decay: float = 0.99,
        alpha_mix_ema: float = 0.5,
        infonce_temperature: float = 0.2,
        max_infonce_samples: int = 1024,
    ) -> None:
        super().__init__()
        self.num_bits = int(num_bits)
        self.H_cap_bit = float(H_cap_bit)
        self.ema_decay = float(ema_decay)
        self.alpha_mix_ema = float(alpha_mix_ema)
        self.infonce_temperature = float(infonce_temperature)
        self.max_infonce_samples = int(max_infonce_samples)
        ema = torch.ones(self.num_bits, dtype=torch.float32) * 0.5
        self.register_buffer("ema_prior", ema, persistent=True)

    @torch.no_grad()
    def _update_ema(self, p_marg: torch.Tensor) -> None:
        p = p_marg.detach().to(self.ema_prior.dtype)
        self.ema_prior.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * p)

    def forward(self, p_bit: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        assert p_bit.dim() == 3, "p_bit must be [B,N,Bits]"
        B, N, Bits = p_bit.shape
        if Bits != self.num_bits:
            raise ValueError(f"bit mismatch: Bits={Bits} vs reg Bits={self.num_bits}")
        eps = 1e-12

        if mask is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=p_bit.device)
        else:
            valid = mask.to(torch.bool)

        p = p_bit.clamp(eps, 1.0 - eps)
        H = -(p * p.log() + (1.0 - p) * (1.0 - p).log())  # [B,N,Bits]

        tau_soft = 0.3
        L_pos = (torch.log1p(torch.exp((H - self.H_cap_bit) / tau_soft)) * tau_soft)
        if valid.any():
            L_pos = L_pos[valid].mean()
        else:
            L_pos = L_pos.mean()

        # marginal to Bernoulli(0.5)
        p_marg = (p[valid].mean(dim=0) if valid.any() else p.reshape(-1, Bits).mean(dim=0))
        p_marg_add = (1.0 - self.alpha_mix_ema) * p_marg + self.alpha_mix_ema * self.ema_prior.detach()
        p_marg_add = p_marg_add.clamp(eps, 1.0 - eps)
        # KL(Bern(p)||Bern(0.5)) = p log(2p) + (1-p) log(2(1-p))
        L_marg = (p_marg_add * torch.log(2.0 * p_marg_add) + (1.0 - p_marg_add) * torch.log(2.0 * (1.0 - p_marg_add))).sum()

        with torch.no_grad():
            self._update_ema(p_marg)

        # InfoNCE diversity on bit-prob vectors
        z = p - p.mean(dim=-1, keepdim=True)
        z = F.normalize(z, dim=-1)
        z_flat = z[valid] if valid.any() else z.reshape(-1, Bits)
        M = int(z_flat.shape[0])
        if self.max_infonce_samples > 0 and M > self.max_infonce_samples:
            idx = torch.randperm(M, device=z_flat.device)[: self.max_infonce_samples]
            z_flat = z_flat[idx]
        L_info = info_nce_self(z_flat, temperature=self.infonce_temperature)

        stats = {
            "H_bit_mean": float((H[valid].mean() if valid.any() else H.mean()).detach().cpu().item()),
            "p_marg_mean": float(p_marg.mean().detach().cpu().item()),
            "valid_tokens": int(valid.sum().detach().cpu().item()),
        }
        return L_pos, L_marg, L_info, stats


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


def masked_mse(x_hat: torch.Tensor, x_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x_hat, x_true: [B,T,D]; mask: [B,T] bool."""
    mse = (x_hat - x_true).pow(2).mean(dim=-1)  # [B,T]
    if mask is None:
        return mse.mean()
    mask_f = mask.to(mse.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (mse * mask_f).sum() / denom


def is_power_of_two(x: int) -> bool:
    x = int(x)
    return x > 0 and (x & (x - 1)) == 0


# -------------------------
# Bottlenecks
# -------------------------


class BottleneckOutput:
    def __init__(
        self,
        z: torch.Tensor,
        ids: torch.Tensor,
        aux_losses: Dict[str, torch.Tensor],
        logits_for_reg: Optional[torch.Tensor] = None,
        bit_probs_for_reg: Optional[torch.Tensor] = None,
        slice_logits_for_reg: Optional[List[torch.Tensor]] = None,
    ) -> None:
        self.z = z
        self.ids = ids
        self.aux_losses = aux_losses
        self.logits_for_reg = logits_for_reg
        self.bit_probs_for_reg = bit_probs_for_reg
        self.slice_logits_for_reg = slice_logits_for_reg


class GumbelBottleneck(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        tie_logits_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.code_embed = nn.Embedding(self.vocab_size, int(d_model))
        self.to_logits = nn.Linear(int(d_model), self.vocab_size, bias=True)
        if tie_logits_embedding:
            self.to_logits.weight = self.code_embed.weight

    def forward(self, h: torch.Tensor, *, mode: str, tau: float, hard: bool) -> BottleneckOutput:
        logits = self.to_logits(h)  # [B,N,V]
        if mode == "train":
            z_onehot = F.gumbel_softmax(logits, tau=max(float(tau), 1e-6), hard=bool(hard), dim=-1)
            z = torch.einsum("bnv,vd->bnd", z_onehot, self.code_embed.weight)
        elif mode == "argmax":
            idx = torch.argmax(logits, dim=-1)
            z = self.code_embed(idx)
        elif mode == "soft":
            q = F.softmax(logits, dim=-1)
            z = torch.einsum("bnv,vd->bnd", q, self.code_embed.weight)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        ids = torch.argmax(logits, dim=-1)
        return BottleneckOutput(z=z, ids=ids, aux_losses={}, logits_for_reg=logits)


class VectorQuantizer(nn.Module):
    """Single-codebook VQ (gradient-based, straight-through), with optional orth regularizer."""

    def __init__(
        self,
        codebook_size: int,
        d_model: int,
        *,
        beta: float = 0.25,
        use_l2_norm: bool = False,
        orth_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.d_model = int(d_model)
        self.beta = float(beta)
        self.use_l2_norm = bool(use_l2_norm)
        self.orth_reg_weight = float(orth_reg_weight)
        self.codebook = nn.Embedding(self.codebook_size, self.d_model)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def _orth_loss(self) -> torch.Tensor:
        if self.orth_reg_weight <= 0.0:
            return self.codebook.weight.new_tensor(0.0)
        W = self.codebook.weight
        W = F.normalize(W, dim=-1)
        G = W @ W.t()
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        return ((G - I) ** 2).mean()

    def forward(self, z_e: torch.Tensor, *, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """z_e: [B,N,D]. Returns (z_q_st, ids, aux_losses, logits_for_reg(optional))."""
        B, N, D = z_e.shape
        flat = z_e.reshape(B * N, D)
        W = self.codebook.weight
        if self.use_l2_norm:
            flat_n = F.normalize(flat, dim=-1)
            W_n = F.normalize(W, dim=-1)
            # cosine distance: 2 - 2 cos
            dist = 2.0 - 2.0 * (flat_n @ W_n.t())
        else:
            flat_sq = (flat ** 2).sum(dim=1, keepdim=True)
            W_sq = (W ** 2).sum(dim=1).unsqueeze(0)
            dist = flat_sq + W_sq - 2.0 * (flat @ W.t())

        ids = torch.argmin(dist, dim=1)
        z_q = self.codebook(ids).view(B, N, D)

        # VQ losses
        loss_codebook = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        loss_vq = loss_codebook + self.beta * loss_commit
        loss_orth = self._orth_loss()
        loss_total = loss_vq + self.orth_reg_weight * loss_orth

        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        aux = {
            "vq_loss": loss_vq,
            "codebook_loss": loss_codebook,
            "commit_loss": loss_commit,
            "orth_loss": loss_orth,
            "vq_total": loss_total,
        }
        logits = None
        if return_logits:
            # Use negative distance as logits (higher is better).
            logits = (-dist).view(B, N, self.codebook_size)
        return z_q_st, ids.view(B, N), aux, logits


class VQVAEBottleneck(nn.Module):
    def __init__(
        self,
        d_model: int,
        codebook_size: int,
        *,
        beta: float = 0.25,
        use_l2_norm: bool = False,
        orth_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            d_model=d_model,
            beta=beta,
            use_l2_norm=use_l2_norm,
            orth_reg_weight=orth_reg_weight,
        )

    def forward(self, h: torch.Tensor, *, return_logits: bool = False) -> BottleneckOutput:
        z, ids, aux, logits = self.vq(h, return_logits=return_logits)
        return BottleneckOutput(z=z, ids=ids, aux_losses=aux, logits_for_reg=logits)


def _saturating_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # sigma'(x) = max(0, min(1, 1.2*sigmoid(x) - 0.1))
    return torch.clamp(1.2 * torch.sigmoid(x) - 0.1, min=0.0, max=1.0)


class ImprovedSemanticHashingBottleneck(nn.Module):
    """Improved Semantic Hashing (ISH) bottleneck.

    We follow the "improved semantic hashing" family described in:
      - Kaiser & Bengio, 2018 (Discrete Autoencoders for Sequence Models)
      - and referenced by Kaiser et al., 2018 (Fast Decoding...).

    The encoder hidden h [B,N,D] is projected to `num_bits` scalars, noise is added during training,
    passed through a saturating sigmoid, then rounded to bits. We use a straight-through estimator
    (optionally mixing soft/hard in the forward pass) and map bits back to a D-dimensional vector via
    an MLP bottleneck.
    """

    def __init__(
        self,
        d_model: int,
        num_bits: int,
        *,
        filter_size: int = 2048,
        noise_std: float = 1.0,
        forward_hard_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_bits = int(num_bits)
        self.filter_size = int(filter_size)
        self.noise_std = float(noise_std)
        self.forward_hard_rate = float(forward_hard_rate)

        self.to_bits = nn.Linear(self.d_model, self.num_bits)

        # bottleneck MLP similar to Kaiser & Bengio 2018 description
        self.ff1a = nn.Linear(self.num_bits, self.filter_size)
        self.ff1b = nn.Linear(self.num_bits, self.filter_size)
        self.ff2 = nn.Linear(self.filter_size, self.filter_size)
        self.ff3 = nn.Linear(self.filter_size, self.d_model)

    def _bits_to_embedding(self, bits: torch.Tensor) -> torch.Tensor:
        # bits: [B,N,b]
        h1a = self.ff1a(bits)
        h1b = self.ff1b(1.0 - bits)
        h2 = self.ff2(F.relu(h1a + h1b))
        out = self.ff3(F.relu(h2))
        return out

    @staticmethod
    def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
        """bits: [B,N,b] bool or 0/1 -> ids [B,N] int64 (little-endian)."""
        if bits.dtype != torch.long:
            b = bits.to(torch.long)
        else:
            b = bits
        B, N, nb = b.shape
        weights = (2 ** torch.arange(nb, device=b.device, dtype=torch.long)).view(1, 1, nb)
        return (b * weights).sum(dim=-1)

    def forward(self, h: torch.Tensor, *, mode: str) -> BottleneckOutput:
        # h: [B,N,D]
        logits_bit = self.to_bits(h)  # pre-sigmoid
        if mode == "train":
            noise = torch.randn_like(logits_bit) * self.noise_std
        else:
            noise = torch.zeros_like(logits_bit)

        p = _saturating_sigmoid(logits_bit + noise)  # [B,N,b]
        hard_bits = (p > 0.5).to(p.dtype)

        if mode == "train":
            # forward uses soft/hard mixture; backward always through p
            if self.forward_hard_rate <= 0.0:
                used = p
            elif self.forward_hard_rate >= 1.0:
                used = hard_bits
            else:
                bern = (torch.rand(p.shape[:2], device=p.device) < self.forward_hard_rate).to(p.dtype)
                bern = bern.unsqueeze(-1)
                used = bern * hard_bits + (1.0 - bern) * p
            bits_st = used + (p - p.detach())
            z = self._bits_to_embedding(bits_st)
        else:
            z = self._bits_to_embedding(hard_bits)

        ids = self.bits_to_int(hard_bits)
        aux: Dict[str, torch.Tensor] = {}
        return BottleneckOutput(z=z, ids=ids, aux_losses=aux, bit_probs_for_reg=p)


class DVQBottleneck(nn.Module):
    """Decomposed Vector Quantization (DVQ).

    Implements "sliced" DVQ (Section 2.4.1) and an optional "projected" variant (2.4.2)
    by quantizing multiple smaller subspaces and combining their indices.

    For a total vocabulary size K (=power-of-two), we allocate bits across n_slices
    so that product(K_i) == K.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_slices: int,
        *,
        mode: str = "sliced",  # sliced|projected
        beta: float = 0.25,
        use_l2_norm: bool = False,
        orth_reg_weight: float = 0.0,
        proj_seed: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.n_slices = int(n_slices)
        self.mode = str(mode)
        if self.mode not in {"sliced", "projected"}:
            raise ValueError("DVQ mode must be 'sliced' or 'projected'")
        if not is_power_of_two(self.vocab_size):
            raise ValueError("DVQ expects vocab_size to be a power of two (so it has an integer #bits)")
        if self.d_model % self.n_slices != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_slices ({self.n_slices})")

        total_bits = int(round(math.log2(self.vocab_size)))
        base = total_bits // self.n_slices
        rem = total_bits % self.n_slices
        bits_per = [base + (1 if i < rem else 0) for i in range(self.n_slices)]
        self.sizes = [2 ** b for b in bits_per]
        # offsets for packing indices into a single id
        offsets = [1]
        for s in self.sizes[:-1]:
            offsets.append(offsets[-1] * s)
        self.register_buffer("_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

        slice_dim = self.d_model // self.n_slices
        self.slice_dim = int(slice_dim)

        # One quantizer per slice
        self.codebooks = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=int(k),
                    d_model=self.slice_dim,
                    beta=beta,
                    use_l2_norm=use_l2_norm,
                    orth_reg_weight=orth_reg_weight,
                )
                for k in self.sizes
            ]
        )

        if self.mode == "projected":
            # fixed random projections pi^i in R^{D x (D/n)}
            g = torch.Generator()
            g.manual_seed(int(proj_seed))
            mats = []
            for _ in range(self.n_slices):
                A = torch.randn(self.d_model, self.slice_dim, generator=g)
                # QR for roughly-orthonormal columns
                Q, _ = torch.linalg.qr(A, mode="reduced")
                mats.append(Q[:, : self.slice_dim])
            proj = torch.stack(mats, dim=0)  # [S, D, slice_dim]
            self.register_buffer("proj", proj, persistent=False)

    def pack_ids(self, ids_per_slice: List[torch.Tensor]) -> torch.Tensor:
        # ids_per_slice: list of [B,N]
        out = ids_per_slice[0].to(torch.long) * 0
        for i, ids in enumerate(ids_per_slice):
            out = out + ids.to(torch.long) * self._offsets[i]
        return out

    def forward(self, h: torch.Tensor, *, return_logits: bool = False) -> BottleneckOutput:
        B, N, D = h.shape
        if D != self.d_model:
            raise ValueError("DVQ input dim mismatch")

        if self.mode == "sliced":
            # [B,N,S,sd]
            hs = h.view(B, N, self.n_slices, self.slice_dim)
        else:
            # projected: each slice sees a projected subspace
            # hs[i] = h @ proj[i]
            hs_list = []
            for i in range(self.n_slices):
                # [B,N,sd]
                hs_list.append(torch.einsum("bnd,ds->bns", h, self.proj[i]))
            hs = torch.stack(hs_list, dim=2)  # [B,N,S,sd]

        z_q_slices: List[torch.Tensor] = []
        ids_slices: List[torch.Tensor] = []
        aux_losses: Dict[str, torch.Tensor] = {}
        slice_logits: List[torch.Tensor] = []

        vq_total = h.new_tensor(0.0)
        for i, q in enumerate(self.codebooks):
            zi = hs[:, :, i, :]
            zq, ids, aux, logits = q(zi, return_logits=return_logits)
            z_q_slices.append(zq)
            ids_slices.append(ids)
            vq_total = vq_total + aux["vq_total"]
            # record slice losses
            aux_losses[f"vq_total_s{i}"] = aux["vq_total"].detach()
            if return_logits and logits is not None:
                slice_logits.append(logits)

        z_q = torch.cat(z_q_slices, dim=-1)  # [B,N,D]
        ids_packed = self.pack_ids(ids_slices)

        aux_losses["vq_total"] = vq_total
        return BottleneckOutput(
            z=z_q,
            ids=ids_packed,
            aux_losses=aux_losses,
            slice_logits_for_reg=slice_logits if return_logits else None,
        )


# -------------------------
# Full Autoencoder
# -------------------------


class MotionDiscreteAE(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        bottleneck: nn.Module,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.bottleneck = bottleneck

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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        mode: str,
        tau: float = 1.0,
        hard: bool = True,
        return_logits_for_reg: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, float], Optional[BottleneckOutput]]:
        """Returns:
        x_hat: [B,T_pad,D]
        patch_mask: [B,N]
        ids: [B,N]
        aux_losses: dict
        stats: dict
        bn_out: BottleneckOutput (optional)
        """

        x_patch, patch_mask, x_pad, mask_pad, T_pad = self.patchify(x, mask)
        h = self.in_proj(x_patch)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=~patch_mask)

        bn_out: Optional[BottleneckOutput] = None
        if isinstance(self.bottleneck, GumbelBottleneck):
            bn_out = self.bottleneck(h, mode=mode, tau=tau, hard=hard)
        elif isinstance(self.bottleneck, VQVAEBottleneck):
            bn_out = self.bottleneck(h, return_logits=return_logits_for_reg)
        elif isinstance(self.bottleneck, ImprovedSemanticHashingBottleneck):
            bn_out = self.bottleneck(h, mode=mode)
        elif isinstance(self.bottleneck, DVQBottleneck):
            bn_out = self.bottleneck(h, return_logits=return_logits_for_reg)
        else:
            raise TypeError(f"Unknown bottleneck type: {type(self.bottleneck)}")

        z = self.pos_enc(bn_out.z)
        y = self.decoder(z, src_key_padding_mask=~patch_mask)
        y_patch = self.out_proj(y)
        x_hat = self.unpatchify(y_patch, T_pad)

        # basic stats
        with torch.no_grad():
            ids = bn_out.ids
            ids_flat = ids[patch_mask] if patch_mask.any() else ids.reshape(-1)
            uniq = int(torch.unique(ids_flat).numel())
            stats = {
                "num_latent": int(patch_mask.shape[1]),
                "uniq_ids_batch": uniq,
                "valid_patches": int(patch_mask.sum().item()),
            }
        return x_hat, patch_mask, bn_out.ids, bn_out.aux_losses, stats, bn_out


# -------------------------
# Training / inference
# -------------------------


@dataclass
class TrainConfig:
    # data
    data_root: str
    save_dir: str
    motion_dir: str = "new_joint_vecs"
    max_motion_len: int = 64
    min_motion_len: int = 40
    random_crop: bool = True
    normalize: bool = True
    mean_path: str = ""
    std_path: str = ""
    # model
    patch_len: int = 1
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    ff_mult: int = 4
    bottleneck: str = "gumbel"  # gumbel|vqvae|ish|dvq
    vocab_size: int = 8192
    codebook_size: int = 8192
    # gumbel
    tau_start: float = 1.0
    tau_end: float = 0.3
    tau_anneal_steps: int = 20000
    gumbel_hard: bool = True
    kl_weight: float = 0.0
    kl_anneal_steps: int = 20000
    # vq/dvq
    beta: float = 0.25
    use_l2_norm: bool = False
    orth_reg_weight: float = 0.0
    # ish
    ish_bits: int = 13
    ish_filter_size: int = 2048
    ish_noise_std: float = 1.0
    ish_forward_hard_rate: float = 0.5
    # dvq
    dvq_slices: int = 4
    dvq_mode: str = "sliced"  # sliced|projected
    dvq_proj_seed: int = 0
    # proposal constraints
    proposal_enable: bool = False
    proposal_pos_w: float = 0.0
    proposal_marg_w: float = 0.0
    proposal_info_w: float = 0.0
    proposal_H_cap: float = 3.0
    proposal_tau_ent: float = 0.7
    proposal_tau_marg: float = 1.3
    proposal_ema_decay: float = 0.99
    proposal_alpha_mix: float = 0.5
    proposal_infonce_temp: float = 0.2
    proposal_infonce_max_samples: int = 1024
    # ish-specific proposal
    proposal_H_cap_bit: float = 0.3
    # optim
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    log_every: int = 50
    val_every: int = 1
    save_every: int = 1


def anneal_linear(step: int, start: float, end: float, steps: int) -> float:
    if steps <= 0:
        return float(end)
    t = min(max(step / steps, 0.0), 1.0)
    return float(start + (end - start) * t)


def build_model(cfg: TrainConfig, feat_dim: int) -> Tuple[MotionDiscreteAE, Optional[nn.Module]]:
    bn: nn.Module
    reg: Optional[nn.Module] = None

    if cfg.bottleneck == "gumbel":
        bn = GumbelBottleneck(cfg.d_model, cfg.vocab_size, tie_logits_embedding=True)
        if cfg.proposal_enable:
            reg = ProposalConstraintsSoftmax(
                vocab_size=cfg.vocab_size,
                H_cap=cfg.proposal_H_cap,
                tau_ent=cfg.proposal_tau_ent,
                tau_marg=cfg.proposal_tau_marg,
                ema_decay=cfg.proposal_ema_decay,
                alpha_mix_ema=cfg.proposal_alpha_mix,
                infonce_temperature=cfg.proposal_infonce_temp,
                max_infonce_samples=cfg.proposal_infonce_max_samples,
            )

    elif cfg.bottleneck == "vqvae":
        bn = VQVAEBottleneck(
            d_model=cfg.d_model,
            codebook_size=cfg.codebook_size,
            beta=cfg.beta,
            use_l2_norm=cfg.use_l2_norm,
            orth_reg_weight=cfg.orth_reg_weight,
        )
        if cfg.proposal_enable:
            reg = ProposalConstraintsSoftmax(
                vocab_size=cfg.codebook_size,
                H_cap=cfg.proposal_H_cap,
                tau_ent=cfg.proposal_tau_ent,
                tau_marg=cfg.proposal_tau_marg,
                ema_decay=cfg.proposal_ema_decay,
                alpha_mix_ema=cfg.proposal_alpha_mix,
                infonce_temperature=cfg.proposal_infonce_temp,
                max_infonce_samples=cfg.proposal_infonce_max_samples,
            )

    elif cfg.bottleneck == "ish":
        if not is_power_of_two(cfg.vocab_size):
            raise ValueError("ISH expects vocab_size to be power-of-two (K=2^bits)")
        bits = int(round(math.log2(cfg.vocab_size)))
        if cfg.ish_bits > 0:
            bits = int(cfg.ish_bits)
            if 2 ** bits != cfg.vocab_size:
                raise ValueError(f"ISH: vocab_size must equal 2^ish_bits. got vocab={cfg.vocab_size}, bits={bits}")
        bn = ImprovedSemanticHashingBottleneck(
            d_model=cfg.d_model,
            num_bits=bits,
            filter_size=cfg.ish_filter_size,
            noise_std=cfg.ish_noise_std,
            forward_hard_rate=cfg.ish_forward_hard_rate,
        )
        if cfg.proposal_enable:
            reg = ProposalConstraintsBernoulli(
                num_bits=bits,
                H_cap_bit=cfg.proposal_H_cap_bit,
                ema_decay=cfg.proposal_ema_decay,
                alpha_mix_ema=cfg.proposal_alpha_mix,
                infonce_temperature=cfg.proposal_infonce_temp,
                max_infonce_samples=cfg.proposal_infonce_max_samples,
            )

    elif cfg.bottleneck == "dvq":
        bn = DVQBottleneck(
            d_model=cfg.d_model,
            vocab_size=cfg.vocab_size,
            n_slices=cfg.dvq_slices,
            mode=cfg.dvq_mode,
            beta=cfg.beta,
            use_l2_norm=cfg.use_l2_norm,
            orth_reg_weight=cfg.orth_reg_weight,
            proj_seed=cfg.dvq_proj_seed,
        )
        if cfg.proposal_enable:
            # DVQ uses per-slice logits; we will apply reg per slice (same reg object reused).
            # We create one reg with max slice vocab size, but will recreate in training per slice size.
            reg = None

    else:
        raise ValueError(f"Unknown bottleneck: {cfg.bottleneck}")

    model = MotionDiscreteAE(
        feat_dim=feat_dim,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        ff_mult=cfg.ff_mult,
        bottleneck=bn,
    )
    return model, reg


def evaluate_loss(model: MotionDiscreteAE, loader: DataLoader, device: str) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for _, x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            x_hat, _, _, aux, _, _ = model(x, mask, mode="argmax", tau=1.0, hard=True, return_logits_for_reg=False)
            loss_rec = masked_mse(x_hat, x, mask)
            loss_aux = x_hat.new_tensor(0.0)
            if "vq_total" in aux:
                loss_aux = loss_aux + aux["vq_total"]
            losses.append(float((loss_rec + loss_aux).detach().cpu().item()))
    return float(np.mean(losses))


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    (save_dir / "logs").mkdir(exist_ok=True)

    # normalization
    mean_path = Path(cfg.mean_path) if cfg.mean_path else (save_dir / "norm" / "mean.npy")
    std_path = Path(cfg.std_path) if cfg.std_path else (save_dir / "norm" / "std.npy")
    if cfg.normalize:
        (save_dir / "norm").mkdir(exist_ok=True)
        if not mean_path.is_file() or not std_path.is_file():
            mean, std = compute_mean_std(Path(cfg.data_root), "train", cfg.motion_dir, None, max_files=0, seed=cfg.seed)
            np.save(str(mean_path), mean)
            np.save(str(std_path), std)
        else:
            mean = np.load(str(mean_path)).astype(np.float32)
            std = np.maximum(np.load(str(std_path)).astype(np.float32), 1e-6)
    else:
        mean = None
        std = None

    # datasets
    ds_train = HumanML3DMotionDataset(
        data_root=Path(cfg.data_root),
        split="train",
        motion_dir=cfg.motion_dir,
        max_motion_len=cfg.max_motion_len,
        min_motion_len=cfg.min_motion_len,
        random_crop=cfg.random_crop,
        normalize=cfg.normalize,
        mean_path=mean_path if cfg.normalize else None,
        std_path=std_path if cfg.normalize else None,
    )
    ds_val = HumanML3DMotionDataset(
        data_root=Path(cfg.data_root),
        split="val",
        motion_dir=cfg.motion_dir,
        max_motion_len=cfg.max_motion_len,
        min_motion_len=cfg.min_motion_len,
        random_crop=False,
        normalize=cfg.normalize,
        mean_path=mean_path if cfg.normalize else None,
        std_path=std_path if cfg.normalize else None,
    )

    feat_dim = int(ds_train.feat_dim)
    model, reg = build_model(cfg, feat_dim=feat_dim)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    if reg is not None:
        reg.to(device)

    # For DVQ, proposal reg is per-slice-size. We'll lazily create in training.
    dvq_regs: Optional[List[ProposalConstraintsSoftmax]] = None
    if cfg.bottleneck == "dvq" and cfg.proposal_enable:
        assert isinstance(model.bottleneck, DVQBottleneck)
        dvq_regs = []
        for k in model.bottleneck.sizes:
            dvq_regs.append(
                ProposalConstraintsSoftmax(
                    vocab_size=int(k),
                    H_cap=cfg.proposal_H_cap,
                    tau_ent=cfg.proposal_tau_ent,
                    tau_marg=cfg.proposal_tau_marg,
                    ema_decay=cfg.proposal_ema_decay,
                    alpha_mix_ema=cfg.proposal_alpha_mix,
                    infonce_temperature=cfg.proposal_infonce_temp,
                    max_infonce_samples=cfg.proposal_infonce_max_samples,
                ).to(device)
            )

    # dataloaders
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_motion, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_motion)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Save config
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    step = 0
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        running = []
        for it, (_, x, mask) in enumerate(dl_train, start=1):
            step += 1
            x = x.to(device)
            mask = mask.to(device)

            tau = anneal_linear(step, cfg.tau_start, cfg.tau_end, cfg.tau_anneal_steps) if cfg.bottleneck == "gumbel" else 1.0
            kl_w = anneal_linear(step, 0.0, cfg.kl_weight, cfg.kl_anneal_steps) if cfg.kl_weight > 0 else 0.0

            return_logits = bool(cfg.proposal_enable)
            x_hat, patch_mask, ids, aux, stats, bn_out = model(
                x,
                mask,
                mode="train" if cfg.bottleneck == "gumbel" else "argmax" if cfg.bottleneck == "vqvae" else "train",
                tau=tau,
                hard=cfg.gumbel_hard,
                return_logits_for_reg=return_logits,
            )

            loss_rec = masked_mse(x_hat, x, mask)
            loss = loss_rec

            # bottleneck auxiliary losses
            if "vq_total" in aux:
                loss = loss + aux["vq_total"]

            # KL-to-uniform for gumbel (optional baseline regularizer)
            kl_u = x_hat.new_tensor(0.0)
            if cfg.bottleneck == "gumbel" and cfg.kl_weight > 0.0:
                assert bn_out is not None
                q = F.softmax(bn_out.logits_for_reg, dim=-1)
                q_flat = q[patch_mask] if patch_mask.any() else q.reshape(-1, q.size(-1))
                p_marg = q_flat.mean(dim=0)
                U = torch.ones_like(p_marg) / float(p_marg.numel())
                kl_u = F.kl_div(p_marg.log().clamp_min(1e-12), U, reduction="sum")
                loss = loss + float(kl_w) * kl_u

            # Proposal constraints
            reg_pos = x_hat.new_tensor(0.0)
            reg_marg = x_hat.new_tensor(0.0)
            reg_info = x_hat.new_tensor(0.0)
            reg_stats: Dict[str, float] = {}
            if cfg.proposal_enable and bn_out is not None:
                if cfg.bottleneck in {"gumbel", "vqvae"}:
                    assert isinstance(reg, ProposalConstraintsSoftmax)
                    Lp, Lm, Li, reg_stats = reg(bn_out.logits_for_reg, mask=patch_mask)
                    reg_pos, reg_marg, reg_info = Lp, Lm, Li
                elif cfg.bottleneck == "ish":
                    assert isinstance(reg, ProposalConstraintsBernoulli)
                    Lp, Lm, Li, reg_stats = reg(bn_out.bit_probs_for_reg, mask=patch_mask)
                    reg_pos, reg_marg, reg_info = Lp, Lm, Li
                elif cfg.bottleneck == "dvq":
                    assert dvq_regs is not None
                    assert bn_out.slice_logits_for_reg is not None
                    for sl, reg_i in zip(bn_out.slice_logits_for_reg, dvq_regs):
                        Lp, Lm, Li, _ = reg_i(sl, mask=patch_mask)
                        reg_pos = reg_pos + Lp
                        reg_marg = reg_marg + Lm
                        reg_info = reg_info + Li
                else:
                    pass

                loss = loss + cfg.proposal_pos_w * reg_pos + cfg.proposal_marg_w * reg_marg + cfg.proposal_info_w * reg_info

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running.append(float(loss.detach().cpu().item()))

            if step % cfg.log_every == 0:
                row = {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(np.mean(running[-cfg.log_every :])),
                    "loss_rec": float(loss_rec.detach().cpu().item()),
                    "vq_total": float(aux.get("vq_total", x_hat.new_tensor(0.0)).detach().cpu().item()) if isinstance(aux, dict) else 0.0,
                    "kl_u": float(kl_u.detach().cpu().item()),
                    "tau": float(tau),
                    "stats": stats,
                    "reg_pos": float(reg_pos.detach().cpu().item()),
                    "reg_marg": float(reg_marg.detach().cpu().item()),
                    "reg_info": float(reg_info.detach().cpu().item()),
                    "reg_stats": reg_stats,
                    "time": human_time(time.time() - t0),
                }
                save_jsonl(save_dir / "logs" / "train_log.jsonl", row)

        # validation
        if epoch % cfg.val_every == 0:
            val_loss = evaluate_loss(model, dl_val, device=device)
            save_jsonl(save_dir / "logs" / "val_log.jsonl", {"epoch": epoch, "val_loss": val_loss, "time": human_time(time.time() - t0)})
            if val_loss < best_val:
                best_val = val_loss
                ckpt_best = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                    "step": step,
                    "best_val": best_val,
                }
                torch.save(ckpt_best, str(save_dir / "checkpoints" / "ckpt_best.pt"))

        # checkpoint last
        if epoch % cfg.save_every == 0:
            ckpt_last = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch,
                "step": step,
                "best_val": best_val,
            }
            torch.save(ckpt_last, str(save_dir / "checkpoints" / "ckpt_last.pt"))

    print(f"[done] training finished. best_val={best_val:.6f} save_dir={save_dir}")


def load_model_from_ckpt(ckpt_path: Path, device: str) -> Tuple[MotionDiscreteAE, TrainConfig, Optional[nn.Module]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    cfg = TrainConfig(**cfg_dict)
    # Build a dataset stub to get feat_dim
    ds = HumanML3DMotionDataset(
        data_root=Path(cfg.data_root),
        split="train",
        motion_dir=cfg.motion_dir,
        max_motion_len=cfg.max_motion_len,
        min_motion_len=cfg.min_motion_len,
        random_crop=False,
        normalize=False,
    )
    model, reg = build_model(cfg, feat_dim=int(ds.feat_dim))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    if reg is not None:
        reg.to(device)
        reg.eval()
    return model, cfg, reg


@torch.no_grad()
def encode_or_reconstruct(
    *,
    mode: str,
    ckpt_path: Path,
    data_root: Path,
    split: str,
    save_dir: Path,
    out_tokens_dir: Optional[Path],
    out_recon_dir: Optional[Path],
    batch_size: int,
    num_workers: int,
    device: str,
    save_input: bool,
) -> None:
    model, cfg, _ = load_model_from_ckpt(ckpt_path, device=device)

    mean_path = Path(cfg.mean_path) if cfg.mean_path else (save_dir / "norm" / "mean.npy")
    std_path = Path(cfg.std_path) if cfg.std_path else (save_dir / "norm" / "std.npy")

    ds = HumanML3DMotionDataset(
        data_root=Path(data_root),
        split=split,
        motion_dir=cfg.motion_dir,
        max_motion_len=cfg.max_motion_len,
        min_motion_len=cfg.min_motion_len,
        random_crop=False,
        normalize=cfg.normalize,
        mean_path=mean_path if cfg.normalize else None,
        std_path=std_path if cfg.normalize else None,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_motion)

    mean = np.load(str(mean_path)).astype(np.float32) if cfg.normalize and mean_path.is_file() else None
    std = np.maximum(np.load(str(std_path)).astype(np.float32), 1e-6) if cfg.normalize and std_path.is_file() else None

    if mode == "encode":
        assert out_tokens_dir is not None
        out_tokens_dir = Path(out_tokens_dir)
        out_tokens_dir.mkdir(parents=True, exist_ok=True)
    if mode == "reconstruct":
        assert out_recon_dir is not None
        out_recon_dir = Path(out_recon_dir)
        (out_recon_dir / split).mkdir(parents=True, exist_ok=True)
        if save_input:
            (out_recon_dir / "_input" / split).mkdir(parents=True, exist_ok=True)

    for mids, x, mask in dl:
        x = x.to(device)
        mask = mask.to(device)
        x_hat, _, ids, _, _, _ = model(x, mask, mode="argmax", tau=1.0, hard=True, return_logits_for_reg=False)

        # De-normalize if needed
        x_np = x.detach().cpu().numpy().astype(np.float32)
        xh_np = x_hat.detach().cpu().numpy().astype(np.float32)
        mask_np = mask.detach().cpu().numpy().astype(np.bool_)
        if cfg.normalize and mean is not None and std is not None:
            x_np = x_np * std + mean
            xh_np = xh_np * std + mean

        ids_np = ids.detach().cpu().numpy()
        for i, mid in enumerate(mids):
            valid_len = int(mask_np[i].sum())
            if mode == "encode":
                out_path = out_tokens_dir / f"{mid}.txt"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # Save only the latent tokens that correspond to valid (non-padded) frames.
                # valid_len is in frames; latent length is ceil(valid_len / patch_len).
                num_latent = int(math.ceil(valid_len / float(cfg.patch_len)))
                seq = ids_np[i].reshape(-1)[:num_latent]
                out_path.write_text(" ".join(map(str, seq.tolist())) + "\n", encoding="utf-8")
            else:
                out_path = out_recon_dir / split / f"{mid}.npy"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path), xh_np[i, :valid_len])
                if save_input:
                    in_path = out_recon_dir / "_input" / split / f"{mid}.npy"
                    in_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(in_path), x_np[i, :valid_len])

    print(f"[done] {mode} finished for split={split} save_dir={save_dir}")


# -------------------------
# CLI
# -------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--data_root", type=str, required=True)
    tr.add_argument("--save_dir", type=str, required=True)
    tr.add_argument("--motion_dir", type=str, default="new_joint_vecs")
    tr.add_argument("--max_motion_len", type=int, default=64)
    tr.add_argument("--min_motion_len", type=int, default=40)
    tr.add_argument("--no_random_crop", action="store_true")
    tr.add_argument("--normalize", action="store_true")
    tr.add_argument("--mean_path", type=str, default="")
    tr.add_argument("--std_path", type=str, default="")

    tr.add_argument("--patch_len", type=int, default=1)
    tr.add_argument("--d_model", type=int, default=512)
    tr.add_argument("--n_heads", type=int, default=8)
    tr.add_argument("--n_layers", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--ff_mult", type=int, default=4)

    tr.add_argument("--bottleneck", type=str, default="gumbel", choices=["gumbel", "vqvae", "ish", "dvq"])
    tr.add_argument("--vocab_size", type=int, default=8192, help="For gumbel/ish/dvq: total K")
    tr.add_argument("--codebook_size", type=int, default=8192, help="For vqvae: codebook size")

    tr.add_argument("--tau_start", type=float, default=1.0)
    tr.add_argument("--tau_end", type=float, default=0.3)
    tr.add_argument("--tau_anneal_steps", type=int, default=20000)
    tr.add_argument("--no_gumbel_hard", action="store_true")
    tr.add_argument("--kl_weight", type=float, default=0.0)
    tr.add_argument("--kl_anneal_steps", type=int, default=20000)

    tr.add_argument("--beta", type=float, default=0.25)
    tr.add_argument("--use_l2_norm", action="store_true")
    tr.add_argument("--orth_reg_weight", type=float, default=0.0)

    tr.add_argument("--ish_bits", type=int, default=13)
    tr.add_argument("--ish_filter_size", type=int, default=2048)
    tr.add_argument("--ish_noise_std", type=float, default=1.0)
    tr.add_argument("--ish_forward_hard_rate", type=float, default=0.5)

    tr.add_argument("--dvq_slices", type=int, default=4)
    tr.add_argument("--dvq_mode", type=str, default="sliced", choices=["sliced", "projected"])
    tr.add_argument("--dvq_proj_seed", type=int, default=0)

    tr.add_argument("--proposal_enable", action="store_true")
    tr.add_argument("--proposal_pos_w", type=float, default=0.0)
    tr.add_argument("--proposal_marg_w", type=float, default=0.0)
    tr.add_argument("--proposal_info_w", type=float, default=0.0)
    tr.add_argument("--proposal_H_cap", type=float, default=3.0)
    tr.add_argument("--proposal_tau_ent", type=float, default=0.7)
    tr.add_argument("--proposal_tau_marg", type=float, default=1.3)
    tr.add_argument("--proposal_ema_decay", type=float, default=0.99)
    tr.add_argument("--proposal_alpha_mix", type=float, default=0.5)
    tr.add_argument("--proposal_infonce_temp", type=float, default=0.2)
    tr.add_argument("--proposal_infonce_max_samples", type=int, default=1024)
    tr.add_argument("--proposal_H_cap_bit", type=float, default=0.3)

    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-2)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--num_workers", type=int, default=4)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", type=str, default="cuda")
    tr.add_argument("--log_every", type=int, default=50)
    tr.add_argument("--val_every", type=int, default=1)
    tr.add_argument("--save_every", type=int, default=1)

    # encode
    enc = sub.add_parser("encode")
    enc.add_argument("--data_root", type=str, required=True)
    enc.add_argument("--save_dir", type=str, required=True)
    enc.add_argument("--ckpt", type=str, required=True)
    enc.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    enc.add_argument("--out_tokens_dir", type=str, default="")
    enc.add_argument("--batch_size", type=int, default=64)
    enc.add_argument("--num_workers", type=int, default=4)
    enc.add_argument("--device", type=str, default="cuda")

    # reconstruct
    rec = sub.add_parser("reconstruct")
    rec.add_argument("--data_root", type=str, required=True)
    rec.add_argument("--save_dir", type=str, required=True)
    rec.add_argument("--ckpt", type=str, required=True)
    rec.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    rec.add_argument("--out_recon_dir", type=str, default="")
    rec.add_argument("--batch_size", type=int, default=64)
    rec.add_argument("--num_workers", type=int, default=4)
    rec.add_argument("--device", type=str, default="cuda")
    rec.add_argument("--save_input", action="store_true")

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            data_root=args.data_root,
            save_dir=args.save_dir,
            motion_dir=args.motion_dir,
            max_motion_len=int(args.max_motion_len),
            min_motion_len=int(args.min_motion_len),
            random_crop=not bool(args.no_random_crop),
            normalize=bool(args.normalize),
            mean_path=str(args.mean_path),
            std_path=str(args.std_path),
            patch_len=int(args.patch_len),
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            n_layers=int(args.n_layers),
            dropout=float(args.dropout),
            ff_mult=int(args.ff_mult),
            bottleneck=str(args.bottleneck),
            vocab_size=int(args.vocab_size),
            codebook_size=int(args.codebook_size),
            tau_start=float(args.tau_start),
            tau_end=float(args.tau_end),
            tau_anneal_steps=int(args.tau_anneal_steps),
            gumbel_hard=not bool(args.no_gumbel_hard),
            kl_weight=float(args.kl_weight),
            kl_anneal_steps=int(args.kl_anneal_steps),
            beta=float(args.beta),
            use_l2_norm=bool(args.use_l2_norm),
            orth_reg_weight=float(args.orth_reg_weight),
            ish_bits=int(args.ish_bits),
            ish_filter_size=int(args.ish_filter_size),
            ish_noise_std=float(args.ish_noise_std),
            ish_forward_hard_rate=float(args.ish_forward_hard_rate),
            dvq_slices=int(args.dvq_slices),
            dvq_mode=str(args.dvq_mode),
            dvq_proj_seed=int(args.dvq_proj_seed),
            proposal_enable=bool(args.proposal_enable),
            proposal_pos_w=float(args.proposal_pos_w),
            proposal_marg_w=float(args.proposal_marg_w),
            proposal_info_w=float(args.proposal_info_w),
            proposal_H_cap=float(args.proposal_H_cap),
            proposal_tau_ent=float(args.proposal_tau_ent),
            proposal_tau_marg=float(args.proposal_tau_marg),
            proposal_ema_decay=float(args.proposal_ema_decay),
            proposal_alpha_mix=float(args.proposal_alpha_mix),
            proposal_infonce_temp=float(args.proposal_infonce_temp),
            proposal_infonce_max_samples=int(args.proposal_infonce_max_samples),
            proposal_H_cap_bit=float(args.proposal_H_cap_bit),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            device=str(args.device),
            log_every=int(args.log_every),
            val_every=int(args.val_every),
            save_every=int(args.save_every),
        )
        train(cfg)

    elif args.cmd == "encode":
        save_dir = Path(args.save_dir)
        out_tokens = Path(args.out_tokens_dir) if args.out_tokens_dir else (save_dir / "tokens" / args.split)
        encode_or_reconstruct(
            mode="encode",
            ckpt_path=Path(args.ckpt),
            data_root=Path(args.data_root),
            split=str(args.split),
            save_dir=save_dir,
            out_tokens_dir=out_tokens,
            out_recon_dir=None,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"),
            save_input=False,
        )

    elif args.cmd == "reconstruct":
        save_dir = Path(args.save_dir)
        out_recon = Path(args.out_recon_dir) if args.out_recon_dir else (save_dir / "recon_out")
        encode_or_reconstruct(
            mode="reconstruct",
            ckpt_path=Path(args.ckpt),
            data_root=Path(args.data_root),
            split=str(args.split),
            save_dir=save_dir,
            out_tokens_dir=None,
            out_recon_dir=out_recon,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"),
            save_input=bool(args.save_input),
        )


if __name__ == "__main__":
    main()
