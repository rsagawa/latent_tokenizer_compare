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
import copy
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

from tqdm import tqdm

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


def _hamming_match_rate(a: List[int], b: List[int]) -> float:
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    min_len = min(len(a), len(b))
    matches = 0
    for i in range(min_len):
        if a[i] == b[i]:
            matches += 1
    return matches / float(max_len)


def _edit_match_rate(a: List[int], b: List[int], max_len_cap: int = 512) -> float:
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    if max_len > max_len_cap:
        a = a[:max_len_cap]
        b = b[:max_len_cap]
        max_len = max_len_cap

    # Levenshtein distance with O(min(n,m)) memory.
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, 1):
        curr = [j]
        for i, ai in enumerate(a, 1):
            cost = 0 if ai == bj else 1
            curr.append(min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost))
        prev = curr
    dist = prev[-1]
    return 1.0 - (dist / float(max_len))


def _update_ngram_counts(seq: List[int], n: int, counts: Dict[Tuple[int, ...], int]) -> None:
    if len(seq) < n:
        return
    end = len(seq) - n + 1
    for i in range(end):
        key = tuple(seq[i : i + n])
        counts[key] = counts.get(key, 0) + 1


def _ngram_l1_distance(a: Dict[Tuple[int, ...], int], b: Dict[Tuple[int, ...], int]) -> float:
    total_a = float(sum(a.values()))
    total_b = float(sum(b.values()))
    if total_a == 0.0 and total_b == 0.0:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    dist = 0.0
    for k in keys:
        pa = a.get(k, 0) / total_a if total_a > 0.0 else 0.0
        pb = b.get(k, 0) / total_b if total_b > 0.0 else 0.0
        dist += abs(pa - pb)
    return dist


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
      - top-K ID occupancy ratio (%): top1 / top10 / top100 / top1000
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
            top_counts = self.counts.to(torch.float32).sort(descending=True).values
            total = float(top_counts.sum().item())
            for k in (1, 10, 100, 1000):
                kk = min(k, top_counts.numel())
                occ = float(top_counts[:kk].sum().item()) / max(total, 1.0)
                out[f"top{k}_occupancy_pct"] = 100.0 * occ
        else:
            out["pmax"] = 0.0
            out["token_entropy"] = 0.0
            out["marg_entropy"] = 0.0
            out["eff_num_h1"] = 0.0
            out["hill2"] = 0.0
            out["unique_ids"] = 0.0
            out["top1_occupancy_pct"] = 0.0
            out["top10_occupancy_pct"] = 0.0
            out["top100_occupancy_pct"] = 0.0
            out["top1000_occupancy_pct"] = 0.0

        out["self_transition_rate"] = float(self.self_tr / self.n_pairs) if self.n_pairs > 0 else 0.0
        out["mean_latent_len"] = float(self.latent_len_sum / self.n_seqs) if self.n_seqs > 0 else 0.0
        return out

    def compute(self) -> Dict[str, float]:
        return self.finalize()

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


def _read_token_ids_txt(path: Path) -> List[int]:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if txt == "":
        return []
    return [int(s) for s in txt.split()]


class HumanML3DMotionTokenDataset(Dataset):
    """Paired motion + pre-encoded token IDs for B->A-only training."""

    def __init__(
        self,
        data_root: Path,
        token_root: Path,
        split: str,
        max_motion_len: int,
        latent_len_max: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        random_crop: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.token_root = Path(token_root)
        self.split = str(split)
        self.max_motion_len = int(max_motion_len)
        self.latent_len_max = int(latent_len_max)
        self.random_crop = bool(random_crop)
        self.mean = mean
        self.std = std

        motion_files = list_motion_files(self.data_root, self.split)
        self.motion_files: List[Path] = []
        self.token_files: List[Path] = []
        token_split_dir = self.token_root / self.split
        for mp in motion_files:
            seq_id = mp.stem
            cand = token_split_dir / f"{seq_id}.txt"
            if not cand.exists():
                cand = self.token_root / f"{seq_id}.txt"
            if cand.exists():
                self.motion_files.append(mp)
                self.token_files.append(cand)

        if len(self.motion_files) == 0:
            raise FileNotFoundError(
                f"No paired token files found under token_root={self.token_root} for split='{self.split}'. "
                f"Expected files like {self.token_root}/{self.split}/<id>.txt"
            )

    def __len__(self) -> int:
        return len(self.motion_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.motion_files[idx]
        tok_path = self.token_files[idx]

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

        ids = _read_token_ids_txt(tok_path)
        if len(ids) == 0:
            ids = [0]
        ids = ids[: self.latent_len_max]
        tlen = len(ids)
        token_ids = np.zeros((self.latent_len_max,), dtype=np.int64)
        token_mask = np.zeros((self.latent_len_max,), dtype=np.bool_)
        token_ids[:tlen] = np.asarray(ids, dtype=np.int64)
        token_mask[:tlen] = True

        return {
            "motion": torch.from_numpy(x_norm),      # [T,D]
            "mask": torch.from_numpy(mask),          # [T]
            "token_ids": torch.from_numpy(token_ids),   # [Tb]
            "token_mask": torch.from_numpy(token_mask), # [Tb]
            "id": torch.tensor([idx], dtype=torch.long),
            "path": str(path),
            "token_path": str(tok_path),
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
        self.register_buffer("div_term", div_term, persistent=False)

    def _build_dynamic_pe(self, positions: torch.Tensor, d_model: int) -> torch.Tensor:
        phase = positions.to(torch.float32).unsqueeze(-1) * self.div_term.view(*([1] * positions.ndim), -1)
        pe = torch.zeros((*positions.shape, d_model), dtype=torch.float32, device=positions.device)
        pe[..., 0::2] = torch.sin(phase)
        n_cos = pe[..., 1::2].shape[-1]
        if n_cos > 0:
            pe[..., 1::2] = torch.cos(phase[..., :n_cos])
        return pe

    def forward(
        self,
        x: torch.Tensor,
        offset: int | float | torch.Tensor = 0,
        pos_step: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """
        x: [B,L,D]
        """
        B, L, D = x.shape
        if L <= 0:
            return x

        use_fast = (
            not isinstance(offset, torch.Tensor)
            and not isinstance(pos_step, torch.Tensor)
            and float(pos_step) == 1.0
            and float(offset).is_integer()
        )
        if use_fast:
            off = int(float(offset))
            if off >= 0 and (off + L) <= int(self.pe.size(0)):
                return x + self.pe[off : off + L].unsqueeze(0).to(x.dtype)

        if isinstance(offset, torch.Tensor):
            off = offset.to(device=x.device, dtype=torch.float32)
            if off.ndim == 0:
                off = off.view(1).expand(B)
            elif off.ndim == 1 and off.numel() == 1:
                off = off.expand(B)
            elif off.ndim == 1 and off.numel() == B:
                pass
            else:
                raise ValueError(f"offset tensor must be scalar or [B], got {tuple(off.shape)} for batch={B}")
        else:
            off = torch.full((B,), float(offset), device=x.device, dtype=torch.float32)

        if isinstance(pos_step, torch.Tensor):
            step = pos_step.to(device=x.device, dtype=torch.float32)
            if step.ndim == 0:
                step = step.view(1).expand(B)
            elif step.ndim == 1 and step.numel() == 1:
                step = step.expand(B)
            elif step.ndim == 1 and step.numel() == B:
                pass
            else:
                raise ValueError(f"pos_step tensor must be scalar or [B], got {tuple(step.shape)} for batch={B}")
        else:
            step = torch.full((B,), float(pos_step), device=x.device, dtype=torch.float32)

        t = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(0)
        pos = off.unsqueeze(1) + step.unsqueeze(1) * t  # [B,L]
        pe_dyn = self._build_dynamic_pe(pos, d_model=D)
        return x + pe_dyn.to(x.dtype)


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
        if bank is None or bank.numel() == 0:
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


# -----------------------------
# Seq2Seq (AR) tokenizer model (two networks: networkA / networkB)
# -----------------------------

class DecoderLayerFlex(nn.Module):
    """Transformer block with causal self-attn + optional cross-attn.

    - Used as a *shared* backbone for both encoding (causal=False or True) and decoding (causal=True).
    - When memory is None, cross-attn is skipped.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        self_attn_drop_path: float = 0.0,
    ) -> None:
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
        p = float(self_attn_drop_path)
        self.self_attn_drop_path = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)

    def _apply_self_attn_drop_path(self, y: torch.Tensor) -> torch.Tensor:
        """Drop whole self-attn residual branch during training (stochastic depth)."""
        p = float(self.self_attn_drop_path)
        if (not self.training) or p <= 0.0:
            return y
        keep_prob = 1.0 - p
        if keep_prob <= 0.0:
            return torch.zeros_like(y)
        if torch.rand((), device=y.device) < p:
            return torch.zeros_like(y)
        return y / keep_prob

    def forward(
        self,
        x: torch.Tensor,                 # [B,L,D]
        *,
        causal: bool,
        self_key_padding_mask: Optional[torch.Tensor] = None,      # [B,L] True=pad
        memory: Optional[torch.Tensor] = None,                     # [B,S,D]
        memory_key_padding_mask: Optional[torch.Tensor] = None,    # [B,S] True=pad
        need_cross_attn: bool = False,
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
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        if memory is not None:
            y = self._apply_self_attn_drop_path(y)
        x = self.norm1(x + self.drop(y))

        # optional cross-attn
        w = None
        if memory is not None:
            y, w = self.cross_attn(
                x, memory, memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=bool(need_cross_attn),
                average_attn_weights=False,
            )
            x = self.norm2(x + self.drop(y))
        else:
            x = self.norm2(x)

        # ffn
        y = self.lin2(self.drop(self.act(self.lin1(x))))
        x = self.norm3(x + self.drop(y))
        return x, w  # w: [B,H,L,S] if requested

    def forward_one_step(
        self,
        x_cur: torch.Tensor,             # [B,1,D]
        *,
        past_key_value: Optional[torch.Tensor] = None,            # [B,Lpast,D]
        self_key_padding_mask: Optional[torch.Tensor] = None,     # [B,Lpast+1] True=pad
        memory: Optional[torch.Tensor] = None,                    # [B,S,D]
        memory_key_padding_mask: Optional[torch.Tensor] = None,   # [B,S] True=pad
        need_cross_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Incremental forward for one token with per-layer cache."""
        if x_cur.size(1) != 1:
            raise ValueError(f"forward_one_step expects L=1, got shape={tuple(x_cur.shape)}")

        # Cache stores layer inputs used as K/V for self-attention.
        if past_key_value is None:
            kv = x_cur
        else:
            kv = torch.cat([past_key_value, x_cur], dim=1)

        # self-attn (query length = 1; causal mask unnecessary with prefix-only K/V)
        y, _ = self.self_attn(
            x_cur, kv, kv,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        if memory is not None:
            y = self._apply_self_attn_drop_path(y)
        x = self.norm1(x_cur + self.drop(y))

        # optional cross-attn
        w = None
        if memory is not None:
            y, w = self.cross_attn(
                x, memory, memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=bool(need_cross_attn),
                average_attn_weights=False,
            )
            x = self.norm2(x + self.drop(y))
        else:
            x = self.norm2(x)

        # ffn
        y = self.lin2(self.drop(self.act(self.lin1(x))))
        x = self.norm3(x + self.drop(y))
        return x, w, kv



class ARFlexNetwork(nn.Module):
    """Shared AR transformer backbone for both networkA (motion) and networkB (tokens).

    - kind="cont": motion patches in/out transformer used for both encoding and decoding.
    - kind="token": token-embedding in / token-logits out transformer used for both encoding and AR generation.
    """

    def __init__(
        self,
        *,
        kind: str,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        self_attn_drop_path: float = 0.0,
        patch_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        token_out_patch_dim: Optional[int] = None,
        to_logits_vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        kind = str(kind).lower().strip()
        if kind not in {"cont", "token"}:
            raise ValueError(f"Unknown kind: {kind} (expected 'cont' or 'token')")
        self.kind = kind
        self.d_model = int(d_model)

        # per-kind IO
        self.patch_dim: Optional[int] = None
        self.vocab_size: Optional[int] = None

        if self.kind == "cont":
            if patch_dim is None:
                raise ValueError("patch_dim is required for kind='cont'")
            self.patch_dim = int(patch_dim)
            self.in_proj = nn.Linear(self.patch_dim, self.d_model)
            self.out_proj = nn.Linear(self.d_model, self.patch_dim)

            # BOS for AR generation / shifted decoding
            self.start_token = nn.Parameter(torch.zeros(self.d_model))

            # (No separate feedback projector: we feed back via in_proj(out_proj(h_last)).)

            # optional logits head (A->B parallel path)
            if to_logits_vocab_size is not None:
                self.to_logits = nn.Linear(self.d_model, int(to_logits_vocab_size))
            else:
                self.to_logits = None

            # placeholders to keep mypy happy (not used in this kind)
            self.token_embed = None
        else:
            if vocab_size is None:
                raise ValueError("vocab_size is required for kind='token'")
            self.vocab_size = int(vocab_size)
            self.token_embed = nn.Embedding(self.vocab_size, self.d_model)

            # BOS (start) embedding for token generation
            self.start_token = nn.Parameter(torch.zeros(self.d_model))

            self.to_logits = nn.Linear(self.d_model, self.vocab_size)
            self.out_proj = nn.Linear(self.d_model, int(token_out_patch_dim)) if token_out_patch_dim is not None else None

            # placeholders (not used in this kind)
            self.in_proj = None
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                DecoderLayerFlex(
                    self.d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    self_attn_drop_path=self_attn_drop_path,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.ln_out = nn.LayerNorm(self.d_model)

    def _topk_mask(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        v, _ = torch.topk(logits, top_k, dim=-1)
        thr = v[..., -1].unsqueeze(-1)
        return logits.masked_fill(logits < thr, -1.0e9)

    def _resolve_recent_kv_steps(self, recent_kv_frames: int, compression_rate: float = 1.0) -> int:
        """Map frame-based recent window to step count for this network kind.

        - kind='cont' : 1 step ~= 1 frame-unit given by caller (typically patch step).
        - kind='token': scale by compression_rate (B covers wider temporal span).
        """
        n = int(recent_kv_frames)
        if n <= 0:
            return 0
        if self.kind == "token":
            ratio = max(float(compression_rate), 1e-6)
            n = int(math.ceil(float(n) / ratio))
        return max(1, n)

    @staticmethod
    def _trim_past_key_value(past_key_value: Optional[torch.Tensor], keep_steps: int) -> Optional[torch.Tensor]:
        if past_key_value is None or keep_steps <= 0:
            return past_key_value
        if past_key_value.size(1) <= keep_steps:
            return past_key_value
        return past_key_value[:, -keep_steps:, :]

    def encode(
        self,
        x: torch.Tensor,                   # cont: [B,N,PD] / token: [B,T,D]
        mask: Optional[torch.Tensor],      # [B,N] or [B,T] (True=valid)
        *,
        causal: bool = True,
        pos_step: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Encode a sequence into hidden states [B,L,D].

        - cont: x is patch vectors, projected by in_proj.
        - token: x is *token embeddings* (already in d_model).
        """
        if self.kind == "cont":
            if self.in_proj is None:
                raise RuntimeError("in_proj missing for kind='cont'")
            x = self.in_proj(x)
        x = self.pos_enc(x, offset=0, pos_step=pos_step)

        if mask is None:
            self_kpm = None
        else:
            self_kpm = ~mask.to(torch.bool)  # True=pad

        for layer in self.layers:
            x, _ = layer(
                x,
                causal=bool(causal),
                self_key_padding_mask=self_kpm,
                memory=None,
                memory_key_padding_mask=None,
                need_cross_attn=False,
            )
        return self.ln_out(x)

    def teacher_forcing_decode(
        self,
        x_patch: torch.Tensor,                              # [B,N,PD]
        patch_mask: torch.Tensor,                           # [B,N] bool
        *,
        memory: Optional[torch.Tensor],                     # [B,S,D]
        memory_key_padding_mask: Optional[torch.Tensor],    # [B,S] True=pad
        return_attn: bool,
        return_hidden: bool,
        attn_apply: str = "all",
        input_drop_prob: float = 0.0,
        input_drop_mode: str = "prob",
        input_drop_mse_thresh: float = 0.0,
        recent_kv_frames: int = 0,
        compression_rate: float = 1.0,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], torch.Tensor]:
        """Teacher-forcing prediction for kind='cont' (motion).

        Shifted input: [<s>, x0, x1, ..., x_{N-2}] -> predict [x0, x1, ..., x_{N-1}]
        """
        if self.kind != "cont":
            raise RuntimeError("teacher_forcing_decode is only supported for kind='cont'")
        if self.in_proj is None or self.out_proj is None or self.start_token is None:
            raise RuntimeError("cont projections/tokens are not initialized")

        Bsz, N, PD = x_patch.shape
        x_emb = self.in_proj(x_patch.detach())  # [B,N,D]

        # shifted input
        s = self.start_token.view(1, 1, self.d_model).expand(Bsz, 1, self.d_model)
        dec_in = torch.cat([s, x_emb[:, :-1, :]], dim=1)  # [B,N,D]

        # input keep mask aligned to target positions
        inp_mask = torch.zeros_like(patch_mask, dtype=torch.bool)
        if inp_mask.numel() > 0:
            inp_mask[:, 0] = True
            if N > 1:
                inp_mask[:, 1:] = patch_mask[:, :-1]

        # Optional input-dropout on teacher-forcing inputs:
        # build shifted inputs first (drop positions use one-step free-run feedback),
        # then run a standard teacher-forcing pass to collect training outputs/attention.
        drop_p = float(input_drop_prob)
        drop_p = 0.0 if drop_p < 0.0 else (1.0 if drop_p > 1.0 else drop_p)
        drop_mode = str(input_drop_mode).lower().strip()
        if drop_mode not in {"prob", "mse_thresh"}:
            raise ValueError(f"Unknown input_drop_mode: {input_drop_mode} (expected 'prob' or 'mse_thresh')")
        mse_thresh = float(input_drop_mse_thresh)
        if mse_thresh < 0.0:
            mse_thresh = 0.0
        fr_accept_num = torch.zeros((), device=x_patch.device, dtype=torch.float32)
        fr_accept_den = torch.zeros((), device=x_patch.device, dtype=torch.float32)
        recent_kv_steps = self._resolve_recent_kv_steps(recent_kv_frames, compression_rate=compression_rate)
        if (drop_p > 0.0 or drop_mode == "mse_thresh") and N > 1:
            valid_prev = patch_mask[:, :-1].to(torch.bool)  # input positions tied to x_t
            if drop_mode == "prob":
                if drop_p >= 1.0:
                    # time-shared decision: all samples use free-run at the same t
                    drop_steps = torch.ones((N - 1,), device=x_patch.device, dtype=torch.bool)
                else:
                    # time-shared Bernoulli over t (not per-sample)
                    drop_steps = (torch.rand((N - 1,), device=x_patch.device) < drop_p)
            else:
                drop_steps = None

            # Per-layer cache of decoder self-attention K/V ("past_key_values").
            past_key_values: List[Optional[torch.Tensor]] = [None for _ in range(len(self.layers))]
            emb_in = s  # [B,1,D], token placed at position t
            with torch.no_grad():
                for t in range(N - 1):
                    inp_mask_t = torch.cat(
                        [
                            torch.ones((Bsz, 1), device=patch_mask.device, dtype=torch.bool),
                            patch_mask[:, :t].to(torch.bool),
                        ],
                        dim=1,
                    )  # [B, t+1] for prefix [<s>, x0, ..., x_{t-1}]

                    x_cur = self.pos_enc(emb_in, offset=t)  # [B,1,D]
                    next_past_key_values: List[Optional[torch.Tensor]] = []
                    for li, layer in enumerate(self.layers):
                        past_kv = self._trim_past_key_value(past_key_values[li], keep_steps=recent_kv_steps)
                        if past_kv is None:
                            self_kpm = ~inp_mask_t[:, -1:]
                        else:
                            self_kpm = ~inp_mask_t[:, -(past_kv.size(1) + 1):]
                        x_cur, _, kv_cur = layer.forward_one_step(
                            x_cur,
                            past_key_value=past_kv,
                            self_key_padding_mask=self_kpm,
                            memory=memory,
                            memory_key_padding_mask=memory_key_padding_mask,
                            need_cross_attn=False,
                        )
                        next_past_key_values.append(kv_cur.detach())
                    past_key_values = next_past_key_values

                    h_last = self.ln_out(x_cur[:, 0, :])  # [B,D]
                    y_t = self.out_proj(h_last)           # [B,PD]
                    emb_fb = self.in_proj(y_t)            # [B,D]

                    valid_t = valid_prev[:, t]  # [B]
                    fr_accept_den = fr_accept_den + valid_t.to(torch.float32).sum()
                    if drop_mode == "prob":
                        dropped_t = bool(drop_steps[t].item())
                        if dropped_t:
                            use_free = valid_t
                        else:
                            use_free = torch.zeros_like(valid_t)
                    else:
                        mse_t = (y_t - x_patch[:, t, :]).pow(2).sum(dim=-1)  # [B]
                        use_free = valid_t & (mse_t <= mse_thresh)
                    fr_accept_num = fr_accept_num + use_free.to(torch.float32).sum()

                    # keep GT on invalid slots (and on non-accepted slots)
                    emb_next = torch.where(use_free.unsqueeze(1), emb_fb, x_emb[:, t, :])

                    dec_in[:, t + 1, :] = emb_next
                    emb_in = emb_next.unsqueeze(1).detach()

        x = self.pos_enc(dec_in, offset=0)

        attn_layers: List[Optional[torch.Tensor]] = []
        hidden_layers: List[Optional[torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            need_attn = bool(return_attn and _layer_selected(li, len(self.layers), attn_apply))
            x, w = layer(
                x,
                causal=True,
                self_key_padding_mask=~inp_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                need_cross_attn=need_attn,
            )
            if return_attn:
                attn_layers.append(w if need_attn else None)
            if return_hidden:
                hidden_layers.append(x)

        y = self.out_proj(self.ln_out(x))  # [B,N,PD]
        fr_accept_rate = fr_accept_num / fr_accept_den.clamp_min(1.0)
        return y, attn_layers, hidden_layers, fr_accept_rate

    def teacher_forcing_token_decode(
        self,
        token_ids: torch.Tensor,                            # [B,T]
        token_mask: torch.Tensor,                           # [B,T] bool
        *,
        memory: Optional[torch.Tensor],                     # [B,S,D]
        memory_key_padding_mask: Optional[torch.Tensor],    # [B,S] True=pad
        return_attn: bool,
        return_hidden: bool,
        attn_apply: str = "all",
        input_drop_prob: float = 0.0,
        tau: float = 1.0,
        scale: float = 1.0,
        hard: bool = True,
        top_k: int = 0,
        deterministic: bool = True,
        embed_mode: str = "sample",
        pos_step: float | torch.Tensor = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Teacher-forcing token prediction for kind='token'.

        Shifted input: [<s>, b0, b1, ..., b_{T-2}] -> predict [b0, b1, ..., b_{T-1}]
        """
        if self.kind != "token":
            raise RuntimeError("teacher_forcing_token_decode is only supported for kind='token'")
        if self.token_embed is None or self.to_logits is None or self.start_token is None:
            raise RuntimeError("token modules are not initialized")

        Bsz, T = token_ids.shape
        tok_emb = self.token_embed(token_ids.to(torch.long)).detach()  # [B,T,D]
        s = self.start_token.view(1, 1, self.d_model).expand(Bsz, 1, self.d_model)
        dec_in = torch.cat([s, tok_emb[:, :-1, :]], dim=1)  # [B,T,D]
        embed_mode = str(embed_mode).lower().strip()

        inp_mask = torch.zeros_like(token_mask, dtype=torch.bool)
        if inp_mask.numel() > 0:
            inp_mask[:, 0] = True
            if T > 1:
                inp_mask[:, 1:] = token_mask[:, :-1]

        # Optional input-dropout on teacher-forcing token inputs:
        # drop positions use one-step AR feedback (argmax token embedding).
        drop_p = float(input_drop_prob)
        drop_p = 0.0 if drop_p < 0.0 else (1.0 if drop_p > 1.0 else drop_p)
        if drop_p > 0.0 and T > 1:
            valid_prev = token_mask[:, :-1].to(torch.bool)
            if drop_p >= 1.0:
                drop_steps = torch.ones((T - 1,), device=token_ids.device, dtype=torch.bool)
            else:
                drop_steps = (torch.rand((T - 1,), device=token_ids.device) < drop_p)

            past_key_values: List[Optional[torch.Tensor]] = [None for _ in range(len(self.layers))]
            emb_in = s  # [B,1,D]
            with torch.no_grad():
                for t in range(T - 1):
                    inp_mask_t = torch.cat(
                        [
                            torch.ones((Bsz, 1), device=token_mask.device, dtype=torch.bool),
                            token_mask[:, :t].to(torch.bool),
                        ],
                        dim=1,
                    )  # [B, t+1] for prefix [<s>, b0, ..., b_{t-1}]

                    if isinstance(pos_step, torch.Tensor):
                        step_off = pos_step.to(device=emb_in.device, dtype=torch.float32) * float(t)
                    else:
                        step_off = float(pos_step) * float(t)
                    x_cur = self.pos_enc(emb_in, offset=step_off, pos_step=1.0)  # [B,1,D]
                    next_past_key_values: List[Optional[torch.Tensor]] = []
                    for li, layer in enumerate(self.layers):
                        past_kv = past_key_values[li]
                        if past_kv is None:
                            self_kpm = ~inp_mask_t[:, -1:]
                        else:
                            self_kpm = ~inp_mask_t[:, -(past_kv.size(1) + 1):]
                        x_cur, _, kv_cur = layer.forward_one_step(
                            x_cur,
                            past_key_value=past_kv,
                            self_key_padding_mask=self_kpm,
                            memory=memory,
                            memory_key_padding_mask=memory_key_padding_mask,
                            need_cross_attn=False,
                        )
                        next_past_key_values.append(kv_cur.detach())
                    past_key_values = next_past_key_values

                    h_last = self.ln_out(x_cur[:, 0, :])   # [B,D]
                    logits_t = self.to_logits(h_last) * float(scale)  # [B,V]
                    logits_t = self._topk_mask(logits_t, int(top_k))
                    if embed_mode == "softmax":
                        probs_t = F.softmax(logits_t / max(float(tau), 1e-6), dim=-1)
                        ids_fb = probs_t.argmax(dim=-1)
                        emb_fb = probs_t @ self.token_embed.weight
                    elif embed_mode == "sample":
                        if bool(deterministic):
                            ids_fb = logits_t.argmax(dim=-1)
                            onehot_t = F.one_hot(ids_fb, num_classes=self.vocab_size).to(dtype=logits_t.dtype)
                        else:
                            onehot_t = F.gumbel_softmax(logits_t, tau=max(float(tau), 1e-6), hard=bool(hard), dim=-1)
                            ids_fb = onehot_t.argmax(dim=-1)
                        emb_fb = onehot_t @ self.token_embed.weight
                    else:
                        raise ValueError(f"Unknown embed_mode: {embed_mode} (expected 'sample' or 'softmax')")

                    dropped_t = bool(drop_steps[t].item())
                    if dropped_t:
                        use_free = valid_prev[:, t]
                    else:
                        use_free = torch.zeros_like(valid_prev[:, t])
                    emb_next = torch.where(use_free.unsqueeze(1), emb_fb, tok_emb[:, t, :])

                    dec_in[:, t + 1, :] = emb_next
                    emb_in = emb_next.unsqueeze(1).detach()

        x = self.pos_enc(dec_in, offset=0, pos_step=pos_step)
        attn_layers: List[Optional[torch.Tensor]] = []
        hidden_layers: List[Optional[torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            need_attn = bool(return_attn and _layer_selected(li, len(self.layers), attn_apply))
            x, w = layer(
                x,
                causal=True,
                self_key_padding_mask=~inp_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                need_cross_attn=need_attn,
            )
            if return_attn:
                attn_layers.append(w if need_attn else None)
            if return_hidden:
                hidden_layers.append(x)

        logits = self.to_logits(self.ln_out(x)) * float(scale)  # [B,T,V]
        logits = self._topk_mask(logits, int(top_k))
        if embed_mode == "softmax":
            probs = F.softmax(logits / max(float(tau), 1e-6), dim=-1)
            token_ids = probs.argmax(dim=-1)
            token_embs = probs @ self.token_embed.weight
        elif embed_mode == "sample":
            if bool(deterministic):
                token_ids = logits.argmax(dim=-1)
                onehot = F.one_hot(token_ids, num_classes=self.vocab_size).to(dtype=logits.dtype)
            else:
                onehot = F.gumbel_softmax(logits, tau=max(float(tau), 1e-6), hard=bool(hard), dim=-1)
                token_ids = onehot.argmax(dim=-1)
            token_embs = onehot @ self.token_embed.weight
        else:
            raise ValueError(f"Unknown embed_mode: {embed_mode} (expected 'sample' or 'softmax')")
        return token_ids, logits, token_embs, attn_layers, hidden_layers

    def generate(
        self,
        memory: torch.Tensor,                             # [B,S,D]
        memory_key_padding_mask: Optional[torch.Tensor],   # [B,S] True=pad
        T: int,
        *,
        # token-only args (ignored for kind='cont')
        tau: float = 1.0,
        scale: float = 1.0,
        hard: bool = True,
        top_k: int = 0,
        deterministic: bool = True,
        embed_mode: str = "sample",
        # common args
        return_attn: bool = False,
        return_hidden: bool = False,
        attn_apply: str = "all",   # all|mid|last
        recent_kv_frames: int = 0,
        compression_rate: float = 1.0,
        pos_step: float | torch.Tensor = 1.0,
    ) -> Any:
        """Autoregressively generate a sequence conditioned on `memory`.

        - kind='token': generate discrete token IDs B (plus logits/embeddings).
          Uses a learned BOS embedding (`start_token`) and the token embedding table.

        - kind='cont' : generate continuous motion patches A.
          Uses a learned BOS embedding (`start_token`). For feedback, uses
          `in_proj(out_proj(h_last))` (i.e., re-embed the predicted patch).

        Notes on prefix growth
        ----------------------
        Uses per-layer `past_key_values` cache for incremental decoding.
        """

        Bsz = memory.size(0)
        T = int(T)
        if T <= 0:
            raise ValueError("T must be >= 1")

        is_token = (self.kind == "token")

        if is_token:
            if self.token_embed is None or self.to_logits is None or self.start_token is None or self.vocab_size is None:
                raise RuntimeError("token modules are not initialized")
        else:
            if self.in_proj is None or self.out_proj is None or self.start_token is None:
                raise RuntimeError("cont projections/tokens are not initialized")

        # incremental input starts from BOS
        emb_in = self.start_token.view(1, 1, self.d_model).expand(Bsz, 1, self.d_model)  # [B,1,D]
        past_key_values: List[Optional[torch.Tensor]] = [None for _ in range(len(self.layers))]

        # outputs
        out_ids: List[torch.Tensor] = []
        out_logits: List[torch.Tensor] = []
        out_embs: List[torch.Tensor] = []
        out_patches: List[torch.Tensor] = []

        # per-step attn/hidden capture (last query row only)
        attn_rows_per_layer: List[List[torch.Tensor]] = [[] for _ in range(len(self.layers))]
        hid_rows_per_layer: Optional[List[List[torch.Tensor]]] = None
        if return_hidden:
            hid_rows_per_layer = [[] for _ in range(len(self.layers))]

        embed_mode = str(embed_mode).lower().strip()
        recent_kv_steps = self._resolve_recent_kv_steps(recent_kv_frames, compression_rate=compression_rate)

        for _t in range(T):
            if is_token:
                if isinstance(pos_step, torch.Tensor):
                    step_off = pos_step.to(device=emb_in.device, dtype=torch.float32) * float(_t)
                else:
                    step_off = float(pos_step) * float(_t)
                x_cur = self.pos_enc(emb_in, offset=step_off, pos_step=1.0)  # [B,1,D]
            else:
                x_cur = self.pos_enc(emb_in, offset=_t)  # [B,1,D]
            next_past_key_values: List[Optional[torch.Tensor]] = []

            for li, layer in enumerate(self.layers):
                need_attn = bool(return_attn and _layer_selected(li, len(self.layers), attn_apply))
                past_kv = self._trim_past_key_value(past_key_values[li], keep_steps=recent_kv_steps)
                x_cur, w, kv_cur = layer.forward_one_step(
                    x_cur,
                    past_key_value=past_kv,
                    self_key_padding_mask=None,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    need_cross_attn=need_attn,
                )
                next_past_key_values.append(kv_cur.detach())
                if need_attn and w is not None:
                    attn_rows_per_layer[li].append(w[:, :, 0, :])  # [B,H,S]
                if hid_rows_per_layer is not None:
                    hid_rows_per_layer[li].append(x_cur[:, 0, :])  # [B,D]
            past_key_values = next_past_key_values

            h_last = self.ln_out(x_cur[:, 0, :])  # [B,D]

            if is_token:
                logits = self.to_logits(h_last) * float(scale)  # [B,V]
                logits = self._topk_mask(logits, int(top_k))

                if embed_mode == "softmax":
                    probs = F.softmax(logits / max(float(tau), 1e-6), dim=-1)
                    ids = probs.argmax(dim=-1)
                    emb_next = probs @ self.token_embed.weight
                elif embed_mode == "sample":
                    if bool(deterministic):
                        ids = logits.argmax(dim=-1)
                        onehot = F.one_hot(ids, num_classes=self.vocab_size).to(dtype=logits.dtype)
                    else:
                        # straight-through Gumbel-Softmax
                        onehot = F.gumbel_softmax(logits, tau=max(float(tau), 1e-6), hard=bool(hard), dim=-1)
                        ids = onehot.argmax(dim=-1)
                    emb_next = onehot @ self.token_embed.weight
                else:
                    raise ValueError(f"Unknown embed_mode: {embed_mode} (expected 'sample' or 'softmax')")

                out_ids.append(ids)
                out_logits.append(logits)
                out_embs.append(emb_next)
            else:
                y = self.out_proj(h_last)  # [B,PD]
                out_patches.append(y)
                emb_next = self.in_proj(y)  # [B,D]  (re-embed predicted patch)

            # feed next token embedding only (prefix stays in cache)
            emb_in = emb_next.unsqueeze(1).detach()

        # pack attention/hidden logs
        attn_layers: List[Optional[torch.Tensor]] = []
        if return_attn:
            for li in range(len(self.layers)):
                if len(attn_rows_per_layer[li]) == 0:
                    attn_layers.append(None)
                else:
                    attn_layers.append(torch.stack(attn_rows_per_layer[li], dim=2))  # [B,H,T,S]

        hidden_layers: List[Optional[torch.Tensor]] = []
        if hid_rows_per_layer is not None:
            for li in range(len(self.layers)):
                if len(hid_rows_per_layer[li]) == 0:
                    hidden_layers.append(None)
                else:
                    hidden_layers.append(torch.stack(hid_rows_per_layer[li], dim=1))  # [B,T,D]

        if is_token:
            token_ids = torch.stack(out_ids, dim=1) if len(out_ids) > 0 else torch.zeros((Bsz, 0), device=memory.device, dtype=torch.long)
            token_logits = torch.stack(out_logits, dim=1) if len(out_logits) > 0 else torch.zeros((Bsz, 0, int(self.vocab_size)), device=memory.device, dtype=torch.float32)
            token_embs = torch.stack(out_embs, dim=1) if len(out_embs) > 0 else torch.zeros((Bsz, 0, self.d_model), device=memory.device, dtype=torch.float32)
            return token_ids, token_logits, token_embs, attn_layers, hidden_layers

        x_patch_hat = torch.stack(out_patches, dim=1) if len(out_patches) > 0 else torch.zeros((Bsz, 0, int(self.patch_dim)), device=memory.device, dtype=torch.float32)
        return x_patch_hat, attn_layers, hidden_layers



class MotionSeq2SeqARAE(nn.Module):
    """Two-network (networkA/networkB) seq2seq discrete autoencoder for motion.

    A -> B:
      - encode A with networkA (causal self-attn)
      - generate B autoregressively with networkB, cross-attending to hiddenA

    B -> A:
      - encode B with networkB (causal self-attn)
      - predict A with teacher forcing using networkA, cross-attending to hiddenB
    """

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
        self_attn_drop_path: float = 0.0,
        gauss_cfg: Optional[Dict] = None,
        softptr_cfg: Optional[Dict] = None,
        past_kv_recent_frames: int = 0,
        token_posenc_scale_with_compression: bool = False,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.patch_len = int(patch_len)
        self.vocab_size = int(vocab_size)
        self.compression_ratio = float(compression_ratio)
        self.past_kv_recent_frames = int(past_kv_recent_frames)
        self.latent_len_max = int(latent_len_max)
        self.d_model = int(d_model)
        self.max_motion_len = int(max_motion_len)
        self.token_posenc_scale_with_compression = bool(token_posenc_scale_with_compression)

        patch_dim = self.patch_len * self.feat_dim

        # networkA (motion) / networkB (tokens)
        # Note: enc_layers/dec_layers are kept for CLI compatibility:
        #   - networkA layers = enc_layers
        #   - networkB layers = dec_layers
        self.netA = ARFlexNetwork(
            kind="cont",
            patch_dim=patch_dim,
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=int(enc_layers),
            d_ff=d_ff,
            dropout=dropout,
            max_len=8192,
            self_attn_drop_path=self_attn_drop_path,
            to_logits_vocab_size=self.vocab_size,
        )
        self.netB = ARFlexNetwork(
            kind="token",
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=int(dec_layers),
            d_ff=d_ff,
            dropout=dropout,
            max_len=8192,
            self_attn_drop_path=self_attn_drop_path,
            token_out_patch_dim=patch_dim,
        )
        # Conditioning vector from compression ratio (and its inverse) shared across A<->B.
        self.compression_cond_proj = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # alignment helper (always available as metric; loss enabled by cfg)

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

    @staticmethod
    def _resample_indices(src_len: int, tgt_len: int, device: torch.device) -> torch.Tensor:
        if tgt_len <= 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        if src_len <= 1:
            return torch.zeros((tgt_len,), dtype=torch.long, device=device)
        idx = torch.floor(torch.arange(tgt_len, device=device, dtype=torch.float32) * (float(src_len) / float(tgt_len))).to(torch.long)
        return idx.clamp(min=0, max=src_len - 1)

    def _downsample_token_steps(
        self,
        token_ids_full: torch.Tensor,      # [B,Ts]
        token_logits_full: torch.Tensor,   # [B,Ts,V]
        target_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _Bsz, Ts = token_ids_full.shape
        Tt = int(target_len)
        if Tt <= 0:
            raise ValueError(f"target_len must be > 0, got {Tt}")
        idx = self._resample_indices(Ts, Tt, token_ids_full.device)
        ids = token_ids_full.index_select(dim=1, index=idx)
        logits = token_logits_full.index_select(dim=1, index=idx)
        return ids, logits

    def _interpolate_token_embs(
        self,
        token_embs: torch.Tensor,         # [B,Ts,D]
        tgt_len: int,
    ) -> torch.Tensor:
        if token_embs.size(1) == int(tgt_len):
            return token_embs
        x = token_embs.transpose(1, 2)  # [B,D,Ts]
        y = F.interpolate(x, size=int(tgt_len), mode="linear", align_corners=False)
        return y.transpose(1, 2)

    def _pad_generated_latent_to_len(
        self,
        token_ids: torch.Tensor,
        token_logits: torch.Tensor,
        token_embs: torch.Tensor,
        attn_layers: List[Optional[torch.Tensor]],
        hidden_layers: List[Optional[torch.Tensor]],
        tgt_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Pad/slice generated latent outputs to a fixed target length."""
        cur_len = int(token_ids.size(1))
        tgt_len = int(tgt_len)
        if cur_len == tgt_len:
            return token_ids, token_logits, token_embs, attn_layers, hidden_layers

        Bsz = token_ids.size(0)
        V = token_logits.size(-1)
        D = token_embs.size(-1)
        device = token_ids.device

        if cur_len > tgt_len:
            token_ids = token_ids[:, :tgt_len]
            token_logits = token_logits[:, :tgt_len, :]
            token_embs = token_embs[:, :tgt_len, :]
            attn_layers_out: List[Optional[torch.Tensor]] = []
            for a in attn_layers:
                if a is None:
                    attn_layers_out.append(None)
                else:
                    attn_layers_out.append(a[:, :, :tgt_len, :])
            hidden_layers_out: List[Optional[torch.Tensor]] = []
            for h in hidden_layers:
                if h is None:
                    hidden_layers_out.append(None)
                else:
                    hidden_layers_out.append(h[:, :tgt_len, :])
            return token_ids, token_logits, token_embs, attn_layers_out, hidden_layers_out

        pad = tgt_len - cur_len
        token_ids_pad = torch.zeros((Bsz, pad), device=device, dtype=token_ids.dtype)
        token_logits_pad = torch.zeros((Bsz, pad, V), device=token_logits.device, dtype=token_logits.dtype)
        token_embs_pad = torch.zeros((Bsz, pad, D), device=token_embs.device, dtype=token_embs.dtype)
        token_ids = torch.cat([token_ids, token_ids_pad], dim=1)
        token_logits = torch.cat([token_logits, token_logits_pad], dim=1)
        token_embs = torch.cat([token_embs, token_embs_pad], dim=1)

        attn_layers_out = []
        for a in attn_layers:
            if a is None:
                attn_layers_out.append(None)
            else:
                pad_a = torch.zeros(
                    (a.size(0), a.size(1), pad, a.size(3)),
                    device=a.device,
                    dtype=a.dtype,
                )
                attn_layers_out.append(torch.cat([a, pad_a], dim=2))

        hidden_layers_out = []
        for h in hidden_layers:
            if h is None:
                hidden_layers_out.append(None)
            else:
                pad_h = torch.zeros((h.size(0), pad, h.size(2)), device=h.device, dtype=h.dtype)
                hidden_layers_out.append(torch.cat([h, pad_h], dim=1))

        return token_ids, token_logits, token_embs, attn_layers_out, hidden_layers_out

    def _resolve_ratio_per_sample(
        self,
        patch_mask: torch.Tensor,
        compression_ratio_override: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        Bsz = patch_mask.shape[0]
        if compression_ratio_override is None:
            ratio = torch.full((Bsz,), float(self.compression_ratio), device=patch_mask.device, dtype=torch.float32)
        elif isinstance(compression_ratio_override, torch.Tensor):
            ratio = compression_ratio_override.to(device=patch_mask.device, dtype=torch.float32)
            if ratio.ndim == 0:
                ratio = ratio.view(1).expand(Bsz)
            elif ratio.ndim == 1 and ratio.numel() == 1:
                ratio = ratio.expand(Bsz)
            elif ratio.ndim == 1 and ratio.numel() == Bsz:
                pass
            else:
                raise ValueError(
                    f"compression_ratio_override tensor shape must be scalar or [B]. "
                    f"Got {tuple(ratio.shape)} for batch={Bsz}"
                )
        else:
            ratio = torch.full((Bsz,), float(compression_ratio_override), device=patch_mask.device, dtype=torch.float32)
        return ratio.clamp_min(1e-6)

    def compute_latent_mask(
        self,
        patch_mask: torch.Tensor,
        compression_ratio_override: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        """patch_mask: [B,N] bool -> latent_mask: [B,Tb] bool"""
        Bsz, N = patch_mask.shape
        Tb = int(self.latent_len_max)
        ratio = self._resolve_ratio_per_sample(patch_mask, compression_ratio_override=compression_ratio_override)
        valid_patches = patch_mask.sum(dim=1)  # [B]
        valid_latent = torch.ceil(valid_patches.to(torch.float32) / ratio).to(torch.long)
        valid_latent = valid_latent.clamp(min=1, max=Tb)
        t = torch.arange(Tb, device=patch_mask.device).unsqueeze(0).expand(Bsz, Tb)
        latent_mask = t < valid_latent.unsqueeze(1)
        return latent_mask

    def _recent_kv_patch_steps(self) -> int:
        n = int(self.past_kv_recent_frames)
        if n <= 0:
            return 0
        p = max(int(self.patch_len), 1)
        return max(1, int(math.ceil(float(n) / float(p))))

    def forward(
        self,
        x: torch.Tensor,          # [B,T,D] normalized
        mask: torch.Tensor,       # [B,T] bool
        *,
        task: str = "ae",         # "ae" | "a_lm"
        a2b_mode: str = "ar",     # "ar" | "logits" | "tf_logits" | "tf_teacher"
        b2a_mode: str = "ar",     # "ar" | "parallel" | "tf_proj"
        teacher_token_ids: Optional[torch.Tensor] = None,     # [B,Tb], used by a2b_mode='tf_teacher'
        teacher_token_mask: Optional[torch.Tensor] = None,    # [B,Tb] bool, used by a2b_mode='tf_teacher'
        memA_next_loss_enable: bool = False,
        tau: float = 0.5,
        scale: float = 1.0,
        hard: bool = True,
        top_k: int = 0,
        deterministic_tokens: bool = False,
        a2b_teacher_forcing_prob: float = 0.0,
        latent_embed_mode: str = "sample",
        l2p_a_drop_prob: float = 0.0,
        l2p_a_drop_mode: str = "prob",
        l2p_a_drop_mse_thresh: float = 0.0,
        compression_ratio_override: Optional[torch.Tensor | float] = None,
        eval_teacher_forcing_decode: bool = False,
        return_attn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward for both tasks.

        task="ae":
          A->B->A autoencoding. A->B and B->A behaviors are controlled by a2b_mode/b2a_mode.

        task="a_lm":
          Next-patch prediction for A (teacher forcing; no B involved).
        """
        task = str(task).lower().strip()
        a2b_mode = str(a2b_mode).lower().strip()
        b2a_mode = str(b2a_mode).lower().strip()
        if a2b_mode not in {"ar", "logits", "tf_logits", "tf_teacher"}:
            raise ValueError(
                f"Unknown a2b_mode: {a2b_mode} (expected ar|logits|tf_logits|tf_teacher)"
            )
        if b2a_mode not in {"ar", "parallel", "tf_proj"}:
            raise ValueError(
                f"Unknown b2a_mode: {b2a_mode} (expected ar|parallel|tf_proj)"
            )
        x_pad, mask_pad, x_patch, patch_mask = patchify(x, mask, self.patch_len)
        Bsz, N, PD = x_patch.shape
        recent_kv_steps = self._recent_kv_patch_steps()

        if task == "a_lm":
            # A-only next-token prediction (no cross-attn memory)
            x_patch_hat, _attn, _hid, _fr_rate = self.netA.teacher_forcing_decode(
                x_patch,
                patch_mask,
                memory=None,
                memory_key_padding_mask=None,
                return_attn=False,
                return_hidden=False,
                recent_kv_frames=recent_kv_steps,
            )
            x_hat = unpatchify(x_patch_hat, self.patch_len, self.feat_dim)  # [B,T_pad,D]
            return {
                "x_hat": x_hat,
                "mask_pad": mask_pad,
                "patch_mask": patch_mask,
                "l2p_free_run_accept_rate": torch.zeros((), device=x.device),
            }

        # ------------------
        # Autoencoder (A->B->A)
        # ------------------
        # latent masks
        Tb = int(self.latent_len_max)
        ratio_per_sample = self._resolve_ratio_per_sample(
            patch_mask,
            compression_ratio_override=compression_ratio_override,
        )  # [B]
        ratio_mean = float(ratio_per_sample.detach().mean().item())
        latent_mask = self.compute_latent_mask(patch_mask, compression_ratio_override=ratio_per_sample)  # [B,Tb] bool
        latent_steps_eff = int(latent_mask.sum(dim=1).max().item())
        latent_steps_eff = max(1, min(latent_steps_eff, Tb))
        latent_mask_eff = latent_mask[:, :latent_steps_eff]

        token_pos_step: float | torch.Tensor = 1.0
        cond_ratio = ratio_per_sample
        if self.token_posenc_scale_with_compression:
            # In PE-scaled mode, keep cond_emb equivalent to compression_ratio=1.
            cond_ratio = torch.ones_like(ratio_per_sample)
            token_pos_step = ratio_per_sample

        # encode A with networkA (causal, prefix-style)
        memA = self.netA.encode(x_patch, patch_mask, causal=True)  # [B,N,D]
        cond_feat = torch.stack([cond_ratio, cond_ratio.reciprocal()], dim=-1).to(memA.dtype)  # [B,2]
        cond_emb = self.compression_cond_proj(cond_feat)  # [B,D]
        memA_cond = memA + cond_emb.unsqueeze(1)
        memA_key_padding_mask = ~patch_mask  # True=pad
        memA_keep_mask = patch_mask          # True=valid
        a2b_supervised_kl_loss = torch.zeros((), device=x.device)

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

        # generate B conditioned on hiddenA
        if a2b_mode == "tf_teacher":
            if teacher_token_ids is None:
                raise ValueError("teacher_token_ids is required for a2b_mode='tf_teacher'")
            if self.netB.token_embed is None:
                raise RuntimeError("token_embed missing for kind='token'")
            # Keep fixed-length teacher interface for compatibility.
            latent_mask_tf = latent_mask
            token_ids = teacher_token_ids.to(torch.long)
            if token_ids.ndim != 2 or token_ids.size(0) != Bsz:
                raise ValueError(
                    f"teacher_token_ids shape mismatch: got {tuple(token_ids.shape)} "
                    f"expected [B,T] with B={Bsz}"
                )
            if token_ids.size(1) != latent_mask_tf.size(1):
                idx = self._resample_indices(token_ids.size(1), latent_mask_tf.size(1), token_ids.device)
                token_ids = token_ids.index_select(dim=1, index=idx)
            if teacher_token_mask is not None:
                tmask = teacher_token_mask.to(torch.bool)
                if tmask.ndim != 2 or tmask.size(0) != Bsz:
                    raise ValueError(
                        f"teacher_token_mask shape mismatch: got {tuple(tmask.shape)} "
                        f"expected [B,T] with B={Bsz}"
                    )
                if tmask.size(1) != latent_mask_tf.size(1):
                    idx = self._resample_indices(tmask.size(1), latent_mask_tf.size(1), tmask.device)
                    tmask = tmask.index_select(dim=1, index=idx)
                latent_mask_tf = latent_mask_tf & tmask
            token_logits, token_embs, attn_layers_p2l, hidden_layers_p2l = self.netB.teacher_forcing_token_decode(
                token_ids,
                latent_mask_tf,
                memory=memA_cond,
                memory_key_padding_mask=memA_key_padding_mask,
                return_attn=need_attn_tok,
                return_hidden=need_softptr,
                attn_apply=attn_apply,
                input_drop_prob=a2b_teacher_forcing_prob,
                tau=tau,
                scale=scale,
                hard=hard,
                top_k=top_k,
                deterministic=deterministic_tokens,
                embed_mode=latent_embed_mode,
                pos_step=token_pos_step,
            )
            if self.netA.to_logits is not None:
                logits_a_full = self.netA.to_logits(memA_cond).detach()  # [B,N,V]
                if logits_a_full.size(1) == token_logits.size(1):
                    logits_a_ref = logits_a_full
                else:
                    idx = self._resample_indices(logits_a_full.size(1), token_logits.size(1), logits_a_full.device)
                    logits_a_ref = logits_a_full.index_select(dim=1, index=idx)  # [B,Tb,V]
                valid_bt = latent_mask_tf
                if bool(valid_bt.any()):
                    log_q = F.log_softmax(token_logits[valid_bt], dim=-1)
                    p_ref = F.softmax(logits_a_ref[valid_bt], dim=-1)
                    a2b_supervised_kl_loss = F.kl_div(log_q, p_ref, reduction="batchmean")
            latent_mask_eff = latent_mask_tf
            a2b_tf_use_rate = torch.ones((), device=x.device)
        elif a2b_mode == "tf_logits":
            if self.netA.to_logits is None:
                raise RuntimeError("netA.to_logits is required for a2b_mode='tf_logits'")
            if self.netB.token_embed is None:
                raise RuntimeError("token_embed missing for kind='token'")
            token_logits_full = self.netA.to_logits(memA_cond) * float(scale)  # [B,N,V]
            token_logits_full = self.netB._topk_mask(token_logits_full, int(top_k))
            if str(latent_embed_mode).lower().strip() == "softmax":
                probs_full = F.softmax(token_logits_full / max(float(tau), 1e-6), dim=-1)
                token_ids_full = probs_full.argmax(dim=-1)
            else:
                if bool(deterministic_tokens):
                    token_ids_full = token_logits_full.argmax(dim=-1)
                else:
                    onehot_full = F.gumbel_softmax(token_logits_full, tau=max(float(tau), 1e-6), hard=bool(hard), dim=-1)
                    token_ids_full = onehot_full.argmax(dim=-1)
            token_ids, _token_logits_ds = self._downsample_token_steps(
                token_ids_full,
                token_logits_full,
                target_len=latent_steps_eff,
            )
            token_ids, token_logits, token_embs, attn_layers_p2l, hidden_layers_p2l = self.netB.teacher_forcing_token_decode(
                token_ids,
                latent_mask_eff,
                memory=memA_cond,
                memory_key_padding_mask=memA_key_padding_mask,
                return_attn=need_attn_tok,
                return_hidden=need_softptr,
                attn_apply=attn_apply,
                input_drop_prob=a2b_teacher_forcing_prob,
                tau=tau,
                scale=scale,
                hard=hard,
                top_k=top_k,
                deterministic=deterministic_tokens,
                embed_mode=latent_embed_mode,
                pos_step=token_pos_step,
            )
            a2b_tf_use_rate = torch.ones((), device=x.device)
        elif a2b_mode == "ar":
            latent_steps = int(latent_mask_eff.sum(dim=1).max().item())
            latent_steps = max(1, min(latent_steps, Tb))
            token_ids, token_logits, token_embs, attn_layers_p2l, hidden_layers_p2l = self.netB.generate(
                memA_cond,
                memA_key_padding_mask,
                latent_steps,
                tau=tau,
                scale=scale,
                hard=hard,
                top_k=top_k,
                return_attn=need_attn_tok,
                return_hidden=need_softptr,
                deterministic=deterministic_tokens,
                embed_mode=latent_embed_mode,
                attn_apply=attn_apply,
                recent_kv_frames=recent_kv_steps,
                compression_rate=ratio_mean,
                pos_step=token_pos_step,
            )
            latent_mask_eff = latent_mask[:, :latent_steps]
            a2b_tf_use_rate = torch.zeros((), device=x.device)
        else:
            if self.netA.to_logits is None:
                raise RuntimeError("netA.to_logits is required for a2b_mode='logits'")
            if self.netB.token_embed is None:
                raise RuntimeError("token_embed missing for kind='token'")
            token_logits_full = self.netA.to_logits(memA_cond) * float(scale)  # [B,N,V]
            token_logits_full = self.netB._topk_mask(token_logits_full, int(top_k))
            if str(latent_embed_mode).lower().strip() == "softmax":
                probs_full = F.softmax(token_logits_full / max(float(tau), 1e-6), dim=-1)
                token_ids_full = probs_full.argmax(dim=-1)
            else:
                if bool(deterministic_tokens):
                    token_ids_full = token_logits_full.argmax(dim=-1)
                else:
                    onehot_full = F.gumbel_softmax(token_logits_full, tau=max(float(tau), 1e-6), hard=bool(hard), dim=-1)
                    token_ids_full = onehot_full.argmax(dim=-1)
            token_ids, token_logits = self._downsample_token_steps(
                token_ids_full,
                token_logits_full,
                target_len=latent_steps_eff,
            )
            if str(latent_embed_mode).lower().strip() == "softmax":
                probs = F.softmax(token_logits / max(float(tau), 1e-6), dim=-1)
                token_embs = probs @ self.netB.token_embed.weight
            else:
                token_embs = self.netB.token_embed(token_ids)
            attn_layers_p2l = [None for _ in range(len(self.netB.layers))]
            hidden_layers_p2l = [None for _ in range(len(self.netB.layers))]
            a2b_tf_use_rate = torch.zeros((), device=x.device)

        # encode B with networkB (causal) to build memory for A decoding
        token_embs_cond = token_embs + cond_emb.unsqueeze(1)
        memB_tok = self.netB.encode(token_embs_cond, latent_mask_eff, causal=True, pos_step=token_pos_step)  # [B,Tm,D]
        if self.netB.to_logits is None:
            raise RuntimeError("to_logits missing for kind='token'")
        token_logits_b2a = self.netB.to_logits(memB_tok)  # [B,Tm,V], for B->A next-token CE

        memB = memB_tok
        memB_mask = latent_mask_eff
        if a2b_mode == "logits" and ratio_mean > 1.0:
            # Interpolate B tokens back to patch resolution before B->A.
            token_embs_up = self._interpolate_token_embs(token_embs_cond, N)
            memB = self.netB.encode(token_embs_up, patch_mask, causal=True)  # [B,N,D]
            memB_mask = patch_mask
        memB_key_padding_mask = ~memB_mask  # True=pad

        # A decoding conditioned on hiddenB
        sp_cfg = self.softptr_cfg or {}
        softptr_l2p_enable = bool(sp_cfg.get("l2p_enable", True))
        need_softptr_l2p = bool(need_softptr and softptr_l2p_enable and b2a_mode in {"ar", "tf_proj"})
        need_attn_l2p = bool((return_attn and b2a_mode in {"ar", "tf_proj"}) or need_softptr_l2p)

        if b2a_mode == "parallel":
            if self.netB.out_proj is None:
                raise RuntimeError("netB.out_proj is required for b2a_mode='parallel'")
            z_patch = self.netB.out_proj(memB)  # [B,Tb,PD]
            if z_patch.size(1) == N:
                x_patch_hat = z_patch
            else:
                idx = self._resample_indices(z_patch.size(1), N, z_patch.device)
                x_patch_hat = z_patch.index_select(dim=1, index=idx)  # [B,N,PD]
            attn_layers_l2p = [None for _ in range(len(self.netA.layers))]
            hidden_layers_l2p = [None for _ in range(len(self.netA.layers))]
            fr_accept_rate = torch.zeros((), device=x.device)
        elif b2a_mode == "tf_proj":
            if self.netB.out_proj is None:
                raise RuntimeError("netB.out_proj is required for b2a_mode='tf_proj'")
            x_patch_tf_in = self.netB.out_proj(memB)  # [B,Tm,PD]
            if x_patch_tf_in.size(1) != N:
                idx = self._resample_indices(x_patch_tf_in.size(1), N, x_patch_tf_in.device)
                x_patch_tf_in = x_patch_tf_in.index_select(dim=1, index=idx)
            x_patch_hat, attn_layers_l2p, hidden_layers_l2p, fr_accept_rate = self.netA.teacher_forcing_decode(
                x_patch_tf_in,
                patch_mask,
                memory=memB,
                memory_key_padding_mask=memB_key_padding_mask,
                return_attn=need_attn_l2p,
                return_hidden=need_softptr_l2p,
                attn_apply=attn_apply,
                input_drop_prob=l2p_a_drop_prob,
                input_drop_mode=l2p_a_drop_mode,
                input_drop_mse_thresh=l2p_a_drop_mse_thresh,
                recent_kv_frames=recent_kv_steps,
            )
        else:
            # autoregressive generation A <- B (no teacher forcing)
            x_patch_hat, attn_layers_l2p, hidden_layers_l2p = self.netA.generate(
                memB,
                memB_key_padding_mask,
                N,
                return_attn=need_attn_l2p,
                return_hidden=need_softptr_l2p,
                attn_apply=attn_apply,
                recent_kv_frames=recent_kv_steps,
            )
            fr_accept_rate = torch.zeros((), device=x.device)

        x_hat = unpatchify(x_patch_hat, self.patch_len, self.feat_dim)  # [B,T_pad,D]

        if bool(memA_next_loss_enable):
            if self.netA.out_proj is None:
                raise RuntimeError("netA.out_proj missing for kind='cont'")
            if N >= 2:
                a_next_pred = self.netA.out_proj(memA[:, :-1, :])  # [B,N-1,PD]
                a_next_tgt = x_patch[:, 1:, :]
                a_next_mask = patch_mask[:, :-1] & patch_mask[:, 1:]
                a_next_loss = masked_mse(a_next_pred, a_next_tgt, a_next_mask)
            else:
                a_next_loss = torch.zeros((), device=x.device)
        else:
            a_next_loss = torch.zeros((), device=x.device)

        out: Dict[str, torch.Tensor] = {
            "x_hat": x_hat,
            "mask_pad": mask_pad,
            "token_ids": token_ids,
            "token_logits": token_logits,
            "token_logits_b2a": token_logits_b2a,
            "latent_mask": latent_mask_eff,
            "patch_mask": patch_mask,
            "a2b_teacher_forcing_use_rate": a2b_tf_use_rate,
            "l2p_free_run_accept_rate": fr_accept_rate,
            "a_mem_next_loss": a_next_loss,
            "a2b_supervised_kl_loss": a2b_supervised_kl_loss,
            "compression_ratio_used_mean": ratio_per_sample.mean(),
            "compression_ratio_used_min": ratio_per_sample.min(),
            "compression_ratio_used_max": ratio_per_sample.max(),
        }

        # gaussian alignment + band mass
        if return_attn:
            sigma_scale_b2a = 1.0 if memB_mask.shape[1] == patch_mask.shape[1] else (1.0 / max(ratio_mean, 1e-6))
            if self.gauss_align is None:
                loss_p2l = torch.zeros((), device=x.device)
                loss_l2p = torch.zeros((), device=x.device)
            else:
                if a2b_mode not in {"ar", "tf_logits", "tf_teacher"}:
                    loss_p2l = torch.zeros((), device=x.device)
                else:
                    loss_p2l = self.gauss_align(
                        attn_layers_p2l,
                        src_keep_mask=memA_keep_mask,
                        tgt_keep_mask=latent_mask_eff,
                        sigma_scale=1.0,
                    )
                if b2a_mode == "parallel":
                    loss_l2p = torch.zeros((), device=x.device)
                else:
                    loss_l2p = self.gauss_align(
                        attn_layers_l2p,
                        src_keep_mask=memB_mask,
                        tgt_keep_mask=patch_mask,
                        sigma_scale=sigma_scale_b2a,
                    )

            out["gauss_align_loss_p2l"] = loss_p2l
            out["gauss_align_loss_l2p"] = loss_l2p
            out["gauss_align_loss"] = 0.5 * (loss_p2l + loss_l2p)
            if a2b_mode in {"ar", "tf_logits", "tf_teacher"}:
                out["band_mass_p2l"] = self.gauss_align_metric.compute_band_mass(
                    attn_layers_p2l,
                    src_keep_mask=memA_keep_mask,
                    tgt_keep_mask=latent_mask_eff,
                    sigma_scale=1.0,
                )
            else:
                out["band_mass_p2l"] = torch.zeros((), device=x.device)
            if b2a_mode == "parallel":
                out["band_mass_l2p"] = torch.zeros((), device=x.device)
            else:
                out["band_mass_l2p"] = self.gauss_align_metric.compute_band_mass(
                    attn_layers_l2p,
                    src_keep_mask=memB_mask,
                    tgt_keep_mask=patch_mask,
                    sigma_scale=sigma_scale_b2a,
                )
        else:
            z = torch.zeros((), device=x.device)
            out["gauss_align_loss_p2l"] = z
            out["gauss_align_loss_l2p"] = z
            out["gauss_align_loss"] = z
            out["band_mass_p2l"] = z
            out["band_mass_l2p"] = z

        # soft pointer loss (optional)
        can_softptr_p2l = bool(a2b_mode in {"ar", "tf_logits", "tf_teacher"})
        can_softptr_l2p = bool(b2a_mode in {"ar", "tf_proj"})
        if self.softptr is not None and need_softptr and (can_softptr_p2l or can_softptr_l2p):
            sp_apply = str(sp_cfg.get("apply", "all")).lower().strip()
            if sp_apply == "all":
                L1 = int(sp_cfg.get("L1", 0))
                L2 = int(sp_cfg.get("L2", -1))
                if L2 <= 0:
                    L2 = len(attn_layers_p2l)
                L2 = min(L2, len(attn_layers_p2l))
                L1 = max(0, min(L1, max(L2 - 1, 0)))
            else:
                L1, L2 = _resolve_layer_span(len(attn_layers_p2l), sp_apply)

            head_topk = int(sp_cfg.get("head_topk", 0))
            head_topk_opt = None if head_topk <= 0 else head_topk

            tau_sp = float(sp_cfg.get("tau", 3.0))
            detach_w = bool(sp_cfg.get("detach_w", False))
            lambda_cos = float(sp_cfg.get("lambda_cos", 0.0))

            if can_softptr_p2l:
                loss_sp_p2l, _logs = self.softptr(
                    attn_layers_p2l,
                    memA,
                    hidden_layers_p2l,
                    src_keep_mask=memA_keep_mask,
                    tgt_keep_mask=latent_mask_eff,
                    L1=L1,
                    L2=L2,
                    tau=tau_sp,
                    head_topk=head_topk_opt,
                    detach_w=detach_w,
                    lambda_cos=lambda_cos,
                    direction="p2l",
                )
            else:
                loss_sp_p2l = torch.zeros((), device=x.device)
            if need_softptr_l2p and can_softptr_l2p:
                loss_sp_l2p, _logs2 = self.softptr(
                    attn_layers_l2p,
                    memB,
                    hidden_layers_l2p,
                    src_keep_mask=memB_mask,
                    tgt_keep_mask=patch_mask,
                    L1=L1,
                    L2=L2,
                    tau=tau_sp,
                    head_topk=head_topk_opt,
                    detach_w=detach_w,
                    lambda_cos=lambda_cos,
                    direction="l2p",
                )
            else:
                loss_sp_l2p = torch.zeros((), device=x.device)

            out["softptr_loss_p2l"] = loss_sp_p2l
            out["softptr_loss_l2p"] = loss_sp_l2p
            out["softptr_loss"] = 0.5 * (loss_sp_p2l + loss_sp_l2p)
        else:
            z = torch.zeros((), device=x.device)
            out["softptr_loss_p2l"] = z
            out["softptr_loss_l2p"] = z
            out["softptr_loss"] = z

        return out

@dataclass
class TrainConfig:
    data_root: str
    save_dir: str
    max_motion_len: int = 196
    patch_len: int = 1
    compression_ratio: float = 4.0
    compression_ratio_min: float = 0.0  # <=0 disables variable compression sampling
    compression_ratio_max: float = 0.0  # <=0 disables variable compression sampling
    token_posenc_scale_with_compression: bool = False  # if True, token PE spacing uses compression_ratio and cond_emb is fixed at ratio=1
    latent_len: int = 0  # 0 => derive from (T_pad/patch_len)/compression_ratio

    vocab_size: int = 1024
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    self_attn_drop_path: float = 0.0
    enc_layers: int = 6
    dec_layers: int = 6
    recon_layers: int = 6

    # training stage
    train_stage: str = "ae"  # a_lm | ae
    a2b_mode: str = "ar"     # ar | logits | tf_logits | tf_teacher
    b2a_mode: str = "ar"     # ar | parallel | tf_proj
    freeze_netA: bool = False
    freeze_netB: bool = False
    # when freeze_netA=True, optionally keep netA cross-attn trainable
    freeze_a_cross_attn_trainable: bool = False


    # AR sampling
    gumbel_tau: float = 0.5
    gumbel_scale: float = 1.0
    gumbel_hard: bool = True
    top_k: int = 0
    a2b_teacher_forcing_prob: float = 0.0  # A->B AR only: probability of feeding netA argmax token as teacher
    latent_embed_mode: str = "sample"  # sample|softmax (softmax = no discretization for B embeddings)
    l2p_a_drop_prob: float = 0.0  # input dropout prob for B->A (A decoding)
    l2p_a_drop_mode: str = "prob"  # prob|mse_thresh
    l2p_a_drop_mse_thresh: float = 0.0  # used when l2p_a_drop_mode=mse_thresh
    past_kv_recent_frames: int = 0  # 0 => full prefix KV cache (disabled)

    # losses
    rec_w: float = 1.0
    token_ce_w: float = 0.0   # CE(logits, sampled_ids) as a "self-sharpening" term
    a2b_supervised_kl_w: float = 1.0  # a2b_mode=tf_teacher: KL(netB TF logits || stopgrad(netA.to_logits))
    a_mem_next_w: float = 0.0  # optional A-next prediction loss from initial memA
    norm_gap_enable: bool = False
    norm_gap_w: float = 0.0

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
    parallel_teacher_ckpt: str = ""  # for a2b_mode=tf_teacher (fallback: pretrain_ckpt)


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

    # Backward compatibility: token network query_token -> start_token.
    # (Older versions used a learned query token as the AR "slot"; the new version uses BOS.)
    old_k = "netB.query_token"
    new_k = "netB.start_token"
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
        ratio_for_tb = float(cfg.compression_ratio)
        if cfg.compression_ratio_min > 0.0:
            ratio_for_tb = min(ratio_for_tb, float(cfg.compression_ratio_min))
        if cfg.compression_ratio_max > 0.0:
            ratio_for_tb = min(ratio_for_tb, float(cfg.compression_ratio_max))
        Tb = int(math.ceil(N / max(ratio_for_tb, 1e-6)))

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
        self_attn_drop_path=cfg.self_attn_drop_path,
        max_motion_len=cfg.max_motion_len,
        gauss_cfg=gauss_cfg,
        softptr_cfg=softptr_cfg,
        past_kv_recent_frames=cfg.past_kv_recent_frames,
        token_posenc_scale_with_compression=cfg.token_posenc_scale_with_compression,
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

    # ------------------
    # Train stage control
    # ------------------
    stage = str(getattr(cfg, "train_stage", "ae")).lower().strip()
    if stage not in {"a_lm", "ae"}:
        raise ValueError(f"Unknown train_stage: {stage} (expected a_lm|ae)")

    for p in model.parameters():
        p.requires_grad = True

    if stage == "a_lm":
        # train only networkA (motion next-token)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.netA.parameters():
            p.requires_grad = True

    if bool(getattr(cfg, "freeze_netA", False)):
        for p in model.netA.parameters():
            p.requires_grad = False
        if bool(getattr(cfg, "freeze_a_cross_attn_trainable", False)):
            # optional: allow netA cross-attn (and its norm2) to adapt
            for n, p in model.netA.named_parameters():
                if ("cross_attn" in n) or ("norm2" in n):
                    p.requires_grad = True

    if bool(getattr(cfg, "freeze_netB", False)):
        for p in model.netB.parameters():
            p.requires_grad = False
        if hasattr(model, "softptr") and model.softptr is not None:
            for p in model.softptr.parameters():
                p.requires_grad = False

    # save normalization + cfg
    np.save(save_dir / "mean.npy", mean)
    np.save(save_dir / "std.npy", std)
    save_json(save_dir / f"train_config_{run_id}.json", asdict(cfg))

    # datasets
    ds_train = HumanML3DMotionDataset(Path(cfg.data_root), "train", cfg.max_motion_len, mean=mean, std=std, random_crop=True)
    ds_val = HumanML3DMotionDataset(Path(cfg.data_root), "val", cfg.max_motion_len, mean=mean, std=std, random_crop=False)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters. Check train_stage / freeze settings.")
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    proposal = None
    if stage != "a_lm" and cfg.proposal_enable:
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

    task = "a_lm" if stage == "a_lm" else "ae"
    a2b_mode_eff = str(getattr(cfg, "a2b_mode", "ar")).lower().strip()
    b2a_mode_eff = str(getattr(cfg, "b2a_mode", "ar")).lower().strip()
    ratio_min_train = float(cfg.compression_ratio_min) if float(cfg.compression_ratio_min) > 0.0 else float(cfg.compression_ratio)
    ratio_max_train = float(cfg.compression_ratio_max) if float(cfg.compression_ratio_max) > 0.0 else float(cfg.compression_ratio)
    if ratio_min_train > ratio_max_train:
        raise ValueError(
            f"Invalid compression ratio range: min={ratio_min_train} > max={ratio_max_train}. "
            "Set --compression_ratio_min <= --compression_ratio_max."
        )
    use_variable_compression_train = bool(
        (ratio_max_train - ratio_min_train) > 1e-8
        and stage != "a_lm"
    )
    use_compression_override_train = bool(stage != "a_lm")
    print(
        f"[train] compression ratio mode: "
        f"{'variable' if use_variable_compression_train else 'fixed'} "
        f"(min={ratio_min_train:.6g}, max={ratio_max_train:.6g}, base={float(cfg.compression_ratio):.6g})",
        flush=True,
    )

    def sample_compression_ratio(batch_size: int) -> Optional[torch.Tensor]:
        if not use_compression_override_train:
            return None
        if use_variable_compression_train:
            # One shared ratio value per batch.
            r = torch.empty((), device=device, dtype=torch.float32)
            r.uniform_(ratio_min_train, ratio_max_train)
            return r
        # Fixed-ratio training: still pass explicit override so min=max takes effect.
        return torch.tensor(ratio_min_train, device=device, dtype=torch.float32)

    need_attn_train = bool(stage == "ae")
    eval_tok_path = log_dir / "eval_tokens_B_val_last.jsonl"

    parallel_teacher_model: Optional[MotionSeq2SeqARAE] = None
    if a2b_mode_eff == "tf_teacher":
        teacher_ckpt = str(getattr(cfg, "parallel_teacher_ckpt", "")).strip()
        if teacher_ckpt == "":
            teacher_ckpt = str(getattr(cfg, "pretrain_ckpt", "")).strip()
        if teacher_ckpt == "":
            raise ValueError("a2b_mode=tf_teacher requires --parallel_teacher_ckpt (or --pretrain_ckpt)")
        parallel_teacher_model = copy.deepcopy(model).to(device)
        _load_pretrained_model(parallel_teacher_model, Path(teacher_ckpt), strict=bool(cfg.pretrain_strict))
        parallel_teacher_model.eval()
        for p in parallel_teacher_model.parameters():
            p.requires_grad = False

    def run_eval() -> Dict[str, float]:
        model.eval()

        n_batches = 0
        sum_loss = 0.0
        sum_rec = 0.0
        sum_ce = 0.0
        sum_align = 0.0
        sum_align_p2l = 0.0
        sum_align_l2p = 0.0
        sum_softptr = 0.0
        sum_softptr_p2l = 0.0
        sum_softptr_l2p = 0.0
        sum_band_mass_p2l = 0.0
        sum_band_mass_l2p = 0.0
        sum_Lpos = 0.0
        sum_Lmarg = 0.0
        sum_Linfo = 0.0
        sum_xhat_norm = 0.0
        sum_xpad_norm = 0.0
        sum_norm_gap = 0.0
        sum_loss_norm_gap = 0.0
        sum_a2b_sup_kl = 0.0
        sum_a_mem_next = 0.0
        sum_fr_accept_rate = 0.0
        sum_ratio_mean = 0.0
        sum_ratio_min = 0.0
        sum_ratio_max = 0.0
        tok_stats = TokenStatsAggregator(cfg.vocab_size)
        cur_tokens: Dict[str, List[int]] = {}

        with torch.no_grad():
            for batch in dl_val:
                x = batch["motion"].to(device)
                m = batch["mask"].to(device)
                compression_ratio_override = sample_compression_ratio(x.shape[0])
                teacher_token_ids = None
                teacher_token_mask = None
                if a2b_mode_eff == "tf_teacher":
                    if parallel_teacher_model is None:
                        raise RuntimeError("parallel_teacher_model is not initialized")
                    t_out = parallel_teacher_model(
                        x, m,
                        task="ae",
                        tau=cfg.gumbel_tau,
                        scale=cfg.gumbel_scale,
                        hard=True,
                        top_k=cfg.top_k,
                        deterministic_tokens=True,
                        latent_embed_mode=cfg.latent_embed_mode,
                        a2b_mode="logits",
                        b2a_mode="parallel",
                        compression_ratio_override=compression_ratio_override,
                        return_attn=False,
                    )
                    teacher_token_ids = t_out["token_ids"]
                    teacher_token_mask = t_out["latent_mask"]

                out = model(
                    x, m,
                    task=task,
                    teacher_token_ids=teacher_token_ids,
                    teacher_token_mask=teacher_token_mask,
                    tau=cfg.gumbel_tau,
                    scale=cfg.gumbel_scale,
                    hard=cfg.gumbel_hard,
                    top_k=cfg.top_k,
                    a2b_teacher_forcing_prob=cfg.a2b_teacher_forcing_prob,
                    deterministic_tokens=True,
                    latent_embed_mode=cfg.latent_embed_mode,
                    a2b_mode=a2b_mode_eff,
                    b2a_mode=b2a_mode_eff,
                    memA_next_loss_enable=bool(cfg.a_mem_next_w > 0.0),
                    l2p_a_drop_prob=cfg.l2p_a_drop_prob,
                    l2p_a_drop_mode=cfg.l2p_a_drop_mode,
                    l2p_a_drop_mse_thresh=cfg.l2p_a_drop_mse_thresh,
                    compression_ratio_override=compression_ratio_override,
                    eval_teacher_forcing_decode=False,
                    return_attn=bool(stage != "a_lm"),
                )
                fr_accept_rate = out.get("l2p_free_run_accept_rate", torch.zeros((), device=device))
                ratio_mean_t = out.get("compression_ratio_used_mean", torch.zeros((), device=device))
                ratio_min_t = out.get("compression_ratio_used_min", torch.zeros((), device=device))
                ratio_max_t = out.get("compression_ratio_used_max", torch.zeros((), device=device))

                if "token_ids" in out and "latent_mask" in out and "path" in batch:
                    token_ids = out["token_ids"].detach().cpu().numpy()
                    latent_mask = out["latent_mask"].detach().cpu().numpy()
                    paths = batch["path"]
                    for i, p in enumerate(paths):
                        ids = token_ids[i][latent_mask[i]].astype(int).tolist()
                        cur_tokens[str(p)] = ids

                x_pad = patchify(x, m, cfg.patch_len)[0]
                loss_rec = masked_mse(out["x_hat"], x_pad, out["mask_pad"])

                eps = 1e-8
                valid_bt = out["mask_pad"].to(x_pad.dtype)  # [B,T]
                den = valid_bt.sum().clamp_min(eps)
                xpad_norm_t = x_pad.detach().norm(dim=-1)         # [B,T]
                xhat_norm_t = out["x_hat"].detach().norm(dim=-1)  # [B,T]
                xpad_norm = (valid_bt * xpad_norm_t).sum() / den
                xhat_norm = (valid_bt * xhat_norm_t).sum() / den
                norm_gap = (valid_bt * (xhat_norm_t - xpad_norm_t).abs()).sum() / den
                loss_norm_gap = norm_gap if cfg.norm_gap_enable else torch.zeros((), device=device)
                loss_a_mem_next = out.get("a_mem_next_loss", torch.zeros((), device=device))
                loss_a2b_sup_kl = out.get("a2b_supervised_kl_loss", torch.zeros((), device=device))

                if stage == "a_lm":
                    # only reconstruction loss
                    loss_ce = torch.zeros((), device=device)
                    loss_align = torch.zeros((), device=device)
                    loss_align_p2l = torch.zeros((), device=device)
                    loss_align_l2p = torch.zeros((), device=device)
                    loss_softptr = torch.zeros((), device=device)
                    loss_softptr_p2l = torch.zeros((), device=device)
                    loss_softptr_l2p = torch.zeros((), device=device)
                    band_mass_p2l = torch.zeros((), device=device)
                    band_mass_l2p = torch.zeros((), device=device)
                    L_pos = torch.zeros((), device=device)
                    L_marg = torch.zeros((), device=device)
                    L_info = torch.zeros((), device=device)
                    loss_a_mem_next = torch.zeros((), device=device)
                    loss_a2b_sup_kl = torch.zeros((), device=device)
                    total = cfg.rec_w * loss_rec + cfg.norm_gap_w * loss_norm_gap
                else:
                    if cfg.token_ce_w > 0.0:
                        lm = out["latent_mask"]
                        next_lm = lm[:, :-1] & lm[:, 1:]
                        logits = out["token_logits_b2a"][:, :-1, :][next_lm]
                        ids = out["token_ids"][:, 1:][next_lm]
                        loss_ce = F.cross_entropy(logits, ids) if logits.numel() > 0 else torch.zeros((), device=device)
                    else:
                        loss_ce = torch.zeros((), device=device)

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
                    band_mass_p2l = out.get("band_mass_p2l", torch.zeros((), device=device))
                    band_mass_l2p = out.get("band_mass_l2p", torch.zeros((), device=device))

                    if proposal is not None:
                        L_pos, L_marg, L_info = proposal(out["token_logits"], valid_mask=out["latent_mask"], update_state=False)
                    else:
                        L_pos = torch.zeros((), device=device)
                        L_marg = torch.zeros((), device=device)
                        L_info = torch.zeros((), device=device)

                    total = (
                        cfg.rec_w * loss_rec
                        + cfg.token_ce_w * loss_ce
                        + cfg.a2b_supervised_kl_w * loss_a2b_sup_kl
                        + cfg.a_mem_next_w * loss_a_mem_next
                        + cfg.gauss_align_w * loss_align
                        + cfg.softptr_w * loss_softptr
                        + cfg.norm_gap_w * loss_norm_gap
                        + cfg.possharp_w * L_pos
                        + cfg.marg_w * L_marg
                        + cfg.info_w * L_info
                    )

                    # token stats
                    tok_stats.update(out["token_logits"], out["token_ids"], out["latent_mask"])

                n_batches += 1
                sum_loss += float(total.detach().cpu().item())
                sum_rec += float(loss_rec.detach().cpu().item())
                sum_ce += float(loss_ce.detach().cpu().item())
                sum_align += float(loss_align.detach().cpu().item())
                sum_align_p2l += float(loss_align_p2l.detach().cpu().item())
                sum_align_l2p += float(loss_align_l2p.detach().cpu().item())
                sum_softptr += float(loss_softptr.detach().cpu().item())
                sum_softptr_p2l += float(loss_softptr_p2l.detach().cpu().item())
                sum_softptr_l2p += float(loss_softptr_l2p.detach().cpu().item())
                sum_band_mass_p2l += float(band_mass_p2l.detach().cpu().item())
                sum_band_mass_l2p += float(band_mass_l2p.detach().cpu().item())
                sum_Lpos += float(L_pos.detach().cpu().item())
                sum_Lmarg += float(L_marg.detach().cpu().item())
                sum_Linfo += float(L_info.detach().cpu().item())
                sum_xhat_norm += float(xhat_norm.detach().cpu().item())
                sum_xpad_norm += float(xpad_norm.detach().cpu().item())
                sum_norm_gap += float(norm_gap.detach().cpu().item())
                sum_loss_norm_gap += float(loss_norm_gap.detach().cpu().item())
                sum_a2b_sup_kl += float(loss_a2b_sup_kl.detach().cpu().item())
                sum_a_mem_next += float(loss_a_mem_next.detach().cpu().item())
                sum_fr_accept_rate += float(fr_accept_rate.detach().cpu().item())
                sum_ratio_mean += float(ratio_mean_t.detach().cpu().item())
                sum_ratio_min += float(ratio_min_t.detach().cpu().item())
                sum_ratio_max += float(ratio_max_t.detach().cpu().item())

        if n_batches == 0:
            return {}

        out_m: Dict[str, float] = {
            "loss": sum_loss / n_batches,
            "loss_rec": sum_rec / n_batches,
            "loss_ce": sum_ce / n_batches,
            "loss_align": sum_align / n_batches,
            "loss_align_p2l": sum_align_p2l / n_batches,
            "loss_align_l2p": sum_align_l2p / n_batches,
            "loss_softptr": sum_softptr / n_batches,
            "loss_softptr_p2l": sum_softptr_p2l / n_batches,
            "loss_softptr_l2p": sum_softptr_l2p / n_batches,
            "band_mass_p2l": sum_band_mass_p2l / n_batches,
            "band_mass_l2p": sum_band_mass_l2p / n_batches,
            "L_pos": sum_Lpos / n_batches,
            "L_marg": sum_Lmarg / n_batches,
            "L_info": sum_Linfo / n_batches,
            "xhat_norm": sum_xhat_norm / n_batches,
            "xpad_norm": sum_xpad_norm / n_batches,
            "norm_gap": sum_norm_gap / n_batches,
            "loss_norm_gap": sum_loss_norm_gap / n_batches,
            "loss_a2b_supervised_kl": sum_a2b_sup_kl / n_batches,
            "loss_a_mem_next": sum_a_mem_next / n_batches,
            "l2p_free_run_accept_rate": sum_fr_accept_rate / n_batches,
            "compression_ratio_used_mean": sum_ratio_mean / n_batches,
            "compression_ratio_used_min": sum_ratio_min / n_batches,
            "compression_ratio_used_max": sum_ratio_max / n_batches,
        }

        if stage != "a_lm":
            tok_metrics = tok_stats.compute()
            out_m.update(tok_metrics)

        if len(cur_tokens) > 0:
            prev_tokens: Dict[str, List[int]] = {}
            if eval_tok_path.exists():
                for line in eval_tok_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip() == "":
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    path = rec.get("path")
                    tokens = rec.get("tokens")
                    if isinstance(path, str) and isinstance(tokens, list):
                        prev_tokens[path] = [int(x) for x in tokens]

            common_keys = set(cur_tokens.keys()) & set(prev_tokens.keys())
            if len(common_keys) > 0:
                sum_hamming = 0.0
                sum_edit = 0.0
                ngram_cur: Dict[Tuple[int, ...], int] = {}
                ngram_prev: Dict[Tuple[int, ...], int] = {}
                for k in common_keys:
                    a = prev_tokens[k]
                    b = cur_tokens[k]
                    sum_hamming += _hamming_match_rate(a, b)
                    sum_edit += _edit_match_rate(a, b)
                    _update_ngram_counts(a, 2, ngram_prev)
                    _update_ngram_counts(b, 2, ngram_cur)
                out_m["tokB_hamming_match"] = sum_hamming / float(len(common_keys))
                out_m["tokB_edit_match"] = sum_edit / float(len(common_keys))
                out_m["tokB_ngram2_l1"] = _ngram_l1_distance(ngram_prev, ngram_cur)
                out_m["tokB_seq_cmp_coverage"] = float(len(common_keys)) / float(max(len(cur_tokens), 1))
                out_m["tokB_seq_cmp_count"] = float(len(common_keys))

            with eval_tok_path.open("w", encoding="utf-8") as f:
                for p in sorted(cur_tokens.keys()):
                    f.write(json.dumps({"path": p, "tokens": cur_tokens[p]}) + "\n")
        return out_m

    if 1:
        val = run_eval()
        val_log = {"split": "val", "epoch": 0, "step": step, "time": 0}
        val_log.update(val)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(val_log) + "\n")
        print(val_log, flush=True)

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for batch in dl_train:
            x = batch["motion"].to(device)
            m = batch["mask"].to(device)
            compression_ratio_override = sample_compression_ratio(x.shape[0])
            teacher_token_ids = None
            teacher_token_mask = None
            if a2b_mode_eff == "tf_teacher":
                if parallel_teacher_model is None:
                    raise RuntimeError("parallel_teacher_model is not initialized")
                with torch.no_grad():
                    t_out = parallel_teacher_model(
                        x, m,
                        task="ae",
                        tau=cfg.gumbel_tau,
                        scale=cfg.gumbel_scale,
                        hard=True,
                        top_k=cfg.top_k,
                        deterministic_tokens=False,
                        latent_embed_mode=cfg.latent_embed_mode,
                        a2b_mode="logits",
                        b2a_mode="parallel",
                        compression_ratio_override=compression_ratio_override,
                        return_attn=False,
                    )
                teacher_token_ids = t_out["token_ids"]
                teacher_token_mask = t_out["latent_mask"]

            out = model(
                x, m,
                task=task,
                teacher_token_ids=teacher_token_ids,
                teacher_token_mask=teacher_token_mask,
                tau=cfg.gumbel_tau,
                scale=cfg.gumbel_scale,
                hard=cfg.gumbel_hard,
                top_k=cfg.top_k,
                a2b_teacher_forcing_prob=cfg.a2b_teacher_forcing_prob,
                deterministic_tokens=False,
                latent_embed_mode=cfg.latent_embed_mode,
                a2b_mode=a2b_mode_eff,
                b2a_mode=b2a_mode_eff,
                memA_next_loss_enable=bool(cfg.a_mem_next_w > 0.0),
                l2p_a_drop_prob=cfg.l2p_a_drop_prob,
                l2p_a_drop_mode=cfg.l2p_a_drop_mode,
                l2p_a_drop_mse_thresh=cfg.l2p_a_drop_mse_thresh,
                compression_ratio_override=compression_ratio_override,
                return_attn=need_attn_train,
            )

            x_pad = patchify(x, m, cfg.patch_len)[0]
            loss_rec = masked_mse(out["x_hat"], x_pad, out["mask_pad"])
            eps = 1e-8
            valid_bt = out["mask_pad"].to(x_pad.dtype)  # [B,T]
            den = valid_bt.sum().clamp_min(eps)
            xpad_norm_t = x_pad.norm(dim=-1)         # [B,T]
            xhat_norm_t = out["x_hat"].norm(dim=-1)  # [B,T]
            norm_gap = (valid_bt * (xhat_norm_t - xpad_norm_t).abs()).sum() / den
            loss_norm_gap = norm_gap if cfg.norm_gap_enable else torch.zeros((), device=device)
            loss_a_mem_next = out.get("a_mem_next_loss", torch.zeros((), device=device))
            loss_a2b_sup_kl = out.get("a2b_supervised_kl_loss", torch.zeros((), device=device))

            if stage == "a_lm":
                loss_ce = torch.zeros((), device=device)
                loss_align = torch.zeros((), device=device)
                loss_softptr = torch.zeros((), device=device)
                L_pos = torch.zeros((), device=device)
                L_marg = torch.zeros((), device=device)
                L_info = torch.zeros((), device=device)
                loss_a_mem_next = torch.zeros((), device=device)
                loss_a2b_sup_kl = torch.zeros((), device=device)
                fr_accept_rate = torch.zeros((), device=device)
                total = cfg.rec_w * loss_rec + cfg.norm_gap_w * loss_norm_gap
            else:
                if cfg.token_ce_w > 0.0:
                    lm = out["latent_mask"]
                    next_lm = lm[:, :-1] & lm[:, 1:]
                    logits = out["token_logits_b2a"][:, :-1, :][next_lm]
                    ids = out["token_ids"][:, 1:][next_lm]
                    loss_ce = F.cross_entropy(logits, ids) if logits.numel() > 0 else torch.zeros((), device=device)
                else:
                    loss_ce = torch.zeros((), device=device)

                loss_align = out["gauss_align_loss"] if cfg.gauss_align_enable else torch.zeros((), device=device)
                loss_softptr = out["softptr_loss"] if cfg.softptr_enable else torch.zeros((), device=device)

                if proposal is not None:
                    L_pos, L_marg, L_info = proposal(out["token_logits"], valid_mask=out["latent_mask"], update_state=True)
                else:
                    L_pos = torch.zeros((), device=device)
                    L_marg = torch.zeros((), device=device)
                    L_info = torch.zeros((), device=device)
                fr_accept_rate = out.get("l2p_free_run_accept_rate", torch.zeros((), device=device))

                total = (
                    cfg.rec_w * loss_rec
                    + cfg.token_ce_w * loss_ce
                    + cfg.a2b_supervised_kl_w * loss_a2b_sup_kl
                    + cfg.a_mem_next_w * loss_a_mem_next
                    + cfg.gauss_align_w * loss_align
                    + cfg.softptr_w * loss_softptr
                    + cfg.norm_gap_w * loss_norm_gap
                    + cfg.possharp_w * L_pos
                    + cfg.marg_w * L_marg
                    + cfg.info_w * L_info
                )

            opt.zero_grad(set_to_none=True)
            total.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            opt.step()

            if stage != "a_lm":
                train_tok_stats.update(out["token_logits"], out["token_ids"], out["latent_mask"])

            step += 1
            if step % cfg.train_log_interval == 0:
                log: Dict[str, float] = {
                    "split": "train",
                    "epoch": epoch,
                    "step": step,
                    "time": time.time() - t0,
                    "loss": float(total.detach().cpu().item()),
                    "loss_rec": float(loss_rec.detach().cpu().item()),
                    "loss_ce": float(loss_ce.detach().cpu().item()),
                    "loss_align": float(loss_align.detach().cpu().item()),
                    "loss_softptr": float(loss_softptr.detach().cpu().item()),
                    "loss_norm_gap": float(loss_norm_gap.detach().cpu().item()),
                    "loss_a2b_supervised_kl": float(loss_a2b_sup_kl.detach().cpu().item()),
                    "loss_a_mem_next": float(loss_a_mem_next.detach().cpu().item()),
                    "L_pos": float(L_pos.detach().cpu().item()),
                    "L_marg": float(L_marg.detach().cpu().item()),
                    "L_info": float(L_info.detach().cpu().item()),
                    "l2p_free_run_accept_rate": float(fr_accept_rate.detach().cpu().item()),
                    "compression_ratio_used_mean": float(out.get("compression_ratio_used_mean", torch.zeros((), device=device)).detach().cpu().item()),
                    "compression_ratio_used_min": float(out.get("compression_ratio_used_min", torch.zeros((), device=device)).detach().cpu().item()),
                    "compression_ratio_used_max": float(out.get("compression_ratio_used_max", torch.zeros((), device=device)).detach().cpu().item()),
                }
                if stage != "a_lm":
                    log.update(train_tok_stats.compute())
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log) + "\n")
                print(log, flush=True)

        # validation
        val = run_eval()
        val_log = {"split": "val", "epoch": epoch, "step": step, "time": time.time() - t0}
        val_log.update(val)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(val_log) + "\n")
        print(val_log, flush=True)
        val_rec = float(val.get("loss_rec", float("inf")))

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


# -----------------------------
# Inference helpers (encode / reconstruct)
# -----------------------------

def load_model(ckpt_path: Path, device: torch.device) -> Tuple[MotionSeq2SeqARAE, np.ndarray, np.ndarray, Dict]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing cfg")

    cfg_fields = set(getattr(TrainConfig, "__dataclass_fields__", {}).keys())
    if len(cfg_fields) > 0:
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
    cfg = TrainConfig(**cfg_dict)
    model, mean, std = build_model_and_stats(cfg)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        incompatible = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"[warn] load_state_dict strict=True failed: {e}\n"
            f"       fallback strict=False; missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    model.to(device)
    model.eval()
    return model, mean, std, cfg_dict


@torch.no_grad()
def encode_tokens(
    data_root: str,
    ckpt: str,
    out_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    latent_embed_mode: Optional[str] = None,
    a2b_mode: Optional[str] = None,
    parallel_teacher_ckpt: Optional[str] = None,
    compression_ratio: Optional[float] = None,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, cfg_dict = load_model(Path(ckpt), device)
    cfg = TrainConfig(**cfg_dict)
    embed_mode = latent_embed_mode if latent_embed_mode is not None else getattr(cfg, "latent_embed_mode", "sample")
    a2b_mode_eff = a2b_mode if a2b_mode is not None else getattr(cfg, "a2b_mode", "ar")
    b2a_mode_eff = getattr(cfg, "b2a_mode", "ar")
    compression_ratio_eff: Optional[float] = None
    if compression_ratio is not None and float(compression_ratio) > 0.0:
        compression_ratio_eff = float(compression_ratio)
    parallel_teacher_model: Optional[MotionSeq2SeqARAE] = None
    if str(a2b_mode_eff).lower().strip() == "tf_teacher":
        teacher_ckpt = str(parallel_teacher_ckpt or "").strip()
        if teacher_ckpt == "":
            teacher_ckpt = str(getattr(cfg, "parallel_teacher_ckpt", "")).strip()
        if teacher_ckpt == "":
            teacher_ckpt = str(getattr(cfg, "pretrain_ckpt", "")).strip()
        if teacher_ckpt == "":
            raise ValueError("a2b_mode=tf_teacher requires --parallel_teacher_ckpt (or cfg.parallel_teacher_ckpt/pretrain_ckpt)")
        parallel_teacher_model = copy.deepcopy(model).to(device)
        _load_pretrained_model(parallel_teacher_model, Path(teacher_ckpt), strict=bool(getattr(cfg, "pretrain_strict", False)))
        parallel_teacher_model.eval()
        for p in parallel_teacher_model.parameters():
            p.requires_grad = False

    ds = HumanML3DMotionDataset(Path(data_root), split, cfg.max_motion_len, mean=mean, std=std, random_crop=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(dl):
        x = batch["motion"].to(device)
        m = batch["mask"].to(device)
        paths = batch["path"]

        teacher_token_ids = None
        teacher_token_mask = None
        if str(a2b_mode_eff).lower().strip() == "tf_teacher":
            if parallel_teacher_model is None:
                raise RuntimeError("parallel_teacher_model is not initialized")
            t_out = parallel_teacher_model(
                x, m,
                task="ae",
                tau=cfg.gumbel_tau,
                scale=cfg.gumbel_scale,
                hard=True,
                top_k=cfg.top_k,
                deterministic_tokens=True,
                latent_embed_mode=getattr(cfg, "latent_embed_mode", "sample"),
                a2b_mode="logits",
                b2a_mode="parallel",
                compression_ratio_override=compression_ratio_eff,
                return_attn=False,
            )
            teacher_token_ids = t_out["token_ids"]
            teacher_token_mask = t_out["latent_mask"]

        out = model(
            x, m,
            task="ae",
            teacher_token_ids=teacher_token_ids,
            teacher_token_mask=teacher_token_mask,
            tau=cfg.gumbel_tau,
            scale=cfg.gumbel_scale,

            hard=True,
            # top_k=0,
            # deterministic_tokens=True,
            # hard=False,
            top_k=0,
            deterministic_tokens=False,
            
            latent_embed_mode=embed_mode,
            a2b_mode=a2b_mode_eff,
            b2a_mode=b2a_mode_eff,
            compression_ratio_override=compression_ratio_eff,
            return_attn=False,
        )
        token_ids = out["token_ids"].cpu().numpy()
        latent_mask = out["latent_mask"].cpu().numpy()

        for i, p in enumerate(paths):
            seq_id = Path(p).stem
            ids = token_ids[i][latent_mask[i]].tolist()
            (out_root / f"{seq_id}.txt").write_text(" ".join(map(str, ids)) + "\n", encoding="utf-8")


@torch.no_grad()
def reconstruct(
    data_root: str,
    ckpt: str,
    out_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    latent_embed_mode: Optional[str] = None,
    a2b_mode: Optional[str] = None,
    b2a_mode: Optional[str] = None,
    parallel_teacher_ckpt: Optional[str] = None,
    compression_ratio: Optional[float] = None,
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, cfg_dict = load_model(Path(ckpt), device)
    cfg = TrainConfig(**cfg_dict)
    embed_mode = latent_embed_mode if latent_embed_mode is not None else getattr(cfg, "latent_embed_mode", "sample")
    a2b_mode_eff = a2b_mode if a2b_mode is not None else getattr(cfg, "a2b_mode", "ar")
    b2a_mode_eff = b2a_mode if b2a_mode is not None else getattr(cfg, "b2a_mode", "ar")
    compression_ratio_eff: Optional[float] = None
    if compression_ratio is not None and float(compression_ratio) > 0.0:
        compression_ratio_eff = float(compression_ratio)
    parallel_teacher_model: Optional[MotionSeq2SeqARAE] = None
    if str(a2b_mode_eff).lower().strip() == "tf_teacher":
        teacher_ckpt = str(parallel_teacher_ckpt or "").strip()
        if teacher_ckpt == "":
            teacher_ckpt = str(getattr(cfg, "parallel_teacher_ckpt", "")).strip()
        if teacher_ckpt == "":
            teacher_ckpt = str(getattr(cfg, "pretrain_ckpt", "")).strip()
        if teacher_ckpt == "":
            raise ValueError("a2b_mode=tf_teacher requires --parallel_teacher_ckpt (or cfg.parallel_teacher_ckpt/pretrain_ckpt)")
        parallel_teacher_model = copy.deepcopy(model).to(device)
        _load_pretrained_model(parallel_teacher_model, Path(teacher_ckpt), strict=bool(getattr(cfg, "pretrain_strict", False)))
        parallel_teacher_model.eval()
        for p in parallel_teacher_model.parameters():
            p.requires_grad = False

    ds = HumanML3DMotionDataset(Path(data_root), split, cfg.max_motion_len, mean=mean, std=std, random_crop=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    out_root = Path(out_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)
    out_token_root = out_root / "tokens"
    out_token_root.mkdir(parents=True, exist_ok=True)

    # model.latent_len_max = model.latent_len_max * model.compression_ratio
    # model.compression_ratio = 1.0

    for batch in tqdm(dl):
        x = batch["motion"].to(device)
        m = batch["mask"].to(device)
        paths = batch["path"]

        teacher_token_ids = None
        teacher_token_mask = None
        if str(a2b_mode_eff).lower().strip() == "tf_teacher":
            if parallel_teacher_model is None:
                raise RuntimeError("parallel_teacher_model is not initialized")
            t_out = parallel_teacher_model(
                x, m,
                task="ae",
                tau=cfg.gumbel_tau,
                scale=cfg.gumbel_scale,
                hard=True,
                top_k=cfg.top_k,
                deterministic_tokens=True,
                latent_embed_mode=getattr(cfg, "latent_embed_mode", "sample"),
                a2b_mode="logits",
                b2a_mode="parallel",
                compression_ratio_override=compression_ratio_eff,
                return_attn=False,
            )
            teacher_token_ids = t_out["token_ids"]
            teacher_token_mask = t_out["latent_mask"]

        out = model(
            x, m,
            task="ae",

            teacher_token_ids=teacher_token_ids,
            teacher_token_mask=teacher_token_mask,

            tau=cfg.gumbel_tau,
            scale=cfg.gumbel_scale,
            
            hard=True,
            top_k=0,
            deterministic_tokens=True,
            # deterministic_tokens=False,

            latent_embed_mode=embed_mode,
            a2b_mode=a2b_mode_eff,
            b2a_mode=b2a_mode_eff,
            compression_ratio_override=compression_ratio_eff,
            return_attn=False,

            #for test
            # eval_teacher_forcing_decode=True,
            # l2p_a_drop_prob=0.9,
        )

        x_hat = out["x_hat"].cpu().numpy()     # [B,T_pad,D] normalized
        mask_pad = out["mask_pad"].cpu().numpy()  # [B,T_pad] bool
        latent_mask = out["latent_mask"].cpu().numpy()  # [B,Tb] bool
        token_ids = out["token_ids"].cpu().numpy()

        # de-normalize (frame-level)
        x_hat = x_hat * (std[None, None, :] + 1e-8) + mean[None, None, :]

        for i, p in enumerate(paths):
            seq_id = Path(p).stem
            valid_len = int(mask_pad[i].sum())
            arr = x_hat[i, :valid_len].astype(np.float32)
            np.save(out_root / f"{seq_id}.npy", arr)
            ids = token_ids[i][latent_mask[i]].astype(int).tolist()
            np.savetxt(out_token_root / f"{seq_id}.txt", ids, fmt="%d")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--data_root", type=str, required=True)
    ap_tr.add_argument("--save_dir", type=str, required=True)
    ap_tr.add_argument("--max_motion_len", type=int, default=196)
    ap_tr.add_argument("--patch_len", type=int, default=1)
    ap_tr.add_argument("--compression_ratio", type=float, default=1.0)
    ap_tr.add_argument(
        "--compression_ratio_min", type=float, default=0.0,
        help="If >0, enable variable compression training lower bound (sampled per batch in AR mode)."
    )
    ap_tr.add_argument(
        "--compression_ratio_max", type=float, default=0.0,
        help="If >0, enable variable compression training upper bound (sampled per batch in AR mode)."
    )
    ap_tr.add_argument(
        "--token_posenc_scale_with_compression", action="store_true",
        help="Use compression_ratio-scaled spacing for token positional encoding instead of cond_emb scaling (cond_emb is fixed as ratio=1)."
    )
    ap_tr.add_argument("--latent_len", type=int, default=0)

    ap_tr.add_argument("--vocab_size", type=int, default=1024)
    ap_tr.add_argument("--d_model", type=int, default=512)
    ap_tr.add_argument("--n_heads", type=int, default=8)
    ap_tr.add_argument("--d_ff", type=int, default=2048)
    ap_tr.add_argument("--dropout", type=float, default=0.1)
    ap_tr.add_argument(
        "--self_attn_drop_path", type=float, default=0.0,
        help="Drop-path prob for DecoderLayerFlex self-attn residual when memory is present (train only).",
    )
    ap_tr.add_argument("--enc_layers", type=int, default=6)
    ap_tr.add_argument("--dec_layers", type=int, default=6)
    ap_tr.add_argument("--recon_layers", type=int, default=6)
    ap_tr.add_argument("--a2b_mode", type=str, default="ar",
                       choices=["ar", "logits", "tf_logits", "tf_teacher"],
                       help="A->B path mode: ar (netB.generate), logits (netA.to_logits only), tf_logits (netA.to_logits -> netB.teacher_forcing_token_decode), tf_teacher (external teacher tokens -> netB.teacher_forcing_token_decode).")
    ap_tr.add_argument("--b2a_mode", type=str, default="ar",
                       choices=["ar", "parallel", "tf_proj"],
                       help="B->A path mode: ar (netA.generate), parallel (netB.out_proj), tf_proj (netB.out_proj -> netA.teacher_forcing_decode).")

    ap_tr.add_argument("--gumbel_tau", type=float, default=0.5)
    ap_tr.add_argument("--gumbel_scale", type=float, default=1.0) #0.2
    g_gumbel = ap_tr.add_mutually_exclusive_group()
    g_gumbel.add_argument("--gumbel_hard", dest="gumbel_hard", action="store_true", default=True,
                          help="Use straight-through Gumbel-Softmax (hard one-hot; default)")
    g_gumbel.add_argument("--gumbel_soft", dest="gumbel_hard", action="store_false",
                          help="Use soft Gumbel-Softmax (not straight-through)")
    ap_tr.add_argument("--top_k", type=int, default=0)
    ap_tr.add_argument(
        "--a2b_teacher_forcing_prob", type=float, default=0.0,
        help="A->B (ar mode, train only): probability of feeding netA.to_logits argmax token as teacher instead of free-run."
    )
    ap_tr.add_argument(
        "--latent_embed_mode", type=str, default="sample", choices=["sample", "softmax"],
        help="How to form token embeddings in A->B and feed them to B->A. sample: argmax/Gumbel; softmax: expected embedding (no discretization)."
    )
    ap_tr.add_argument(
        "--l2p_a_drop_prob", type=float, default=0.0,
        help="B->A only: drop A teacher-forcing inputs with this prob and feed back predicted embeddings."
    )
    ap_tr.add_argument(
        "--l2p_a_drop_mode", type=str, default="prob", choices=["prob", "mse_thresh"],
        help="B->A teacher forcing input selection mode. prob: random by l2p_a_drop_prob, mse_thresh: use free-run when MSE(y_t, teacher)<=threshold."
    )
    ap_tr.add_argument(
        "--l2p_a_drop_mse_thresh", type=float, default=0.0,
        help="MSE threshold on y_t space used when --l2p_a_drop_mode=mse_thresh."
    )
    ap_tr.add_argument(
        "--past_kv_recent_frames", type=int, default=0,
        help="If >0, limit AR self-attention KV cache to only the most recent N frames (B is auto-scaled by compression_ratio)."
    )

    ap_tr.add_argument("--rec_w", type=float, default=1.0)
    ap_tr.add_argument("--token_ce_w", type=float, default=0.0)
    ap_tr.add_argument("--a2b_supervised_kl_w", type=float, default=1.0,
                       help="Weight for a2b_mode=tf_teacher KL(netB TF logits || stopgrad(netA.to_logits)).")
    ap_tr.add_argument("--a_mem_next_w", type=float, default=0.0,
                       help="If >0, add masked MSE next-patch loss from initial memA (A->B->A).")
    ap_tr.add_argument("--norm_gap_enable", action="store_true")
    ap_tr.add_argument("--norm_gap_w", type=float, default=0.0)

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
    ap_tr.add_argument("--parallel_teacher_ckpt", type=str, default="",
                       help="Teacher checkpoint for a2b_mode=tf_teacher. If empty, --pretrain_ckpt is used.")


    ap_tr.add_argument("--train_stage", type=str, default="ae",
                       choices=["a_lm", "ae"],
                       help="Training stage: a_lm (train netA only) or ae (autoencoder).")
    ap_tr.add_argument("--freeze_netA", action="store_true",
                       help="Freeze all netA parameters (applies after train_stage logic).")
    ap_tr.add_argument("--freeze_netB", action="store_true",
                       help="Freeze all netB parameters (applies after train_stage logic).")
    ap_tr.add_argument("--freeze_a_cross_attn_trainable", action="store_true",
                       help="When --freeze_netA is set, keep netA cross-attn (+norm2) trainable (optional).")

    # encode
    ap_en = sub.add_parser("encode")
    ap_en.add_argument("--data_root", type=str, required=True)
    ap_en.add_argument("--ckpt", type=str, required=True)
    ap_en.add_argument("--out_dir", type=str, required=True)
    ap_en.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap_en.add_argument("--batch_size", type=int, default=64)
    ap_en.add_argument("--num_workers", type=int, default=4)
    ap_en.add_argument("--seed", type=int, default=0)
    ap_en.add_argument("--latent_embed_mode", type=str, default="from_ckpt", choices=["from_ckpt", "sample", "softmax"])
    ap_en.add_argument("--a2b_mode", type=str, default="from_ckpt",
                       choices=["from_ckpt", "ar", "logits", "tf_logits", "tf_teacher"])
    ap_en.add_argument("--parallel_teacher_ckpt", type=str, default="",
                       help="Teacher checkpoint for encode a2b_mode=tf_teacher. If empty, use cfg.parallel_teacher_ckpt then cfg.pretrain_ckpt.")
    ap_en.add_argument("--compression_ratio", type=float, default=0.0,
                       help="If >0, override compression ratio during encode to control token count. <=0 uses checkpoint setting.")

    # reconstruct
    ap_rc = sub.add_parser("reconstruct")
    ap_rc.add_argument("--data_root", type=str, required=True)
    ap_rc.add_argument("--ckpt", type=str, required=True)
    ap_rc.add_argument("--out_dir", type=str, required=True)
    ap_rc.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap_rc.add_argument("--batch_size", type=int, default=64)
    ap_rc.add_argument("--num_workers", type=int, default=4)
    ap_rc.add_argument("--seed", type=int, default=0)
    ap_rc.add_argument("--latent_embed_mode", type=str, default="from_ckpt", choices=["from_ckpt", "sample", "softmax"])
    ap_rc.add_argument("--a2b_mode", type=str, default="from_ckpt",
                       choices=["from_ckpt", "ar", "logits", "tf_logits", "tf_teacher"])
    ap_rc.add_argument("--b2a_mode", type=str, default="from_ckpt",
                       choices=["from_ckpt", "ar", "parallel", "tf_proj"])
    ap_rc.add_argument("--parallel_teacher_ckpt", type=str, default="",
                       help="Teacher checkpoint for reconstruct a2b_mode=tf_teacher. If empty, use cfg.parallel_teacher_ckpt then cfg.pretrain_ckpt.")
    ap_rc.add_argument("--compression_ratio", type=float, default=0.0,
                       help="If >0, override compression ratio during reconstruct to control token count. <=0 uses checkpoint setting.")

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
            a2b_mode=None if getattr(args, "a2b_mode", "from_ckpt") == "from_ckpt" else args.a2b_mode,
            parallel_teacher_ckpt=str(getattr(args, "parallel_teacher_ckpt", "")).strip() or None,
            compression_ratio=float(getattr(args, "compression_ratio", 0.0)),
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
            a2b_mode=None if getattr(args, "a2b_mode", "from_ckpt") == "from_ckpt" else args.a2b_mode,
            b2a_mode=None if getattr(args, "b2a_mode", "from_ckpt") == "from_ckpt" else args.b2a_mode,
            parallel_teacher_ckpt=str(getattr(args, "parallel_teacher_ckpt", "")).strip() or None,
            compression_ratio=float(getattr(args, "compression_ratio", 0.0)),
        )
        return

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
