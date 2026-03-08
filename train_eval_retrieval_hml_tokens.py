#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_retrieval_hml_tokens.py

Token-swap experiment 1) Text–Motion Retrieval on HumanML3D.

Run twice with different --token_root to compare tokenizers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hml_tokens_data import (
    HMLTokensDataset,
    collate_tokens,
    load_word_vectorizer,
    build_word_pos_tensors,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class MotionTokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int, dropout: float):
        super().__init__()
        self.pad_id = int(vocab_size)
        self.emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=self.pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=~mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = (h * mask.unsqueeze(-1).to(h.dtype)).sum(dim=1) / denom
        return F.normalize(self.proj(pooled), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(315, d_model)  # 300 + 15
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word: torch.Tensor, pos: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = torch.cat([word, pos], dim=-1)
        h = self.dropout(torch.tanh(self.inp(x)))
        B, L, D = h.shape
        mask = torch.arange(L, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = (h * mask.unsqueeze(-1).to(h.dtype)).sum(dim=1) / denom
        return F.normalize(self.proj(pooled), dim=-1)


class DualEncoder(nn.Module):
    def __init__(self, motion_enc: MotionTokenEncoder, text_enc: TextEncoder, temperature: float):
        super().__init__()
        self.motion_enc = motion_enc
        self.text_enc = text_enc
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, x_tok, m_tok, word, pos, tlen):
        m = self.motion_enc(x_tok, m_tok)
        t = self.text_enc(word, pos, tlen)
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        logits = scale * (t @ m.t())
        return logits, t, m


def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
    B = logits.size(0)
    target = torch.arange(B, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


@torch.no_grad()
def retrieval_metrics(emb_a: np.ndarray, emb_b: np.ndarray) -> Dict[str, float]:
    sim = emb_a @ emb_b.T
    ranks = []
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
    ranks = np.asarray(ranks)
    out = {f"R@{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10)}
    out["MedR"] = float(np.median(ranks))
    out["MeanR"] = float(np.mean(ranks))
    return out


@torch.no_grad()
def retrieval_metrics_small(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    *,
    candidates: int,
    top_ks: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    """
    MDM/MotionGPT-style small retrieval:
    for each query, rank the GT pair inside a small candidate pool (e.g. 32).
    """
    n = min(len(emb_a), len(emb_b))
    if n <= 1:
        return {}

    c = int(max(2, min(candidates, n)))
    ks = sorted({int(k) for k in top_ks if int(k) >= 1})
    if not ks:
        ks = [1, 2, 3]
    rep = int(max(1, repeats))

    rng = np.random.default_rng(int(seed))
    hit_counts = np.zeros(len(ks), dtype=np.float64)
    all_ranks = []

    for _ in range(rep):
        for i in range(n):
            if c == n:
                cand_idx = np.arange(n, dtype=np.int64)
            else:
                pool = np.concatenate([np.arange(i, dtype=np.int64), np.arange(i + 1, n, dtype=np.int64)])
                neg = rng.choice(pool, size=c - 1, replace=False)
                cand_idx = np.concatenate([np.asarray([i], dtype=np.int64), neg])
                rng.shuffle(cand_idx)

            scores = emb_a[i] @ emb_b[cand_idx].T
            order = np.argsort(-scores)
            gt_local = int(np.where(cand_idx == i)[0][0])
            rank = int(np.where(order == gt_local)[0][0]) + 1
            all_ranks.append(rank)
            for j, k in enumerate(ks):
                if rank <= k:
                    hit_counts[j] += 1.0

    denom = float(len(all_ranks))
    out: Dict[str, float] = {f"R_precision_top_{k}": float(hit_counts[j] / denom) for j, k in enumerate(ks)}
    ranks_np = np.asarray(all_ranks, dtype=np.float64)
    out["MedR_small"] = float(np.median(ranks_np))
    out["MeanR_small"] = float(np.mean(ranks_np))
    out["small_candidates"] = int(c)
    out["small_repeats"] = int(rep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motiongpt_root", type=str, required=True)
    ap.add_argument("--hml_root", type=str, required=True)
    ap.add_argument("--token_root", type=str, required=True)
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_text_len", type=int, default=20)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--pretrained_ckpt", type=str, default="",
                    help="Path to pretrained checkpoint. If set, skip training and evaluate this model directly.")
    ap.add_argument("--small_retrieval_candidates", type=int, default=0,
                    help="If >0, evaluate MDM/MotionGPT-style small retrieval with this candidate pool size (e.g. 32).")
    ap.add_argument("--small_retrieval_repeats", type=int, default=1,
                    help="Number of repeated random samplings for small retrieval (averaged).")
    ap.add_argument("--small_retrieval_topk", type=str, default="1,2,3",
                    help="Comma-separated k list for R_precision_top_k in small retrieval.")
    args = ap.parse_args()

    seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wvec = load_word_vectorizer(Path(args.motiongpt_root))

    def make_loader(split: str, shuffle: bool):
        ds = HMLTokensDataset(args.hml_root, args.token_root, split, max_tokens=args.max_tokens, drop_if_missing=True)

        def _collate(batch):
            x, mask, lens, caps, toks, mids = collate_tokens(batch, vocab_size=args.vocab_size, max_len=args.max_tokens)
            word_list, pos_list, tlen_list = [], [], []
            for tt in toks:
                w, p, L = build_word_pos_tensors(wvec, tt, max_text_len=args.max_text_len)
                word_list.append(w); pos_list.append(p); tlen_list.append(L)
            word = torch.from_numpy(np.stack(word_list, axis=0))
            pos = torch.from_numpy(np.stack(pos_list, axis=0))
            tlen = torch.tensor(tlen_list, dtype=torch.long)
            return x, mask, word, pos, tlen, mids

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,
                            collate_fn=_collate, drop_last=True)
        return loader, len(ds)

    train_loader, n_train = make_loader("train", True)
    val_loader, n_val = make_loader("val", False)
    test_loader, n_test = make_loader("test", False)

    model = DualEncoder(
        MotionTokenEncoder(args.vocab_size, args.d_model, args.n_heads, args.n_layers, args.max_tokens, args.dropout),
        TextEncoder(args.d_model, args.dropout),
        temperature=args.temperature
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = out_dir / "best.pt"

    if str(args.pretrained_ckpt).strip():
        ck_path = Path(args.pretrained_ckpt)
        if not ck_path.is_file():
            raise FileNotFoundError(f"--pretrained_ckpt not found: {ck_path}")
        ck = torch.load(str(ck_path), map_location="cpu")
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        print(f"[eval-only] loaded pretrained checkpoint: {ck_path}")
    else:
        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, mask, word, pos, tlen, _mids in train_loader:
                x = x.to(device); mask = mask.to(device)
                word = word.to(device); pos = pos.to(device); tlen = tlen.to(device)
                logits, _, _ = model(x, mask, word, pos, tlen)
                loss = info_nce_loss(logits)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            train_loss = float(np.mean(losses)) if losses else float("nan")

            model.eval()
            vlosses = []
            with torch.no_grad():
                for x, mask, word, pos, tlen, _mids in val_loader:
                    x = x.to(device); mask = mask.to(device)
                    word = word.to(device); pos = pos.to(device); tlen = tlen.to(device)
                    logits, _, _ = model(x, mask, word, pos, tlen)
                    vlosses.append(float(info_nce_loss(logits).detach().cpu().item()))
            val_loss = float(np.mean(vlosses)) if vlosses else float("nan")

            print(f"[ep {ep:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}  n_train={n_train} n_val={n_val}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "args": vars(args)}, str(best_path))

        ck = torch.load(str(best_path), map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        model.to(device).eval()

    all_t, all_m = [], []
    with torch.no_grad():
        for x, mask, word, pos, tlen, _mids in test_loader:
            x = x.to(device); mask = mask.to(device)
            word = word.to(device); pos = pos.to(device); tlen = tlen.to(device)
            _, t, m = model(x, mask, word, pos, tlen)
            all_t.append(t.detach().cpu().numpy())
            all_m.append(m.detach().cpu().numpy())

    emb_t = np.concatenate(all_t, axis=0)
    emb_m = np.concatenate(all_m, axis=0)
    N = min(len(emb_t), len(emb_m))
    emb_t = emb_t[:N]; emb_m = emb_m[:N]

    metrics_t2m = retrieval_metrics(emb_t, emb_m)
    metrics_m2t = retrieval_metrics(emb_m, emb_t)

    small_t2m = {}
    small_m2t = {}
    if int(args.small_retrieval_candidates) > 0:
        topks = [int(x.strip()) for x in str(args.small_retrieval_topk).split(",") if x.strip()]
        small_t2m = retrieval_metrics_small(
            emb_t, emb_m,
            candidates=int(args.small_retrieval_candidates),
            top_ks=topks,
            repeats=int(args.small_retrieval_repeats),
            seed=int(args.seed),
        )
        small_m2t = retrieval_metrics_small(
            emb_m, emb_t,
            candidates=int(args.small_retrieval_candidates),
            top_ks=topks,
            repeats=int(args.small_retrieval_repeats),
            seed=int(args.seed),
        )

    out = {
        "token_root": args.token_root,
        "vocab_size": int(args.vocab_size),
        "n_test_used": int(N),
        "t2m": metrics_t2m,
        "m2t": metrics_m2t,
        "t2m_small_retrieval": small_t2m,
        "m2t_small_retrieval": small_m2t,
    }
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
