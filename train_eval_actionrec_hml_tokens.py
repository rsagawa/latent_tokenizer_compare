#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_actionrec_hml_tokens.py

Token-swap experiment 2) Action Recognition on HumanML3D using caption-derived labels.
Run twice with different --token_root and compare actionrec_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hml_tokens_data import HMLTokensDataset, collate_tokens


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pair_action_map(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    amap: Dict[str, List[str]] = {}
    for action, adjective in pairs:
        cls_name = f"{action}__{adjective}"
        action_sp = action.replace("-", " ")
        adjective_sp = adjective.replace("-", " ")
        amap[cls_name] = [
            f"{action_sp} {adjective_sp}",
            f"{action} {adjective}",
        ]
    return amap


HUMANML3D_ACTION_MAP: Dict[str, List[str]] = {
    "walk": ["walk", "walking", "stroll"],
    "run": ["run", "running", "jog"],
    "jump": ["jump", "jumping", "hop", "hopping", "leap", "leaping"],
    "sit": ["sit", "sitting", "sit down"],
    "stand": ["stand", "standing", "stand up"],
    "turn": ["turn", "turning", "rotate", "rotating"],
    "dance": ["dance", "dancing"],
    "kick": ["kick", "kicking"],
    "punch": ["punch", "punching"],
    "throw": ["throw", "throwing", "toss"],
    "catch": ["catch", "catching"],
    "clap": ["clap", "clapping"],
    "wave": ["wave", "waving"],
    "lift": ["lift", "lifting", "pick up", "pickup"],
    "bend": ["bend", "bending"],
    "crouch": ["crouch", "crouching", "squat", "squatting"],
    "crawl": ["crawl", "crawling"],
    "push": ["push", "pushing"],
    "pull": ["pull", "pulling"],
}

_bandai1_common_actions = [
    "bow", "bye", "byebye", "dash", "guide",
    "run", "walk", "walk-back", "walk-left", "walk-right",
]
_bandai1_adjectives = [
    "active", "angry", "childish", "chimpira", "feminine",
    "giant", "happy", "masculinity", "musical", "normal",
    "not-confident", "old", "proud", "sad", "tired",
]
_bandai1_common_pairs = [
    (a, adj) for a in _bandai1_common_actions for adj in _bandai1_adjectives
]
_bandai1_normal_only_pairs = [
    ("call", "normal"),
    ("dance-long", "normal"),
    ("dance-short", "normal"),
    ("kick", "normal"),
    ("punch", "normal"),
    ("respond", "normal"),
    ("slash", "normal"),
]
BANDAI1_ACTION_MAP: Dict[str, List[str]] = _pair_action_map(
    _bandai1_common_pairs + _bandai1_normal_only_pairs
)

_bandai2_actions = [
    "raise-up-both-hands",
    "raise-up-left-hand",
    "raise-up-right-hand",
    "run",
    "walk",
    "walk-turn-left",
    "walk-turn-right",
    "wave-both-hands",
    "wave-left-hand",
    "wave-right-hand",
]
_bandai2_adjectives = [
    "active", "elderly", "exhausted", "feminine",
    "masculine", "normal", "youthful",
]
BANDAI2_ACTION_MAP: Dict[str, List[str]] = _pair_action_map(
    [(a, adj) for a in _bandai2_actions for adj in _bandai2_adjectives]
)

ACTION_MAPS: Dict[str, Dict[str, List[str]]] = {
    "humanml3d": HUMANML3D_ACTION_MAP,
    "bandai1": BANDAI1_ACTION_MAP,
    "bandai2": BANDAI2_ACTION_MAP,
}

ACTION_MAP: Dict[str, List[str]] = ACTION_MAPS["humanml3d"]


CLASS_NAMES = sorted(ACTION_MAP.keys())
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}


def set_action_map(dataset_key: str) -> None:
    global ACTION_MAP, CLASS_NAMES, CLASS_TO_ID
    if dataset_key not in ACTION_MAPS:
        raise ValueError(f"unknown dataset key: {dataset_key}")
    ACTION_MAP = ACTION_MAPS[dataset_key]
    CLASS_NAMES = sorted(ACTION_MAP.keys())
    CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}


def infer_dataset_key(hml_root: str, token_root: str, mids: List[str]) -> str:
    has_d1 = any(mid.startswith("dataset-1_") for mid in mids)
    has_d2 = any(mid.startswith("dataset-2_") for mid in mids)
    if has_d1 and has_d2:
        raise ValueError("mixed Bandai1 and Bandai2 IDs are not supported in one run")
    if has_d1:
        return "bandai1"
    if has_d2:
        return "bandai2"

    joined = f"{hml_root} {token_root}".lower()
    if "bandai2" in joined:
        return "bandai2"
    if "bandai" in joined:
        return "bandai1"
    return "humanml3d"


def caption_to_label(caption: str) -> Optional[int]:
    cap = (caption or "").strip().lower()
    if not cap:
        return None
    for cname, kws in ACTION_MAP.items():
        for kw in kws:
            if " " in kw and kw in cap:
                return CLASS_TO_ID[cname]
    words = cap.replace(".", " ").replace(",", " ").split()
    for cname, kws in ACTION_MAP.items():
        for kw in kws:
            if " " not in kw and kw in words:
                return CLASS_TO_ID[cname]
    return None


class ActionRecDataset(Dataset):
    def __init__(self, base_ds: HMLTokensDataset):
        self.items = []
        for i in range(len(base_ds)):
            tok, cap, toks, mid = base_ds[i]
            y = caption_to_label(cap)
            if y is None or len(tok) == 0:
                continue
            self.items.append((tok, y, mid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tok, y, mid = self.items[idx]
        return tok, y, mid


def collate_action(batch, vocab_size: int, max_len: int):
    fake = [(b[0], "", [], b[2]) for b in batch]
    x, mask, lengths, _caps, _toks, mids = collate_tokens(fake, vocab_size=vocab_size, max_len=max_len)
    y = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return x, mask, lengths, y, mids


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


class ActionClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, d_model: int, n_heads: int, n_layers: int, max_len: int, dropout: float):
        super().__init__()
        self.pad_id = int(vocab_size)
        self.emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=self.pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=~mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = (h * mask.unsqueeze(-1).to(h.dtype)).sum(dim=1) / denom
        return self.head(pooled)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1s))


@torch.no_grad()
def eval_split(model: ActionClassifier, loader: DataLoader, device: torch.device, n_classes: int):
    model.eval()
    ys, ps = [], []
    for x, mask, _lens, y, _mids in loader:
        x = x.to(device); mask = mask.to(device); y = y.to(device)
        pred = model(x, mask).argmax(dim=-1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    if not ys:
        return {"acc": float("nan"), "macro_f1": float("nan")}
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return {"acc": float(np.mean(y_true == y_pred)), "macro_f1": macro_f1(y_true, y_pred, n_classes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hml_root", type=str, required=True)
    ap.add_argument("--token_root", type=str, required=True)
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_train = HMLTokensDataset(args.hml_root, args.token_root, "train", max_tokens=args.max_tokens, drop_if_missing=True)
    base_val = HMLTokensDataset(args.hml_root, args.token_root, "val", max_tokens=args.max_tokens, drop_if_missing=True)
    base_test = HMLTokensDataset(args.hml_root, args.token_root, "test", max_tokens=args.max_tokens, drop_if_missing=True)
    all_mids = list(base_train.ids) + list(base_val.ids) + list(base_test.ids)
    dataset_key = infer_dataset_key(args.hml_root, args.token_root, all_mids)
    set_action_map(dataset_key)
    print(f"[dataset] using {dataset_key} ACTION_MAP with {len(CLASS_NAMES)} classes")

    ds_train, ds_val, ds_test = ActionRecDataset(base_train), ActionRecDataset(base_val), ActionRecDataset(base_test)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,
                          collate_fn=lambda b: collate_action(b, args.vocab_size, args.max_tokens), drop_last=False)

    tr, va, te = loader(ds_train, True), loader(ds_val, False), loader(ds_test, False)

    n_classes = len(CLASS_NAMES)
    model = ActionClassifier(args.vocab_size, n_classes, args.d_model, args.n_heads, args.n_layers, args.max_tokens, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1.0
    best_path = out_dir / "best.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, mask, _lens, y, _mids in tr:
            x = x.to(device); mask = mask.to(device); y = y.to(device)
            loss = F.cross_entropy(model(x, mask), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        valm = eval_split(model, va, device, n_classes)
        print(f"[ep {ep:03d}] train_loss={float(np.mean(losses)):.4f} val_acc={valm['acc']:.4f} val_macro_f1={valm['macro_f1']:.4f} n_train={len(ds_train)}")

        if valm["macro_f1"] > best:
            best = valm["macro_f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": CLASS_NAMES}, str(best_path))

    ck = torch.load(str(best_path), map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    testm = eval_split(model, te, device, n_classes)

    out = {
        "dataset": dataset_key,
        "token_root": args.token_root,
        "hml_root": args.hml_root,
        "vocab_size": int(args.vocab_size),
        "classes": CLASS_NAMES,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "test": testm,
    }
    (out_dir / "actionrec_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
