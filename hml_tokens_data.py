#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hml_tokens_data.py

Utilities for HumanML3D token-based experiments:
- load split IDs
- load motion-token sequences from token_root
- load "full-motion" caption tokens from HumanML3D texts/<id>.txt
- encode text tokens using MotionGPT's WordVectorizer (GloVe + POS one-hot), no internet needed

Assumptions
-----------
HumanML3D root has:
  - train.txt / val.txt / test.txt   (IDs)
  - texts/<id>.txt                   (caption lines)
Token root has (either layout is accepted):
  - <token_root>/<split>/<id>.txt
  - <token_root>/tokens/<split>/<id>.txt
Each token txt is whitespace-separated ints (token IDs).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_LINE_RE = re.compile(r"^([^#]+)#([^#]+)#([^#]+)#([^#]+)$")


def add_motiongpt_to_path(motiongpt_root: str) -> None:
    p = Path(motiongpt_root).resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def read_split_ids(hml_root: Path, split: str) -> List[str]:
    p = hml_root / f"{split}.txt"
    if not p.is_file():
        raise FileNotFoundError(f"split file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def token_file_path(token_root: Path, split: str, mid: str) -> Path:
    p1 = token_root / split / f"{mid}.txt"
    if p1.is_file():
        return p1
    return token_root / "tokens" / split / f"{mid}.txt"


def load_token_ids(path: Path) -> List[int]:
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    return [int(x) for x in txt.split()]


def parse_hml_line(line: str) -> Optional[Tuple[str, List[str], float, float]]:
    line = line.strip()
    if not line:
        return None
    m = _LINE_RE.match(line)
    if not m:
        cap = line.split("#", 1)[0].strip()
        return (cap, [], 0.0, 0.0) if cap else None
    cap = m.group(1).strip()
    tok_str = m.group(2).strip()
    toks = [t for t in tok_str.split(" ") if t]
    try:
        f_tag = float(m.group(3))
        to_tag = float(m.group(4))
    except Exception:
        f_tag, to_tag = 0.0, 0.0
    return cap, toks, f_tag, to_tag


def pick_fullmotion_tokens(text_path: Path) -> Tuple[str, List[str]]:
    if not text_path.is_file():
        return ("", [])
    lines = text_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    parsed = [parse_hml_line(ln) for ln in lines]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return ("", [])
    for cap, toks, f_tag, to_tag in parsed:
        if float(f_tag) == 0.0 and float(to_tag) == 0.0 and cap:
            return cap, toks
    cap, toks, _, _ = parsed[0]
    return cap, toks


@dataclass
class TextBatch:
    word: torch.Tensor
    pos: torch.Tensor
    length: torch.Tensor


def build_word_pos_tensors(wvec, t_tokens: List[str], max_text_len: int = 20):
    toks = ["sos/OTHER"] + list(t_tokens) + ["eos/OTHER"]
    if len(toks) < 2:
        toks = ["sos/OTHER", "eos/OTHER"]
    if len(toks) > max_text_len + 2:
        toks = toks[: max_text_len + 2]
    sent_len = int(len(toks))
    pad_len = (max_text_len + 2) - len(toks)
    if pad_len > 0:
        toks = toks + ["unk/OTHER"] * pad_len

    word_embs = []
    pos_ohots = []
    for tok in toks:
        w, p = wvec[tok]
        word_embs.append(w)
        pos_ohots.append(p)

    word_embs = np.asarray(word_embs, dtype=np.float32)
    pos_ohots = np.asarray(pos_ohots, dtype=np.float32)
    return word_embs, pos_ohots, sent_len


def load_word_vectorizer(motiongpt_root: Path, word_vec_root: Optional[Path] = None):
    add_motiongpt_to_path(str(motiongpt_root))
    from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # type: ignore

    if word_vec_root is None:
        word_vec_root = motiongpt_root / "deps" / "glove"
    word_vec_root = Path(word_vec_root)

    prefix = None
    for cand in ["our_vab", "glove"]:
        if (word_vec_root / f"{cand}_data.npy").is_file():
            prefix = cand
            break
    if prefix is None:
        data_files = list(word_vec_root.glob("*_data.npy"))
        if not data_files:
            raise FileNotFoundError(f"No *_data.npy found in {word_vec_root}")
        prefix = data_files[0].stem.replace("_data", "")

    return WordVectorizer(str(word_vec_root), prefix)


class HMLTokensDataset(Dataset):
    def __init__(self, hml_root: str, token_root: str, split: str, max_tokens: int = 1024, drop_if_missing: bool = True):
        self.hml_root = Path(hml_root)
        self.token_root = Path(token_root)
        self.split = split
        self.max_tokens = int(max_tokens)
        ids = read_split_ids(self.hml_root, split)

        keep: List[str] = []
        for mid in ids:
            tokp = token_file_path(self.token_root, split, mid)
            txtp = self.hml_root / "texts" / f"{mid}.txt"
            if drop_if_missing and (not tokp.is_file() or not txtp.is_file()):
                continue
            keep.append(mid)
        self.ids = keep

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        mid = self.ids[idx]
        tokp = token_file_path(self.token_root, self.split, mid)
        txtp = self.hml_root / "texts" / f"{mid}.txt"
        token_ids = load_token_ids(tokp) if tokp.is_file() else []
        token_ids = token_ids[: self.max_tokens] if self.max_tokens > 0 else token_ids
        cap, toks = pick_fullmotion_tokens(txtp)
        return token_ids, cap, toks, mid


def collate_tokens(batch, vocab_size: int, max_len: int):
    pad_id = int(vocab_size)
    seqs, caps, toks, mids = zip(*batch)
    lengths = [min(len(s), max_len) for s in seqs]
    T = max(lengths) if lengths else 1
    T = min(T, max_len)
    x = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ss = s[:T]
        if len(ss) > 0:
            x[i, :len(ss)] = torch.tensor(ss, dtype=torch.long)
            mask[i, :len(ss)] = True
    return x, mask, torch.tensor(lengths, dtype=torch.long), list(caps), list(toks), list(mids)
