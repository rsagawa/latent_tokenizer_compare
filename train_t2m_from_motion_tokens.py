#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_t2m_from_motion_tokens.py

Train a small text-to-motion (T2M) model that predicts *discrete motion tokens*.

Intended use
------------
1) First, discretize HumanML3D motions with your tokenizer script:
     python motion_tokenizer_humanml3d_seq2seq_two_networks.py encode \
       --data_root /path/to/HumanML3D/new_joint_vecs \
       --ckpt runs/seq2seq_tok/checkpoints/last.pt \
       --out_dir runs/seq2seq_tok/tokens

   This typically creates token files like:
     runs/seq2seq_tok/tokens/<split>/<id>.txt

2) Then train T2M on (text -> token_ids):
     python train_t2m_from_motion_tokens.py train \
       --hml_root /path/to/HumanML3D \
       --token_root runs/seq2seq_tok/tokens \
       --save_dir runs/t2m_tok \
       --motion_vocab_size 1024 \
       --d_model 512 --n_heads 8 --enc_layers 4 --dec_layers 4 \
       --batch_size 64 --epochs 50

3) (Optional) generate tokens from text:
     python train_t2m_from_motion_tokens.py generate \
       --ckpt runs/t2m_tok/checkpoints/best.pt \
       --text "a person walks forward" \
       --out_txt out_tokens.txt

Notes
-----
* This script is standalone (plain PyTorch). It does NOT depend on MotionGPT.
* Text tokenization defaults to a simple word-level vocab built from train captions.
  If you prefer, pass --hf_tokenizer to use a HuggingFace tokenizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Repro
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# HumanML3D text utilities
# -----------------------------

_WS_RE = re.compile(r"\s+")


def _clean_caption(s: str) -> str:
    s = s.strip()
    # HumanML3D caption lines often look like:
    #   caption#tokens#f_tag#to_tag
    # We only need the caption part.
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    s = _WS_RE.sub(" ", s)
    return s


def read_hml_caption_file(txt_path: Path) -> List[str]:
    if not txt_path.is_file():
        return []
    lines = []
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        cap = _clean_caption(ln)
        if cap:
            lines.append(cap)
    return lines


def read_split_ids(hml_root: Path, split: str) -> List[str]:
    p = hml_root / f"{split}.txt"
    if not p.is_file():
        raise FileNotFoundError(f"split file not found: {p}")
    ids = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return ids


def resolve_text_dir(hml_root: Path) -> Path:
    # Common variants
    for cand in ["texts", "text", "caption", "captions"]:
        p = hml_root / cand
        if p.is_dir():
            return p
    # Fallback: return default name (will yield empty captions)
    return hml_root / "texts"


def resolve_token_path(token_root: Path, split: str, mid: str) -> Optional[Path]:
    """Try multiple common layouts.

    - <token_root>/<split>/<id>.txt
    - <token_root>/tokens/<split>/<id>.txt
    """
    cands = [
        token_root / split / f"{mid}.txt",
        token_root / "tokens" / split / f"{mid}.txt",
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def read_token_ids(path: Path) -> List[int]:
    # One id per line (np.savetxt-like) or space-separated.
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    parts = re.split(r"[\s,]+", txt)
    out: List[int] = []
    for t in parts:
        if not t:
            continue
        out.append(int(t))
    return out


# -----------------------------
# Simple word tokenizer
# -----------------------------


@dataclass
class WordVocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = [t for t in re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", text.lower()) if t.strip()]
        ids = [self.bos_id]
        for t in toks[: max(0, max_len - 2)]:
            ids.append(self.stoi.get(t, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        words = []
        for i in ids:
            if i in (self.pad_id, self.bos_id):
                continue
            if i == self.eos_id:
                break
            if 0 <= i < len(self.itos):
                words.append(self.itos[i])
            else:
                words.append("<unk>")
        return " ".join(words)


def build_word_vocab(captions: Iterable[str], max_size: int = 30000, min_freq: int = 2) -> WordVocab:
    from collections import Counter

    counter: Counter[str] = Counter()
    for cap in captions:
        toks = [t for t in re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", cap.lower()) if t.strip()]
        counter.update(toks)

    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    itos = list(specials)

    # Reserve space for specials
    budget = max(0, int(max_size) - len(specials))
    for tok, freq in counter.most_common():
        if freq < int(min_freq):
            break
        if tok in specials:
            continue
        itos.append(tok)
        if len(itos) >= len(specials) + budget:
            break

    stoi = {t: i for i, t in enumerate(itos)}
    return WordVocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi["<pad>"],
        bos_id=stoi["<bos>"],
        eos_id=stoi["<eos>"],
        unk_id=stoi["<unk>"],
    )


def save_word_vocab(v: WordVocab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "itos": v.itos,
        "pad_id": v.pad_id,
        "bos_id": v.bos_id,
        "eos_id": v.eos_id,
        "unk_id": v.unk_id,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_word_vocab(path: Path) -> WordVocab:
    obj = json.loads(path.read_text(encoding="utf-8"))
    itos = list(obj["itos"])
    stoi = {t: i for i, t in enumerate(itos)}
    return WordVocab(
        stoi=stoi,
        itos=itos,
        pad_id=int(obj["pad_id"]),
        bos_id=int(obj["bos_id"]),
        eos_id=int(obj["eos_id"]),
        unk_id=int(obj["unk_id"]),
    )


# -----------------------------
# Dataset
# -----------------------------


@dataclass
class T2MSample:
    text: str
    text_ids: List[int]
    motion_ids: List[int]


class HumanML3DTextToTokens(Dataset):
    def __init__(
        self,
        hml_root: str | Path,
        token_root: str | Path,
        split: str,
        text_vocab: WordVocab,
        max_text_len: int = 32,
        min_motion_tokens: int = 1,
        max_motion_tokens: int = 2048,
        pick_one_text: bool = True,
        motion_pad_id: int = 0,
        motion_bos_id: int = 0,
        motion_eos_id: int = 0,
    ) -> None:
        super().__init__()
        self.hml_root = Path(hml_root)
        self.token_root = Path(token_root)
        self.split = str(split)
        self.text_vocab = text_vocab
        self.max_text_len = int(max_text_len)
        self.min_motion_tokens = int(min_motion_tokens)
        self.max_motion_tokens = int(max_motion_tokens)
        self.pick_one_text = bool(pick_one_text)
        self.motion_pad_id = int(motion_pad_id)
        self.motion_bos_id = int(motion_bos_id)
        self.motion_eos_id = int(motion_eos_id)

        self.text_dir = resolve_text_dir(self.hml_root)
        ids_all = read_split_ids(self.hml_root, self.split)

        items: List[Tuple[str, str, Path]] = []
        dropped_missing_tok = 0
        dropped_missing_txt = 0
        dropped_len = 0

        for mid in ids_all:
            tok_path = resolve_token_path(self.token_root, self.split, mid)
            if tok_path is None:
                dropped_missing_tok += 1
                continue
            txt_path = self.text_dir / f"{mid}.txt"
            caps = read_hml_caption_file(txt_path)
            if len(caps) == 0:
                dropped_missing_txt += 1
                continue

            motion_ids = read_token_ids(tok_path)
            if not (self.min_motion_tokens <= len(motion_ids) <= self.max_motion_tokens):
                dropped_len += 1
                continue

            # Store one representative caption; optionally re-sample at __getitem__.
            items.append((mid, caps[0], tok_path))

        if len(items) == 0:
            msg = (
                f"T2M dataset is empty. split={self.split}, hml_root={self.hml_root}, token_root={self.token_root}. "
                f"Dropped: missing_tok={dropped_missing_tok}, missing_text={dropped_missing_txt}, bad_len={dropped_len}."
            )
            raise ValueError(msg)

        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> T2MSample:
        mid, cap0, tok_path = self._items[idx]

        # (Optional) pick random caption
        if self.pick_one_text:
            caps = read_hml_caption_file(self.text_dir / f"{mid}.txt")
            text = random.choice(caps) if caps else cap0
        else:
            text = cap0

        text_ids = self.text_vocab.encode(text, max_len=self.max_text_len)

        motion_ids = read_token_ids(tok_path)
        # Add BOS/EOS so generation can stop.
        motion_ids = [self.motion_bos_id] + motion_ids + [self.motion_eos_id]

        return T2MSample(text=text, text_ids=text_ids, motion_ids=motion_ids)


def _pad_1d(seqs: List[List[int]], pad: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), int(pad), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate_t2m(batch: List[T2MSample], text_pad: int, motion_pad: int) -> Dict[str, torch.Tensor]:
    text_ids = _pad_1d([b.text_ids for b in batch], pad=text_pad)
    motion_ids = _pad_1d([b.motion_ids for b in batch], pad=motion_pad)
    text_mask = (text_ids != int(text_pad))
    motion_mask = (motion_ids != int(motion_pad))
    return {
        "text_ids": text_ids,
        "text_mask": text_mask,
        "motion_ids": motion_ids,
        "motion_mask": motion_mask,
    }


# -----------------------------
# Model
# -----------------------------


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class Text2MotionTokens(nn.Module):
    def __init__(
        self,
        text_vocab_size: int,
        motion_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        enc_layers: int = 4,
        dec_layers: int = 4,
        dropout: float = 0.1,
        max_text_len: int = 64,
        max_motion_len: int = 2048,
        text_pad_id: int = 0,
        motion_pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.text_pad_id = int(text_pad_id)
        self.motion_pad_id = int(motion_pad_id)

        self.text_emb = nn.Embedding(text_vocab_size, d_model, padding_idx=self.text_pad_id)
        self.motion_emb = nn.Embedding(motion_vocab_size, d_model, padding_idx=self.motion_pad_id)
        self.pos_text = SinusoidalPositionalEncoding(d_model, max_len=max_text_len)
        self.pos_motion = SinusoidalPositionalEncoding(d_model, max_len=max_motion_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        self.out_proj = nn.Linear(d_model, motion_vocab_size)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # True = masked
        m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return m

    def forward(
        self,
        text_ids: torch.Tensor,      # [B,S]
        text_mask: torch.Tensor,     # [B,S] True=valid
        motion_inp: torch.Tensor,    # [B,T]
        motion_mask: torch.Tensor,   # [B,T] True=valid
    ) -> torch.Tensor:
        # Encode text
        x = self.text_emb(text_ids)
        x = self.pos_text(x)
        mem = self.encoder(x, src_key_padding_mask=~text_mask)

        # Decode motion tokens (teacher forcing)
        y = self.motion_emb(motion_inp)
        y = self.pos_motion(y)
        T = y.size(1)
        causal = self._causal_mask(T, device=y.device)
        out = self.decoder(
            tgt=y,
            memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=~motion_mask,
            memory_key_padding_mask=~text_mask,
        )
        logits = self.out_proj(out)  # [B,T,V]
        return logits


# -----------------------------
# Train / eval
# -----------------------------


def _split_teacher_forcing(motion_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """motion_ids: [B,T] with BOS ... EOS ... PAD
    returns:
      inp: [B,T-1] (BOS..)
      tgt: [B,T-1] (..EOS)
    """
    inp = motion_ids[:, :-1].contiguous()
    tgt = motion_ids[:, 1:].contiguous()
    return inp, tgt


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device, motion_pad_id: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in dl:
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        motion_ids = batch["motion_ids"].to(device)
        motion_mask_full = batch["motion_mask"].to(device)

        motion_inp, motion_tgt = _split_teacher_forcing(motion_ids)
        motion_mask = motion_mask_full[:, :-1]

        logits = model(text_ids, text_mask, motion_inp, motion_mask)
        # logits: [B,T-1,V]
        V = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, V),
            motion_tgt.view(-1),
            ignore_index=int(motion_pad_id),
            reduction="sum",
        )
        total_loss += float(loss.item())

        valid = (motion_tgt != int(motion_pad_id))
        total_tokens += int(valid.sum().item())
        pred = logits.argmax(dim=-1)
        total_correct += int(((pred == motion_tgt) & valid).sum().item())

    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "acc": float("nan")}

    avg_loss = total_loss / total_tokens
    ppl = float(math.exp(min(20.0, avg_loss)))
    acc = total_correct / total_tokens
    return {"loss": float(avg_loss), "ppl": float(ppl), "acc": float(acc)}


def save_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": int(step),
            "meta": meta,
        },
        str(path),
    )


def load_ckpt(path: Path, model: nn.Module, opt: Optional[torch.optim.Optimizer] = None) -> Dict:
    sd = torch.load(str(path), map_location="cpu")
    model.load_state_dict(sd["model"], strict=True)
    if opt is not None and "opt" in sd:
        opt.load_state_dict(sd["opt"])
    return sd


def train_main(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))

    hml_root = Path(args.hml_root)
    token_root = Path(args.token_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_log_{run_id}.jsonl"
    config_path = save_dir / f"train_config_{run_id}.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "args": vars(args),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Motion token special ids
    motion_vocab_size = int(args.motion_vocab_size)
    motion_pad_id = motion_vocab_size
    motion_bos_id = motion_vocab_size + 1
    motion_eos_id = motion_vocab_size + 2
    motion_vocab_total = motion_vocab_size + 3

    # Build/load text vocab
    vocab_path = save_dir / "text_vocab.json"
    if args.hf_tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore

            hf_tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
            # We still need IDs for special tokens; map into a WordVocab-like wrapper.
            # For training, we only need encode() and pad_id.
            class _HFWrap:
                def __init__(self, tok):
                    self.tok = tok
                    self.pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else 0
                    # Use BOS/EOS if present; otherwise reuse special tokens.
                    self.bos_id = int(tok.bos_token_id) if tok.bos_token_id is not None else self.pad_id
                    self.eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else self.pad_id
                    self.unk_id = int(tok.unk_token_id) if tok.unk_token_id is not None else self.pad_id
                    self.stoi = {}
                    self.itos = []

                def encode(self, text: str, max_len: int) -> List[int]:
                    enc = self.tok(
                        text,
                        truncation=True,
                        max_length=max_len,
                        add_special_tokens=True,
                    )
                    return list(map(int, enc["input_ids"]))

            text_vocab = _HFWrap(hf_tok)  # type: ignore[assignment]
            text_vocab_size = int(hf_tok.vocab_size) + int(getattr(hf_tok, "added_tokens_encoder", {}).__len__())
        except Exception as e:
            raise RuntimeError(f"Failed to load hf_tokenizer={args.hf_tokenizer}: {e}")
    else:
        if vocab_path.is_file() and not args.rebuild_text_vocab:
            text_vocab = load_word_vocab(vocab_path)
        else:
            # Build from train captions
            text_dir = resolve_text_dir(hml_root)
            train_ids = read_split_ids(hml_root, str(args.split))
            caps: List[str] = []
            for mid in train_ids:
                caps.extend(read_hml_caption_file(text_dir / f"{mid}.txt"))
            text_vocab = build_word_vocab(caps, max_size=int(args.text_vocab_size), min_freq=int(args.text_min_freq))
            save_word_vocab(text_vocab, vocab_path)

        text_vocab_size = len(text_vocab.itos)

    # Datasets
    train_ds = HumanML3DTextToTokens(
        hml_root=hml_root,
        token_root=token_root,
        split=str(args.split),
        text_vocab=text_vocab,  # type: ignore[arg-type]
        max_text_len=int(args.max_text_len),
        min_motion_tokens=int(args.min_motion_tokens),
        max_motion_tokens=int(args.max_motion_tokens),
        pick_one_text=bool(args.pick_one_text),
        motion_pad_id=motion_pad_id,
        motion_bos_id=motion_bos_id,
        motion_eos_id=motion_eos_id,
    )
    val_ds = HumanML3DTextToTokens(
        hml_root=hml_root,
        token_root=token_root,
        split=str(args.val_split),
        text_vocab=text_vocab,  # type: ignore[arg-type]
        max_text_len=int(args.max_text_len),
        min_motion_tokens=int(args.min_motion_tokens),
        max_motion_tokens=int(args.max_motion_tokens),
        pick_one_text=False,
        motion_pad_id=motion_pad_id,
        motion_bos_id=motion_bos_id,
        motion_eos_id=motion_eos_id,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=lambda b: collate_t2m(b, text_pad=int(text_vocab.pad_id), motion_pad=motion_pad_id),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=lambda b: collate_t2m(b, text_pad=int(text_vocab.pad_id), motion_pad=motion_pad_id),
        drop_last=False,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = Text2MotionTokens(
        text_vocab_size=int(text_vocab_size),
        motion_vocab_size=int(motion_vocab_total),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        enc_layers=int(args.enc_layers),
        dec_layers=int(args.dec_layers),
        dropout=float(args.dropout),
        max_text_len=max(64, int(args.max_text_len) + 4),
        max_motion_len=max(2048, int(args.max_motion_tokens) + 4),
        text_pad_id=int(text_vocab.pad_id),
        motion_pad_id=motion_pad_id,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), betas=(0.9, 0.99), weight_decay=float(args.wd))

    start_step = 0
    best_val = float("inf")

    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        sd = load_ckpt(Path(args.resume), model, opt)
        start_step = int(sd.get("step", 0))
        best_val = float(sd.get("meta", {}).get("best_val", best_val))

    grad_accum = max(1, int(args.grad_accum))
    log_every = max(1, int(args.log_every))
    eval_every = max(1, int(args.eval_every))
    save_every = max(1, int(args.save_every))

    print("Num parameters:", sum(p.numel() for p in model.parameters()))

    model.train()
    step = start_step
    running_loss = 0.0
    running_tokens = 0

    for epoch in range(int(args.epochs)):
        for it, batch in enumerate(train_dl):
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            motion_ids = batch["motion_ids"].to(device)
            motion_mask_full = batch["motion_mask"].to(device)

            motion_inp, motion_tgt = _split_teacher_forcing(motion_ids)
            motion_mask = motion_mask_full[:, :-1]

            logits = model(text_ids, text_mask, motion_inp, motion_mask)
            V = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, V),
                motion_tgt.view(-1),
                ignore_index=int(motion_pad_id),
                reduction="sum",
            )
            valid = (motion_tgt != int(motion_pad_id))
            denom = valid.sum().clamp_min(1)
            loss = loss / denom
            loss = loss / float(grad_accum)
            loss.backward()

            running_loss += float(loss.item()) * float(grad_accum)
            running_tokens += int(denom.item())

            if (it + 1) % grad_accum == 0:
                if args.clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0:
                    avg = running_loss / max(1, log_every)
                    tok = max(1, running_tokens)
                    ppl = math.exp(min(20.0, avg))
                    train_log = {
                        "split": "train",
                        "run_id": run_id,
                        "epoch": int(epoch),
                        "step": int(step),
                        "loss": float(avg),
                        "ppl": float(ppl),
                        "tokens": int(tok),
                    }
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log, ensure_ascii=False) + "\n")
                    print(
                        f"[train] epoch={epoch} step={step} loss={avg:.6f} ppl~{ppl:.3f} tokens={tok}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_tokens = 0

                if step % eval_every == 0:
                    val = evaluate(model, val_dl, device=device, motion_pad_id=motion_pad_id)
                    val_log = {
                        "split": "val",
                        "run_id": run_id,
                        "epoch": int(epoch),
                        "step": int(step),
                        "loss": float(val["loss"]),
                        "ppl": float(val["ppl"]),
                        "acc": float(val["acc"]),
                    }
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(val_log, ensure_ascii=False) + "\n")
                    print(
                        f"[val] step={step} loss={val['loss']:.6f} ppl={val['ppl']:.3f} acc={val['acc']:.4f}",
                        flush=True,
                    )
                    model.train()

                    # Save best
                    if val["loss"] < best_val:
                        best_val = float(val["loss"])
                        save_ckpt(
                            ckpt_dir / "best.pt",
                            model,
                            opt,
                            step,
                            meta={
                                "best_val": best_val,
                                "epoch": epoch,
                                "step": step,
                                "run_id": run_id,
                                "args": vars(args),
                            },
                        )

                if step % save_every == 0:
                    save_ckpt(
                        ckpt_dir / "last.pt",
                        model,
                        opt,
                        step,
                        meta={
                            "best_val": best_val,
                            "epoch": epoch,
                            "step": step,
                            "run_id": run_id,
                            "args": vars(args),
                        },
                    )

    # Final save
    save_ckpt(
        ckpt_dir / "last.pt",
        model,
        opt,
        step,
        meta={
            "best_val": best_val,
            "epoch": int(args.epochs) - 1,
            "step": step,
            "run_id": run_id,
            "args": vars(args),
        },
    )
    print(f"Done. best_val={best_val:.6f} checkpoints={ckpt_dir}")


@torch.no_grad()
def generate_main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    sd = torch.load(str(args.ckpt), map_location="cpu")
    meta = sd.get("meta", {})
    train_args = meta.get("args", {})

    motion_vocab_size = int(train_args.get("motion_vocab_size", args.motion_vocab_size))
    motion_pad_id = motion_vocab_size
    motion_bos_id = motion_vocab_size + 1
    motion_eos_id = motion_vocab_size + 2
    motion_vocab_total = motion_vocab_size + 3

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.ckpt).parent.parent

    # Text tokenizer: prefer saved word vocab; if missing, fall back to hf_tokenizer recorded in ckpt.
    vocab_path = save_dir / "text_vocab.json"
    hf_name = str(train_args.get("hf_tokenizer", "") or "").strip()
    if vocab_path.is_file():
        text_vocab = load_word_vocab(vocab_path)
        text_vocab_size = len(text_vocab.itos)
        use_hf = False
    elif hf_name:
        try:
            from transformers import AutoTokenizer  # type: ignore

            hf_tok = AutoTokenizer.from_pretrained(hf_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load hf_tokenizer={hf_name}: {e}")

        class _HFWrap:
            def __init__(self, tok):
                self.tok = tok
                self.pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else 0
                self.bos_id = int(tok.bos_token_id) if tok.bos_token_id is not None else self.pad_id
                self.eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else self.pad_id
                self.unk_id = int(tok.unk_token_id) if tok.unk_token_id is not None else self.pad_id
                self.stoi = {}
                self.itos = []

            def encode(self, text: str, max_len: int) -> List[int]:
                enc = self.tok(
                    text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                )
                return list(map(int, enc["input_ids"]))

        text_vocab = _HFWrap(hf_tok)  # type: ignore[assignment]
        text_vocab_size = int(hf_tok.vocab_size) + int(getattr(hf_tok, "added_tokens_encoder", {}).__len__())
        use_hf = True
    else:
        raise FileNotFoundError(
            f"text vocab not found. Expected {vocab_path}, or store hf_tokenizer name in checkpoint meta."
        )

    model = Text2MotionTokens(
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
        text_pad_id=int(text_vocab.pad_id),
        motion_pad_id=motion_pad_id,
    ).to(device)
    model.load_state_dict(sd["model"], strict=True)
    model.eval()

    # Encode text
    text_ids = torch.tensor(
        [text_vocab.encode(args.text, max_len=int(train_args.get("max_text_len", 32)))],
        dtype=torch.long,
    ).to(device)
    text_mask = (text_ids != int(text_vocab.pad_id))

    # Encode text memory
    mem = model.encoder(model.pos_text(model.text_emb(text_ids)), src_key_padding_mask=~text_mask)

    max_len = int(args.max_len)
    ys: List[int] = [motion_bos_id]

    for _ in range(max_len):
        y = torch.tensor([ys], dtype=torch.long).to(device)
        y_mask = (y != motion_pad_id)
        y_emb = model.pos_motion(model.motion_emb(y))
        causal = model._causal_mask(y.size(1), device=device)
        dec = model.decoder(
            tgt=y_emb,
            memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=~y_mask,
            memory_key_padding_mask=~text_mask,
        )
        logits = model.out_proj(dec[:, -1])  # [B,V]
        next_id = int(logits.argmax(dim=-1).item())
        ys.append(next_id)
        if next_id == motion_eos_id:
            break

    # Strip BOS/EOS/PAD
    gen = ys
    if args.out_txt:
        Path(args.out_txt).write_text("\n".join(map(str, gen)), encoding="utf-8")
        print(f"Saved: {args.out_txt}")
    else:
        print("Generated token ids:")
        print(gen)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--hml_root", type=str, required=True, help="HumanML3D dataset root (contains train.txt, texts/, etc.)")
    tr.add_argument("--token_root", type=str, required=True, help="Root that contains <split>/<id>.txt token files")
    tr.add_argument("--save_dir", type=str, required=True)
    tr.add_argument("--split", type=str, default="train")
    tr.add_argument("--val_split", type=str, default="val")

    tr.add_argument("--motion_vocab_size", type=int, required=True, help="Vocab size used by the motion tokenizer (e.g., 1024)")
    tr.add_argument("--min_motion_tokens", type=int, default=1)
    tr.add_argument("--max_motion_tokens", type=int, default=2048)

    tr.add_argument("--max_text_len", type=int, default=32)
    # HumanML3D commonly samples one caption per motion during training.
    tr.add_argument("--pick_one_text", action="store_true", default=True,
                    help="(default: True) if set, pick a random caption line per motion each epoch")
    tr.add_argument("--no_pick_one_text", dest="pick_one_text", action="store_false",
                    help="Disable random caption sampling (use the first caption only)")

    tr.add_argument("--hf_tokenizer", type=str, default="", help="Optional: HuggingFace tokenizer name/path")
    tr.add_argument("--text_vocab_size", type=int, default=30000)
    tr.add_argument("--text_min_freq", type=int, default=2)
    tr.add_argument("--rebuild_text_vocab", action="store_true")

    # model size
    tr.add_argument("--d_model", type=int, default=512)
    tr.add_argument("--n_heads", type=int, default=8)
    tr.add_argument("--d_ff", type=int, default=2048)
    tr.add_argument("--enc_layers", type=int, default=4)
    tr.add_argument("--dec_layers", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.1)

    # train loop
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--eval_batch_size", type=int, default=64)
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--wd", type=float, default=0.0)
    tr.add_argument("--grad_accum", type=int, default=1)
    tr.add_argument("--clip_grad", type=float, default=1.0)
    tr.add_argument("--num_workers", type=int, default=4)

    tr.add_argument("--log_every", type=int, default=50)
    tr.add_argument("--eval_every", type=int, default=500)
    tr.add_argument("--save_every", type=int, default=500)
    tr.add_argument("--resume", type=str, default="")
    tr.add_argument("--seed", type=int, default=1234)
    tr.add_argument("--device", type=str, default="")

    # generate
    ge = sub.add_parser("generate")
    ge.add_argument("--ckpt", type=str, required=True)
    ge.add_argument("--text", type=str, required=True)
    ge.add_argument("--max_len", type=int, default=256)
    ge.add_argument("--out_txt", type=str, default="")
    ge.add_argument("--save_dir", type=str, default="", help="Where text_vocab.json lives (default: <ckpt>/../..)")
    ge.add_argument("--motion_vocab_size", type=int, default=1024, help="Only used if not stored in ckpt meta")
    ge.add_argument("--device", type=str, default="")

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "train":
        # Normalize hf_tokenizer empty string
        if isinstance(args.hf_tokenizer, str) and args.hf_tokenizer.strip() == "":
            args.hf_tokenizer = ""
        train_main(args)
    elif args.cmd == "generate":
        generate_main(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
