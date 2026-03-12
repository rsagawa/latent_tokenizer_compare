#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_retrieval_actionrec_id_contrib.py

Evaluate token-ID contributions for actionrec/retrieval by:
1) gradient-based direct attribution (Gradient*Input on token embeddings)
2) class-conditional statistics on top IDs/types:
   - class-conditional frequency
   - PMI
   - chi-square (one-vs-rest, 2x2)
   - mutual information (binary item-presence vs class)
3) sequence motif mining:
   - extract high-contribution contiguous ID subsequences per sample
   - validate motif importance with per-sample perturbation
   - summarize recurring motifs across samples
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_eval_actionrec_hml_tokens as ar
import train_eval_retrieval_hml_tokens as rt
from hml_tokens_data import HMLTokensDataset, build_word_pos_tensors, load_word_vectorizer


@dataclass
class Sample:
    mid: str
    token_ids: List[int]
    label: Optional[int]
    text_tokens: List[str]


class SampleDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        y = -1 if s.label is None else int(s.label)
        return s.token_ids, y, s.mid


class IdTypeMapper:
    def __init__(self, mode: str, bucket_size: int, modulo: int):
        self.mode = str(mode)
        self.bucket_size = max(1, int(bucket_size))
        self.modulo = max(1, int(modulo))

    def type_of(self, token_id: int) -> int:
        tid = int(token_id)
        if self.mode == "none":
            return 0
        if self.mode == "bucket":
            return tid // self.bucket_size
        if self.mode == "modulo":
            return tid % self.modulo
        raise ValueError(f"unsupported id_type_mode: {self.mode}")

    def type_name(self, t: int) -> str:
        if self.mode == "none":
            return "all"
        if self.mode == "bucket":
            lo = int(t) * self.bucket_size
            hi = (int(t) + 1) * self.bucket_size - 1
            return f"[{lo},{hi}]"
        if self.mode == "modulo":
            return f"id % {self.modulo} == {int(t)}"
        return str(int(t))


def collate_tokens_safe(
    seqs: Sequence[Sequence[int]],
    vocab_size: int,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = int(vocab_size)
    lengths = [min(len(s), int(max_len)) for s in seqs]
    t = max(lengths) if lengths else 0
    t = max(1, min(t, int(max_len)))
    x = torch.full((len(seqs), t), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), t), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ss = list(s[:t])
        if ss:
            x[i, : len(ss)] = torch.tensor(ss, dtype=torch.long)
            mask[i, : len(ss)] = True
        else:
            # Keep at least one valid token to avoid "all-masked" transformer failures.
            x[i, 0] = 0
            mask[i, 0] = True
    return x, mask, torch.tensor(lengths, dtype=torch.long)


def load_samples(
    hml_root: str,
    token_root: str,
    split: str,
    max_tokens: int,
) -> Tuple[List[Sample], List[Sample], str]:
    base_ds = HMLTokensDataset(hml_root, token_root, split, max_tokens=max_tokens, drop_if_missing=True)
    dataset_key = ar.infer_dataset_key(hml_root, token_root, list(base_ds.ids))
    ar.set_action_map(dataset_key)

    all_samples: List[Sample] = []
    labeled_samples: List[Sample] = []
    for i in range(len(base_ds)):
        tok, cap, toks, mid = base_ds[i]
        if not tok:
            continue
        y = ar.caption_to_label(cap)
        s = Sample(mid=mid, token_ids=list(tok), label=y, text_tokens=list(toks))
        all_samples.append(s)
        if y is not None:
            labeled_samples.append(s)
    return all_samples, labeled_samples, dataset_key


def _safe_log2(x: float) -> float:
    if x <= 0.0:
        return float("-inf")
    return float(math.log2(x))


def _chi2_2x2(a: float, b: float, c: float, d: float, eps: float = 1e-12) -> float:
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom <= eps:
        return 0.0
    num = n * ((a * d - b * c) ** 2)
    return float(num / (denom + eps))


def _binary_item_mi(item_present_by_class: np.ndarray, class_counts: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.sum(class_counts))
    if n <= eps:
        return 0.0
    present = item_present_by_class.astype(np.float64, copy=False)
    absent = class_counts.astype(np.float64, copy=False) - present
    n1 = float(np.sum(present))
    n0 = n - n1
    if n1 <= eps or n0 <= eps:
        return 0.0

    p1 = n1 / n
    p0 = n0 / n
    mi = 0.0
    for c in range(len(class_counts)):
        pc = float(class_counts[c]) / n
        if pc <= eps:
            continue
        p1c = float(present[c]) / n
        p0c = float(absent[c]) / n
        if p1c > eps:
            mi += p1c * math.log2(p1c / (p1 * pc))
        if p0c > eps:
            mi += p0c * math.log2(p0c / (p0 * pc))
    return float(mi)


def compute_item_stats(
    items_top: Sequence[int],
    item_occ: Dict[int, float],
    item_occ_by_class: Dict[int, np.ndarray],
    item_pres: Dict[int, int],
    item_pres_by_class: Dict[int, np.ndarray],
    class_counts: np.ndarray,
    class_token_totals: np.ndarray,
    class_names: Sequence[str],
    item_name_fn: Callable[[int], str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    n_samples = int(np.sum(class_counts))
    n_classes = len(class_names)

    overall_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []

    for it in items_top:
        occ = float(item_occ.get(it, 0.0))
        pres = int(item_pres.get(it, 0))
        pres_by_c = item_pres_by_class.get(it, np.zeros(n_classes, dtype=np.int64))
        occ_by_c = item_occ_by_class.get(it, np.zeros(n_classes, dtype=np.float64))

        p_item = (pres / n_samples) if n_samples > 0 else float("nan")
        mi = _binary_item_mi(pres_by_c.astype(np.float64), class_counts.astype(np.float64))

        overall_rows.append(
                {
                    "item_id": int(it),
                    "item_name": item_name_fn(int(it)),
                    "occurrences": occ,
                "samples_with_item": pres,
                "sample_presence_rate": p_item,
                "mi_item_vs_class_bits": mi,
            }
        )

        for c in range(n_classes):
            n_c = int(class_counts[c])
            a = float(pres_by_c[c])
            b = float(pres - pres_by_c[c])
            c_rest = float(n_c - pres_by_c[c])
            d = float(n_samples - a - b - c_rest)

            p_item_c = (a / n_samples) if n_samples > 0 else 0.0
            p_c = (n_c / n_samples) if n_samples > 0 else 0.0
            pmi = _safe_log2(p_item_c / (max(p_item, 1e-12) * max(p_c, 1e-12))) if a > 0 else float("-inf")
            chi2 = _chi2_2x2(a, b, c_rest, d)

            tok_freq_given_c = float(occ_by_c[c] / max(1.0, float(class_token_totals[c])))
            sample_freq_given_c = float(pres_by_c[c] / max(1.0, float(n_c)))

            class_rows.append(
                {
                    "item_id": int(it),
                    "item_name": item_name_fn(int(it)),
                    "class_id": int(c),
                    "class_name": str(class_names[c]),
                    "occurrences_in_class": float(occ_by_c[c]),
                    "samples_with_item_in_class": int(pres_by_c[c]),
                    "token_freq_given_class": tok_freq_given_c,
                    "sample_freq_given_class": sample_freq_given_c,
                    "pmi_bits": pmi,
                    "chi2_2x2": chi2,
                }
            )
    return overall_rows, class_rows


def collect_stats(
    labeled_samples: Sequence[Sample],
    class_names: Sequence[str],
    top_k_ids: int,
    top_k_types: int,
    type_mapper: IdTypeMapper,
    forced_ids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    n_classes = len(class_names)
    class_counts = np.zeros(n_classes, dtype=np.int64)
    class_token_totals = np.zeros(n_classes, dtype=np.float64)

    id_occ: Dict[int, float] = Counter()
    id_pres: Dict[int, int] = Counter()
    id_occ_by_class: Dict[int, np.ndarray] = {}
    id_pres_by_class: Dict[int, np.ndarray] = {}

    tp_occ: Dict[int, float] = Counter()
    tp_pres: Dict[int, int] = Counter()
    tp_occ_by_class: Dict[int, np.ndarray] = {}
    tp_pres_by_class: Dict[int, np.ndarray] = {}

    for s in labeled_samples:
        if s.label is None:
            continue
        c = int(s.label)
        class_counts[c] += 1
        class_token_totals[c] += 1.0

        c_id = Counter(int(t) for t in s.token_ids)
        c_tp = Counter(type_mapper.type_of(int(t)) for t in s.token_ids)
        tok_den = max(1.0, float(len(s.token_ids)))

        for tid, n in c_id.items():
            frac = float(n) / tok_den
            id_occ[tid] += frac
            id_pres[tid] += 1
            if tid not in id_occ_by_class:
                id_occ_by_class[tid] = np.zeros(n_classes, dtype=np.float64)
                id_pres_by_class[tid] = np.zeros(n_classes, dtype=np.int64)
            id_occ_by_class[tid][c] += frac
            id_pres_by_class[tid][c] += 1

        for tp, n in c_tp.items():
            frac = float(n) / tok_den
            tp_occ[tp] += frac
            tp_pres[tp] += 1
            if tp not in tp_occ_by_class:
                tp_occ_by_class[tp] = np.zeros(n_classes, dtype=np.float64)
                tp_pres_by_class[tp] = np.zeros(n_classes, dtype=np.int64)
            tp_occ_by_class[tp][c] += frac
            tp_pres_by_class[tp][c] += 1

    ranked_ids = [k for k, _ in sorted(id_occ.items(), key=lambda kv: (-kv[1], kv[0]))]
    top_ids = ranked_ids[: max(0, int(top_k_ids))]
    if forced_ids is not None:
        seen = set(int(x) for x in top_ids)
        for fid in forced_ids:
            ii = int(fid)
            if ii not in seen and ii in id_occ:
                top_ids.append(ii)
                seen.add(ii)
    top_tps = [k for k, _ in sorted(tp_occ.items(), key=lambda kv: (-kv[1], kv[0]))[: max(0, int(top_k_types))]]

    id_overall, id_class = compute_item_stats(
        top_ids,
        id_occ,
        id_occ_by_class,
        id_pres,
        id_pres_by_class,
        class_counts,
        class_token_totals,
        class_names,
        item_name_fn=lambda x: str(x),
    )
    tp_overall, tp_class = compute_item_stats(
        top_tps,
        tp_occ,
        tp_occ_by_class,
        tp_pres,
        tp_pres_by_class,
        class_counts,
        class_token_totals,
        class_names,
        item_name_fn=type_mapper.type_name,
    )

    return {
        "class_counts": class_counts.tolist(),
        "class_token_totals": class_token_totals.tolist(),
        "top_ids": top_ids,
        "top_types": top_tps,
        "id_overall_rows": id_overall,
        "id_class_rows": id_class,
        "type_overall_rows": tp_overall,
        "type_class_rows": tp_class,
    }


@torch.no_grad()
def eval_actionrec_samples(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    n_classes: int,
) -> Dict[str, float]:
    ds = SampleDataset(samples)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: _collate_action_safe(b, vocab_size=vocab_size, max_len=max_tokens),
        drop_last=False,
    )

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    true_probs: List[np.ndarray] = []
    removed_total = 0
    token_total = 0

    for x, mask, _lens, y, _mids in loader:
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)
        logits = model(x, mask)
        pred = torch.argmax(logits, dim=-1)
        prob = F.softmax(logits, dim=-1)
        tprob = prob.gather(1, y.unsqueeze(1)).squeeze(1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        true_probs.append(tprob.detach().cpu().numpy())
        token_total += int(mask.numel())
        removed_total += int((~mask).sum().item())

    if not ys:
        return {"acc": float("nan"), "macro_f1": float("nan"), "mean_true_prob": float("nan")}
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    mean_true_prob = float(np.mean(np.concatenate(true_probs, axis=0)))
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "macro_f1": ar.macro_f1(y_true, y_pred, n_classes),
        "mean_true_prob": mean_true_prob,
    }


def _collate_action_safe(batch, vocab_size: int, max_len: int):
    seqs = [b[0] for b in batch]
    mids = [b[2] for b in batch]
    x, mask, lengths = collate_tokens_safe(seqs, vocab_size=vocab_size, max_len=max_len)
    y = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return x, mask, lengths, y, mids


def load_actionrec_model(
    ckpt_path: Path,
    vocab_size: int,
    n_classes: int,
    max_tokens: int,
    device: torch.device,
) -> ar.ActionClassifier:
    ck = torch.load(str(ckpt_path), map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cargs = ck.get("args", {}) if isinstance(ck, dict) else {}

    model = ar.ActionClassifier(
        vocab_size=int(vocab_size),
        n_classes=int(n_classes),
        d_model=int(cargs.get("d_model", 256)),
        n_heads=int(cargs.get("n_heads", 8)),
        n_layers=int(cargs.get("n_layers", 4)),
        max_len=int(max_tokens),
        dropout=float(cargs.get("dropout", 0.1)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_text_embeddings(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    wvec = load_word_vectorizer(Path(motiongpt_root))

    word_np: List[np.ndarray] = []
    pos_np: List[np.ndarray] = []
    len_np: List[int] = []
    for s in samples:
        w, p, L = build_word_pos_tensors(wvec, s.text_tokens, max_text_len=max_text_len)
        word_np.append(w)
        pos_np.append(p)
        len_np.append(L)

    words = torch.from_numpy(np.stack(word_np, axis=0))
    poss = torch.from_numpy(np.stack(pos_np, axis=0))
    lens = torch.tensor(len_np, dtype=torch.long)

    embs: List[np.ndarray] = []
    for st in range(0, len(samples), batch_size):
        ed = min(len(samples), st + batch_size)
        w = words[st:ed].to(device)
        p = poss[st:ed].to(device)
        l = lens[st:ed].to(device)
        t = model.text_enc(w, p, l)
        embs.append(t.detach().cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)


@torch.no_grad()
def encode_motion_embeddings(
    model: rt.DualEncoder,
    token_lists: Sequence[Sequence[int]],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    out: List[np.ndarray] = []
    for st in range(0, len(token_lists), batch_size):
        ed = min(len(token_lists), st + batch_size)
        chunk = token_lists[st:ed]
        x, m, _l = collate_tokens_safe(chunk, vocab_size=vocab_size, max_len=max_tokens)
        x = x.to(device)
        m = m.to(device)
        z = model.motion_enc(x, m)
        out.append(z.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def load_retrieval_model(
    ckpt_path: Path,
    vocab_size: int,
    max_tokens: int,
    device: torch.device,
) -> Tuple[rt.DualEncoder, Dict[str, object]]:
    ck = torch.load(str(ckpt_path), map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cargs = ck.get("args", {}) if isinstance(ck, dict) else {}

    model = rt.DualEncoder(
        rt.MotionTokenEncoder(
            vocab_size=int(vocab_size),
            d_model=int(cargs.get("d_model", 256)),
            n_heads=int(cargs.get("n_heads", 8)),
            n_layers=int(cargs.get("n_layers", 4)),
            max_len=int(max_tokens),
            dropout=float(cargs.get("dropout", 0.1)),
        ),
        rt.TextEncoder(
            d_model=int(cargs.get("d_model", 256)),
            dropout=float(cargs.get("dropout", 0.1)),
        ),
        temperature=float(cargs.get("temperature", 0.07)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cargs


def evaluate_retrieval(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    cached_text_emb: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    if cached_text_emb is None:
        emb_t = encode_text_embeddings(model, samples, motiongpt_root, max_text_len, batch_size, device)
    else:
        emb_t = cached_text_emb
    motion_lists = [s.token_ids for s in samples]
    emb_m = encode_motion_embeddings(model, motion_lists, vocab_size, max_tokens, batch_size, device)

    n = min(len(emb_t), len(emb_m))
    emb_t = emb_t[:n]
    emb_m = emb_m[:n]
    t2m = rt.retrieval_metrics(emb_t, emb_m) if n > 0 else {}
    m2t = rt.retrieval_metrics(emb_m, emb_t) if n > 0 else {}
    return {"n_used": int(n), "t2m": t2m, "m2t": m2t}


def _aggregate_occurrence_stats_from_batch(
    x: torch.Tensor,
    mask: torch.Tensor,
    vocab_size: int,
) -> Tuple[Counter, Counter]:
    occ: Counter = Counter()
    sample_with: Counter = Counter()
    x_cpu = x.detach().cpu()
    mask_cpu = mask.detach().cpu()
    for i in range(x_cpu.size(0)):
        seen = set()
        for j in range(x_cpu.size(1)):
            if not bool(mask_cpu[i, j]):
                continue
            tid = int(x_cpu[i, j].item())
            if tid < 0 or tid >= int(vocab_size):
                continue
            occ[tid] += 1
            seen.add(tid)
        for tid in seen:
            sample_with[tid] += 1
    return occ, sample_with


def _accumulate_sample_normalized_attr(
    x: torch.Tensor,
    mask: torch.Tensor,
    vocab_size: int,
    value_of_id: Dict[int, float],
    signed: Counter,
    per_sample_rows: List[Dict[str, object]],
    batch_samples: Sequence[Sample],
    class_names: Sequence[str],
) -> None:
    x_cpu = x.detach().cpu()
    mask_cpu = mask.detach().cpu()
    for i in range(x_cpu.size(0)):
        tids = set()
        for j in range(x_cpu.size(1)):
            if not bool(mask_cpu[i, j]):
                continue
            tid = int(x_cpu[i, j].item())
            if tid < 0 or tid >= int(vocab_size):
                continue
            tids.add(tid)
        if not tids:
            continue
        vals = {tid: float(value_of_id.get(tid, 0.0)) for tid in tids}
        denom = sum(abs(v) for v in vals.values())
        if denom <= 1e-12:
            continue
        for tid, v in vals.items():
            vn = v / denom
            signed[tid] += vn
            label = batch_samples[i].label
            class_name = str(class_names[int(label)]) if label is not None and 0 <= int(label) < len(class_names) else ""
            verb = ""
            adj = ""
            if "__" in class_name:
                verb, adj = class_name.split("__", 1)
            per_sample_rows.append(
                {
                    "mid": str(batch_samples[i].mid),
                    "class_name": class_name,
                    "verb": verb,
                    "adjective": adj,
                    "item_id": int(tid),
                    "item_name": str(int(tid)),
                    "signed_attr_norm": float(vn),
                    "abs_attr_norm": float(abs(vn)),
                }
            )


def _build_attr_rows(
    signed: Counter,
    occ: Counter,
    sample_with: Counter,
    top_k: int,
) -> List[Dict[str, object]]:
    tids = sorted(signed.keys(), key=lambda t: (-abs(float(signed[t])), int(t)))
    if int(top_k) > 0:
        tids = tids[: int(top_k)]
    rows: List[Dict[str, object]] = []
    for rank, tid in enumerate(tids, start=1):
        o = int(occ.get(tid, 0))
        sw = int(sample_with.get(tid, 0))
        s = float(signed.get(tid, 0.0))
        rows.append(
            {
                "rank": rank,
                "item_id": int(tid),
                "item_name": str(int(tid)),
                "signed_attr_sum": s,
                "occurrences": o,
                "samples_with_item": sw,
            }
        )
    return rows


def compute_actionrec_id_attribution(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    score_target: str,
    top_k: int,
    class_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if score_target not in {"true_logit", "neg_ce"}:
        raise ValueError(f"unsupported actionrec attr score_target: {score_target}")
    signed: Counter = Counter()
    occ: Counter = Counter()
    sample_with: Counter = Counter()
    per_sample_rows: List[Dict[str, object]] = []

    for st in range(0, len(samples), batch_size):
        batch = samples[st : st + batch_size]
        seqs = [s.token_ids for s in batch]
        ys = torch.tensor([int(s.label) for s in batch], dtype=torch.long, device=device)
        x, mask, _lens = collate_tokens_safe(seqs, vocab_size=vocab_size, max_len=max_tokens)
        x = x.to(device)
        mask = mask.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(x, mask)
        if score_target == "true_logit":
            score = logits.gather(1, ys.unsqueeze(1)).squeeze(1).mean()
        else:
            score = -F.cross_entropy(logits, ys)

        grad_w = torch.autograd.grad(score, model.emb.weight, retain_graph=False, create_graph=False)[0]
        emb_w = model.emb.weight.detach()

        o, s = _aggregate_occurrence_stats_from_batch(x, mask, vocab_size=vocab_size)
        occ.update(o)
        sample_with.update(s)

        value_of_id: Dict[int, float] = {}
        for tid in set(o.keys()):
            value_of_id[tid] = float((grad_w[tid] * emb_w[tid]).sum().detach().cpu().item())
        _accumulate_sample_normalized_attr(
            x=x,
            mask=mask,
            vocab_size=vocab_size,
            value_of_id=value_of_id,
            signed=signed,
            per_sample_rows=per_sample_rows,
            batch_samples=batch,
            class_names=class_names,
        )

    return _build_attr_rows(signed, occ, sample_with, top_k=top_k), per_sample_rows


def compute_retrieval_id_attribution(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    score_target: str,
    top_k: int,
    class_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if score_target not in {"diag_cosine", "neg_nce"}:
        raise ValueError(f"unsupported retrieval attr score_target: {score_target}")

    wvec = load_word_vectorizer(Path(motiongpt_root))
    signed: Counter = Counter()
    occ: Counter = Counter()
    sample_with: Counter = Counter()
    per_sample_rows: List[Dict[str, object]] = []

    for st in range(0, len(samples), batch_size):
        batch = samples[st : st + batch_size]
        seqs = [s.token_ids for s in batch]
        x, mask, _lens = collate_tokens_safe(seqs, vocab_size=vocab_size, max_len=max_tokens)
        x = x.to(device)
        mask = mask.to(device)

        word_np: List[np.ndarray] = []
        pos_np: List[np.ndarray] = []
        len_np: List[int] = []
        for s in batch:
            w, p, L = build_word_pos_tensors(wvec, s.text_tokens, max_text_len=max_text_len)
            word_np.append(w)
            pos_np.append(p)
            len_np.append(L)
        word = torch.from_numpy(np.stack(word_np, axis=0)).to(device)
        pos = torch.from_numpy(np.stack(pos_np, axis=0)).to(device)
        tlen = torch.tensor(len_np, dtype=torch.long, device=device)

        model.zero_grad(set_to_none=True)
        m = model.motion_enc(x, mask)
        t = model.text_enc(word, pos, tlen)
        if score_target == "diag_cosine":
            score = (t * m).sum(dim=-1).mean()
        else:
            scale = model.logit_scale.exp().clamp(1e-3, 100.0)
            logits = scale * (t @ m.t())
            score = -rt.info_nce_loss(logits)

        grad_w = torch.autograd.grad(score, model.motion_enc.emb.weight, retain_graph=False, create_graph=False)[0]
        emb_w = model.motion_enc.emb.weight.detach()

        o, s = _aggregate_occurrence_stats_from_batch(x, mask, vocab_size=vocab_size)
        occ.update(o)
        sample_with.update(s)
        value_of_id: Dict[int, float] = {}
        for tid in set(o.keys()):
            value_of_id[tid] = float((grad_w[tid] * emb_w[tid]).sum().detach().cpu().item())
        _accumulate_sample_normalized_attr(
            x=x,
            mask=mask,
            vocab_size=vocab_size,
            value_of_id=value_of_id,
            signed=signed,
            per_sample_rows=per_sample_rows,
            batch_samples=batch,
            class_names=class_names,
        )

    return _build_attr_rows(signed, occ, sample_with, top_k=top_k), per_sample_rows


def _parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for part in str(spec).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _action_primary_metric(metrics: Dict[str, float]) -> float:
    acc = float(metrics.get("acc", float("nan")))
    macro_f1 = float(metrics.get("macro_f1", float("nan")))
    if math.isnan(acc) and math.isnan(macro_f1):
        return float("nan")
    if math.isnan(acc):
        return macro_f1
    if math.isnan(macro_f1):
        return acc
    return 0.5 * (acc + macro_f1)


def _retrieval_primary_metric(metrics: Dict[str, object]) -> float:
    t2m = metrics.get("t2m", {}) if isinstance(metrics.get("t2m", {}), dict) else {}
    m2t = metrics.get("m2t", {}) if isinstance(metrics.get("m2t", {}), dict) else {}
    t2m_r1 = float(t2m.get("R@1", float("nan")))
    m2t_r1 = float(m2t.get("R@1", float("nan")))
    if math.isnan(t2m_r1) and math.isnan(m2t_r1):
        return float("nan")
    if math.isnan(t2m_r1):
        return m2t_r1
    if math.isnan(m2t_r1):
        return t2m_r1
    return 0.5 * (t2m_r1 + m2t_r1)


def _safe_ratio(num: float, den: float) -> float:
    if abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num / den)


def _summarize_action_delta(base_metrics: Dict[str, float], pert_metrics: Dict[str, float]) -> Dict[str, float]:
    base_primary = _action_primary_metric(base_metrics)
    pert_primary = _action_primary_metric(pert_metrics)
    return {
        "base_acc": float(base_metrics.get("acc", float("nan"))),
        "perturbed_acc": float(pert_metrics.get("acc", float("nan"))),
        "acc_drop": float(base_metrics.get("acc", float("nan")) - pert_metrics.get("acc", float("nan"))),
        "base_macro_f1": float(base_metrics.get("macro_f1", float("nan"))),
        "perturbed_macro_f1": float(pert_metrics.get("macro_f1", float("nan"))),
        "macro_f1_drop": float(base_metrics.get("macro_f1", float("nan")) - pert_metrics.get("macro_f1", float("nan"))),
        "base_mean_true_prob": float(base_metrics.get("mean_true_prob", float("nan"))),
        "perturbed_mean_true_prob": float(pert_metrics.get("mean_true_prob", float("nan"))),
        "mean_true_prob_drop": float(base_metrics.get("mean_true_prob", float("nan")) - pert_metrics.get("mean_true_prob", float("nan"))),
        "base_primary_metric": base_primary,
        "perturbed_primary_metric": pert_primary,
        "primary_metric_drop": float(base_primary - pert_primary),
        "primary_metric_ratio": _safe_ratio(pert_primary, base_primary),
        "primary_metric_name": "mean(acc,macro_f1)",
    }


def _summarize_retrieval_delta(base_metrics: Dict[str, object], pert_metrics: Dict[str, object]) -> Dict[str, float]:
    base_t2m = base_metrics.get("t2m", {}) if isinstance(base_metrics.get("t2m", {}), dict) else {}
    base_m2t = base_metrics.get("m2t", {}) if isinstance(base_metrics.get("m2t", {}), dict) else {}
    pert_t2m = pert_metrics.get("t2m", {}) if isinstance(pert_metrics.get("t2m", {}), dict) else {}
    pert_m2t = pert_metrics.get("m2t", {}) if isinstance(pert_metrics.get("m2t", {}), dict) else {}
    base_primary = _retrieval_primary_metric(base_metrics)
    pert_primary = _retrieval_primary_metric(pert_metrics)
    base_t2m_r1 = float(base_t2m.get("R@1", float("nan")))
    base_m2t_r1 = float(base_m2t.get("R@1", float("nan")))
    pert_t2m_r1 = float(pert_t2m.get("R@1", float("nan")))
    pert_m2t_r1 = float(pert_m2t.get("R@1", float("nan")))
    return {
        "base_t2m_R@1": base_t2m_r1,
        "perturbed_t2m_R@1": pert_t2m_r1,
        "t2m_R@1_drop": float(base_t2m_r1 - pert_t2m_r1),
        "base_m2t_R@1": base_m2t_r1,
        "perturbed_m2t_R@1": pert_m2t_r1,
        "m2t_R@1_drop": float(base_m2t_r1 - pert_m2t_r1),
        "base_mean_R@1": base_primary,
        "perturbed_mean_R@1": pert_primary,
        "mean_R@1_drop": float(base_primary - pert_primary),
        "base_t2m_R@5": float(base_t2m.get("R@5", float("nan"))),
        "perturbed_t2m_R@5": float(pert_t2m.get("R@5", float("nan"))),
        "base_m2t_R@5": float(base_m2t.get("R@5", float("nan"))),
        "perturbed_m2t_R@5": float(pert_m2t.get("R@5", float("nan"))),
        "base_primary_metric": base_primary,
        "perturbed_primary_metric": pert_primary,
        "primary_metric_drop": float(base_primary - pert_primary),
        "primary_metric_ratio": _safe_ratio(pert_primary, base_primary),
        "primary_metric_name": "mean(t2m_R@1,m2t_R@1)",
    }


def _rank_ids_from_attr_rows(attr_rows: Sequence[Dict[str, object]], top_k: int) -> List[int]:
    ranked = sorted(
        attr_rows,
        key=lambda r: (-abs(float(r.get("signed_attr_sum", 0.0))), int(r.get("item_id", 0))),
    )
    ids = [int(r.get("item_id", 0)) for r in ranked]
    if int(top_k) > 0:
        ids = ids[: int(top_k)]
    return ids


def _build_mid_to_ranked_attr(per_sample_rows: Sequence[Dict[str, object]]) -> Dict[str, List[Tuple[int, float]]]:
    by_mid: Dict[str, Dict[int, float]] = {}
    for r in per_sample_rows:
        mid = str(r.get("mid", ""))
        if not mid:
            continue
        tid = int(r.get("item_id", 0))
        w = float(r.get("abs_attr_norm", abs(float(r.get("signed_attr_norm", 0.0)))))
        if w <= 0.0:
            continue
        by_mid.setdefault(mid, {})
        by_mid[mid][tid] = by_mid[mid].get(tid, 0.0) + w
    out: Dict[str, List[Tuple[int, float]]] = {}
    for mid, mp in by_mid.items():
        out[mid] = sorted(((int(tid), float(w)) for tid, w in mp.items() if w > 0.0), key=lambda kv: (-kv[1], kv[0]))
    return out


def _select_ids_by_top_p(ranked: Sequence[Tuple[int, float]], p: float, keep: bool) -> Set[int]:
    if not ranked:
        return set()
    pp = min(1.0, max(0.0, float(p)))
    total = sum(float(w) for _tid, w in ranked)
    if total <= 1e-12:
        return set(int(tid) for tid, _w in ranked) if keep else set()
    if keep and pp >= 1.0:
        return set(int(tid) for tid, _w in ranked)
    if (not keep) and pp <= 0.0:
        return set()
    selected: Set[int] = set()
    cum = 0.0
    for tid, w in ranked:
        selected.add(int(tid))
        cum += float(w)
        if cum / total >= pp - 1e-12:
            break
    if keep:
        return selected
    return selected


def _select_ids_by_top_k(ranked: Sequence[Tuple[int, float]], k: int, keep: bool) -> Set[int]:
    if not ranked:
        return set()
    kk = max(0, int(k))
    if keep and kk <= 0:
        return set()
    if keep and kk >= len(ranked):
        return set(int(tid) for tid, _w in ranked)
    picked = set(int(tid) for tid, _w in ranked[:kk])
    return picked


def _perturb_samples_by_sample_attr(
    samples: Sequence[Sample],
    sample_attr_ranked: Dict[str, List[Tuple[int, float]]],
    mode: str,
    budget_value: float,
) -> Tuple[List[Sample], Dict[str, float]]:
    out: List[Sample] = []
    total_tokens = 0
    removed_tokens = 0
    kept_tokens = 0
    affected_samples = 0
    fallback_kept_original = 0
    selected_id_counts: List[int] = []

    for s in samples:
        ranked = sample_attr_ranked.get(str(s.mid), [])
        original = list(s.token_ids)
        total_tokens += len(original)

        if mode == "keep_top_p_mass":
            keep_ids = _select_ids_by_top_p(ranked, float(budget_value), keep=True) if ranked else set(int(t) for t in original)
            new_tokens = [int(t) for t in original if int(t) in keep_ids]
            selected_id_counts.append(len(keep_ids))
        elif mode == "drop_top_p_mass":
            drop_ids = _select_ids_by_top_p(ranked, float(budget_value), keep=False)
            new_tokens = [int(t) for t in original if int(t) not in drop_ids]
            selected_id_counts.append(len(drop_ids))
        elif mode == "keep_top_k_ids":
            keep_ids = _select_ids_by_top_k(ranked, int(round(float(budget_value))), keep=True) if ranked else set(int(t) for t in original)
            new_tokens = [int(t) for t in original if int(t) in keep_ids]
            selected_id_counts.append(len(keep_ids))
        elif mode == "drop_top_k_ids":
            drop_ids = _select_ids_by_top_k(ranked, int(round(float(budget_value))), keep=False)
            new_tokens = [int(t) for t in original if int(t) not in drop_ids]
            selected_id_counts.append(len(drop_ids))
        else:
            raise ValueError(f"unsupported perturbation mode: {mode}")

        if len(new_tokens) != len(original):
            affected_samples += 1
        if not new_tokens:
            new_tokens = list(original[:1]) if original else [0]
            fallback_kept_original += 1
        removed_tokens += max(0, len(original) - len(new_tokens))
        kept_tokens += len(new_tokens)
        out.append(Sample(mid=s.mid, token_ids=new_tokens, label=s.label, text_tokens=list(s.text_tokens)))

    n_samples = len(samples)
    return out, {
        "n_samples": n_samples,
        "n_affected_samples": affected_samples,
        "affected_sample_ratio": _safe_ratio(float(affected_samples), float(max(1, n_samples))),
        "total_tokens_before": total_tokens,
        "total_tokens_after": kept_tokens,
        "removed_token_ratio": _safe_ratio(float(removed_tokens), float(max(1, total_tokens))),
        "mean_selected_id_count": float(np.mean(selected_id_counts)) if selected_id_counts else float("nan"),
        "n_fallback_kept_original": fallback_kept_original,
    }


def _perturb_samples_by_global_id_drop(
    samples: Sequence[Sample],
    drop_id: int,
) -> Tuple[List[Sample], Dict[str, float]]:
    out: List[Sample] = []
    total_tokens = 0
    removed_tokens = 0
    affected_samples = 0
    fallback_kept_original = 0
    samples_with_id = 0

    for s in samples:
        original = list(s.token_ids)
        total_tokens += len(original)
        has_id = any(int(t) == int(drop_id) for t in original)
        if has_id:
            samples_with_id += 1
        new_tokens = [int(t) for t in original if int(t) != int(drop_id)]
        if len(new_tokens) != len(original):
            affected_samples += 1
        if not new_tokens:
            new_tokens = list(original[:1]) if original else [0]
            fallback_kept_original += 1
        removed_tokens += max(0, len(original) - len(new_tokens))
        out.append(Sample(mid=s.mid, token_ids=new_tokens, label=s.label, text_tokens=list(s.text_tokens)))

    return out, {
        "n_samples": len(samples),
        "samples_with_item": samples_with_id,
        "affected_sample_ratio": _safe_ratio(float(affected_samples), float(max(1, len(samples)))),
        "removed_token_ratio": _safe_ratio(float(removed_tokens), float(max(1, total_tokens))),
        "n_fallback_kept_original": fallback_kept_original,
    }


def _perturb_samples_by_global_motif_drop(
    samples: Sequence[Sample],
    motif: Sequence[int],
    drop_all_occurrences: bool,
) -> Tuple[List[Sample], Dict[str, float]]:
    out: List[Sample] = []
    total_tokens = 0
    removed_tokens = 0
    affected_samples = 0
    fallback_kept_original = 0
    samples_with_motif = 0
    total_matched_occurrences = 0
    motif_list = [int(x) for x in motif]

    for s in samples:
        original = list(s.token_ids)
        total_tokens += len(original)
        occ_pos = _find_subsequence_occurrences(original, motif_list)
        if occ_pos:
            samples_with_motif += 1
        new_tokens, n_removed_occ = _drop_motif_occurrences(
            original,
            motif_list,
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        total_matched_occurrences += int(n_removed_occ)
        if len(new_tokens) != len(original):
            affected_samples += 1
        if not new_tokens:
            new_tokens = list(original[:1]) if original else [0]
            fallback_kept_original += 1
        removed_tokens += max(0, len(original) - len(new_tokens))
        out.append(Sample(mid=s.mid, token_ids=new_tokens, label=s.label, text_tokens=list(s.text_tokens)))

    return out, {
        "n_samples": len(samples),
        "motif_len": len(motif_list),
        "motif_key": _motif_to_key(motif_list),
        "samples_with_motif": samples_with_motif,
        "total_matched_occurrences": total_matched_occurrences,
        "affected_sample_ratio": _safe_ratio(float(affected_samples), float(max(1, len(samples)))),
        "removed_token_ratio": _safe_ratio(float(removed_tokens), float(max(1, total_tokens))),
        "n_fallback_kept_original": fallback_kept_original,
    }


def _drop_multiple_motif_occurrences(
    token_ids: Sequence[int],
    motifs: Sequence[Sequence[int]],
    drop_all_occurrences: bool,
) -> Tuple[List[int], int]:
    seq = [int(x) for x in token_ids]
    total_removed_occ = 0
    ordered = sorted(
        ([int(t) for t in m] for m in motifs if len(m) > 0),
        key=lambda m: (-len(m), tuple(m)),
    )
    for motif in ordered:
        seq, n_removed = _drop_motif_occurrences(seq, motif, drop_all_occurrences=drop_all_occurrences)
        total_removed_occ += int(n_removed)
    return seq, total_removed_occ


def _perturb_samples_by_global_motif_set_drop(
    samples: Sequence[Sample],
    motifs: Sequence[Sequence[int]],
    drop_all_occurrences: bool,
) -> Tuple[List[Sample], Dict[str, float]]:
    out: List[Sample] = []
    total_tokens = 0
    removed_tokens = 0
    affected_samples = 0
    fallback_kept_original = 0
    samples_with_any_motif = 0
    total_matched_occurrences = 0
    motif_list = [[int(t) for t in m] for m in motifs if len(m) > 0]

    for s in samples:
        original = list(s.token_ids)
        total_tokens += len(original)
        has_any = False
        for m in motif_list:
            if _find_subsequence_occurrences(original, m):
                has_any = True
                break
        if has_any:
            samples_with_any_motif += 1

        new_tokens, n_removed_occ = _drop_multiple_motif_occurrences(
            original,
            motif_list,
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        total_matched_occurrences += int(n_removed_occ)
        if len(new_tokens) != len(original):
            affected_samples += 1
        if not new_tokens:
            new_tokens = list(original[:1]) if original else [0]
            fallback_kept_original += 1
        removed_tokens += max(0, len(original) - len(new_tokens))
        out.append(Sample(mid=s.mid, token_ids=new_tokens, label=s.label, text_tokens=list(s.text_tokens)))

    return out, {
        "n_samples": len(samples),
        "n_motifs_in_set": len(motif_list),
        "samples_with_any_motif": samples_with_any_motif,
        "total_matched_occurrences": total_matched_occurrences,
        "affected_sample_ratio": _safe_ratio(float(affected_samples), float(max(1, len(samples)))),
        "removed_token_ratio": _safe_ratio(float(removed_tokens), float(max(1, total_tokens))),
        "n_fallback_kept_original": fallback_kept_original,
    }


def _find_motif_occurrences_spans(token_ids: Sequence[int], motif: Sequence[int]) -> List[Tuple[int, int]]:
    pos = _find_subsequence_occurrences(token_ids, motif)
    m = len(motif)
    return [(int(st), int(st + m)) for st in pos]


def _find_ordered_motif_chain_occurrences(
    token_ids: Sequence[int],
    motifs: Sequence[Sequence[int]],
    min_gap: int,
    max_gap: int,
    max_matches: int = 1000,
) -> List[List[Tuple[int, int]]]:
    motif_spans = [_find_motif_occurrences_spans(token_ids, m) for m in motifs]
    if not motif_spans or any(len(sps) == 0 for sps in motif_spans):
        return []
    lo = max(0, int(min_gap))
    hi = max(lo, int(max_gap))
    out: List[List[Tuple[int, int]]] = []

    def _dfs(level: int, prev_span: Optional[Tuple[int, int]], path: List[Tuple[int, int]]) -> None:
        if len(out) >= int(max_matches):
            return
        if level >= len(motif_spans):
            out.append(list(path))
            return
        for sp in motif_spans[level]:
            st, ed = sp
            if prev_span is not None:
                p_st, p_ed = prev_span
                gap = int(st - p_ed)
                if st < p_ed:
                    continue
                if gap < lo or gap > hi:
                    continue
                if st < p_st:
                    continue
            path.append((int(st), int(ed)))
            _dfs(level + 1, (int(st), int(ed)), path)
            path.pop()

    _dfs(0, None, [])
    return out


def _perturb_samples_by_global_motif_chain_drop(
    samples: Sequence[Sample],
    motifs: Sequence[Sequence[int]],
    min_gap: int,
    max_gap: int,
    drop_all_occurrences: bool,
) -> Tuple[List[Sample], Dict[str, float]]:
    out: List[Sample] = []
    total_tokens = 0
    removed_tokens = 0
    affected_samples = 0
    fallback_kept_original = 0
    samples_with_chain = 0
    total_matched_chains = 0
    chain_motifs = [[int(x) for x in m] for m in motifs if len(m) > 0]
    chain_keys = [_motif_to_key(m) for m in chain_motifs]

    for s in samples:
        original = list(s.token_ids)
        total_tokens += len(original)
        chains = _find_ordered_motif_chain_occurrences(
            original,
            chain_motifs,
            min_gap=int(min_gap),
            max_gap=int(max_gap),
        )
        if chains:
            samples_with_chain += 1
        if not chains:
            new_tokens = list(original)
        else:
            chains_to_use = chains if bool(drop_all_occurrences) else chains[:1]
            total_matched_chains += len(chains_to_use)
            drop_idx: Set[int] = set()
            for chain in chains_to_use:
                for st, ed in chain:
                    for i in range(st, ed):
                        drop_idx.add(int(i))
            new_tokens = [int(t) for i, t in enumerate(original) if i not in drop_idx]

        if len(new_tokens) != len(original):
            affected_samples += 1
        if not new_tokens:
            new_tokens = list(original[:1]) if original else [0]
            fallback_kept_original += 1
        removed_tokens += max(0, len(original) - len(new_tokens))
        out.append(Sample(mid=s.mid, token_ids=new_tokens, label=s.label, text_tokens=list(s.text_tokens)))

    return out, {
        "n_samples": len(samples),
        "chain_size": len(chain_motifs),
        "chain_keys": " -> ".join(chain_keys),
        "chain_min_gap": int(min_gap),
        "chain_max_gap": int(max_gap),
        "samples_with_chain": samples_with_chain,
        "total_matched_chains": total_matched_chains,
        "affected_sample_ratio": _safe_ratio(float(affected_samples), float(max(1, len(samples)))),
        "removed_token_ratio": _safe_ratio(float(removed_tokens), float(max(1, total_tokens))),
        "n_fallback_kept_original": fallback_kept_original,
    }


def compute_attr_coverage_rows(
    per_sample_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_mid: Dict[str, Dict[str, object]] = {}
    for r in per_sample_rows:
        mid = str(r.get("mid", ""))
        if not mid:
            continue
        rec = by_mid.setdefault(mid, {
            "class_name": str(r.get("class_name", "")),
            "verb": str(r.get("verb", "")),
            "adjective": str(r.get("adjective", "")),
            "weights": {},
        })
        tid = int(r.get("item_id", 0))
        w = float(r.get("abs_attr_norm", abs(float(r.get("signed_attr_norm", 0.0)))))
        if w > 0.0:
            rec["weights"][tid] = float(rec["weights"].get(tid, 0.0)) + w

    per_sample_out: List[Dict[str, object]] = []
    n_active_list: List[float] = []
    eff_list: List[float] = []
    ent_list: List[float] = []
    ent_n_list: List[float] = []
    top1_list: List[float] = []
    top3_list: List[float] = []
    top5_list: List[float] = []
    top10_list: List[float] = []

    for mid, rec in by_mid.items():
        wmap = {int(k): float(v) for k, v in rec["weights"].items() if float(v) > 0.0}
        vals = sorted(wmap.values(), reverse=True)
        if not vals:
            continue
        total = sum(vals)
        ps = [float(v) / total for v in vals if v > 0.0]
        h = 0.0
        for p in ps:
            if p > 0.0:
                h -= p * math.log(p)
        h_norm = 0.0
        if len(ps) > 1:
            h_norm = h / max(math.log(len(ps)), 1e-12)
        n_eff = float(math.exp(h))
        row = {
            "mid": mid,
            "class_name": rec["class_name"],
            "verb": rec["verb"],
            "adjective": rec["adjective"],
            "n_active_ids": len(ps),
            "entropy": h,
            "normalized_entropy": h_norm,
            "effective_num_ids": n_eff,
            "top1_share": sum(ps[:1]),
            "top3_share": sum(ps[:3]),
            "top5_share": sum(ps[:5]),
            "top10_share": sum(ps[:10]),
        }
        per_sample_out.append(row)
        n_active_list.append(float(row["n_active_ids"]))
        eff_list.append(float(row["effective_num_ids"]))
        ent_list.append(float(row["entropy"]))
        ent_n_list.append(float(row["normalized_entropy"]))
        top1_list.append(float(row["top1_share"]))
        top3_list.append(float(row["top3_share"]))
        top5_list.append(float(row["top5_share"]))
        top10_list.append(float(row["top10_share"]))

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    def _median(xs: List[float]) -> float:
        return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else float("nan")

    summary_rows = [{
        "n_samples": len(per_sample_out),
        "mean_n_active_ids": _mean(n_active_list),
        "median_n_active_ids": _median(n_active_list),
        "mean_effective_num_ids": _mean(eff_list),
        "median_effective_num_ids": _median(eff_list),
        "mean_entropy": _mean(ent_list),
        "mean_normalized_entropy": _mean(ent_n_list),
        "mean_top1_share": _mean(top1_list),
        "mean_top3_share": _mean(top3_list),
        "mean_top5_share": _mean(top5_list),
        "mean_top10_share": _mean(top10_list),
    }]
    return per_sample_out, summary_rows


def compute_actionrec_id_perturbation(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    base_metrics: Dict[str, float],
    attr_rows: Sequence[Dict[str, object]],
    top_k_ids: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ranked_ids = _rank_ids_from_attr_rows(attr_rows, top_k=top_k_ids)
    for rank, item_id in enumerate(ranked_ids, start=1):
        pert_samples, pstats = _perturb_samples_by_global_id_drop(samples, drop_id=int(item_id))
        pert_metrics = eval_actionrec_samples(
            model=model,
            samples=pert_samples,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            batch_size=batch_size,
            device=device,
            n_classes=n_classes,
        )
        delta = _summarize_action_delta(base_metrics, pert_metrics)
        rows.append({
            "rank": rank,
            "item_id": int(item_id),
            "item_name": str(int(item_id)),
            "perturbation": "drop_single_id",
            "importance_score": float(delta["primary_metric_drop"]),
            **delta,
            **pstats,
        })
    return rows


def compute_retrieval_id_perturbation(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    base_metrics: Dict[str, object],
    cached_text_emb: np.ndarray,
    attr_rows: Sequence[Dict[str, object]],
    top_k_ids: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ranked_ids = _rank_ids_from_attr_rows(attr_rows, top_k=top_k_ids)
    for rank, item_id in enumerate(ranked_ids, start=1):
        pert_samples, pstats = _perturb_samples_by_global_id_drop(samples, drop_id=int(item_id))
        pert_metrics = evaluate_retrieval(
            model=model,
            samples=pert_samples,
            motiongpt_root=motiongpt_root,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            max_text_len=max_text_len,
            batch_size=batch_size,
            device=device,
            cached_text_emb=cached_text_emb,
        )
        delta = _summarize_retrieval_delta(base_metrics, pert_metrics)
        rows.append({
            "rank": rank,
            "item_id": int(item_id),
            "item_name": str(int(item_id)),
            "perturbation": "drop_single_id",
            "importance_score": float(delta["primary_metric_drop"]),
            **delta,
            **pstats,
        })
    return rows


def compute_actionrec_motif_perturbation(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    base_metrics: Dict[str, float],
    motif_rows: Sequence[Dict[str, object]],
    top_k_motifs: int,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    candidates = list(motif_rows[: max(0, int(top_k_motifs))]) if int(top_k_motifs) > 0 else list(motif_rows)
    for rank, mrow in enumerate(candidates, start=1):
        motif_seq = [int(x) for x in str(mrow.get("motif_id_sequence", "")).split() if str(x).strip()]
        if not motif_seq:
            continue
        pert_samples, pstats = _perturb_samples_by_global_motif_drop(
            samples=samples,
            motif=motif_seq,
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        pert_metrics = eval_actionrec_samples(
            model=model,
            samples=pert_samples,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            batch_size=batch_size,
            device=device,
            n_classes=n_classes,
        )
        delta = _summarize_action_delta(base_metrics, pert_metrics)
        rows.append({
            "rank": rank,
            "motif_key": str(mrow.get("motif_key", _motif_to_key(motif_seq))),
            "motif_id_sequence": str(mrow.get("motif_id_sequence", _motif_to_key(motif_seq))),
            "motif_len": int(mrow.get("motif_len", len(motif_seq))),
            "support_samples": int(mrow.get("support_samples", 0)),
            "perturbation": "drop_motif_sequence_global",
            "drop_all_occurrences": int(bool(drop_all_occurrences)),
            "importance_score": float(delta["primary_metric_drop"]),
            **delta,
            **pstats,
        })
    return rows


def compute_retrieval_motif_perturbation(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    base_metrics: Dict[str, object],
    cached_text_emb: np.ndarray,
    motif_rows: Sequence[Dict[str, object]],
    top_k_motifs: int,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    candidates = list(motif_rows[: max(0, int(top_k_motifs))]) if int(top_k_motifs) > 0 else list(motif_rows)
    for rank, mrow in enumerate(candidates, start=1):
        motif_seq = [int(x) for x in str(mrow.get("motif_id_sequence", "")).split() if str(x).strip()]
        if not motif_seq:
            continue
        pert_samples, pstats = _perturb_samples_by_global_motif_drop(
            samples=samples,
            motif=motif_seq,
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        pert_metrics = evaluate_retrieval(
            model=model,
            samples=pert_samples,
            motiongpt_root=motiongpt_root,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            max_text_len=max_text_len,
            batch_size=batch_size,
            device=device,
            cached_text_emb=cached_text_emb,
        )
        delta = _summarize_retrieval_delta(base_metrics, pert_metrics)
        rows.append({
            "rank": rank,
            "motif_key": str(mrow.get("motif_key", _motif_to_key(motif_seq))),
            "motif_id_sequence": str(mrow.get("motif_id_sequence", _motif_to_key(motif_seq))),
            "motif_len": int(mrow.get("motif_len", len(motif_seq))),
            "support_samples": int(mrow.get("support_samples", 0)),
            "perturbation": "drop_motif_sequence_global",
            "drop_all_occurrences": int(bool(drop_all_occurrences)),
            "importance_score": float(delta["primary_metric_drop"]),
            **delta,
            **pstats,
        })
    return rows


def compute_actionrec_motif_set_perturbation(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    base_metrics: Dict[str, float],
    motif_rows: Sequence[Dict[str, object]],
    top_k_values: Sequence[int],
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ranked = [r for r in motif_rows if str(r.get("motif_id_sequence", "")).strip()]
    motifs = [[int(x) for x in str(r.get("motif_id_sequence", "")).split() if str(x).strip()] for r in ranked]
    keys = [str(r.get("motif_key", _motif_to_key(m))) for r, m in zip(ranked, motifs)]
    max_k = len(motifs)
    if max_k <= 0:
        return rows
    for k in sorted(set(max(1, int(v)) for v in top_k_values if int(v) > 0)):
        kk = min(k, max_k)
        sel_motifs = motifs[:kk]
        sel_keys = keys[:kk]
        pert_samples, pstats = _perturb_samples_by_global_motif_set_drop(
            samples=samples,
            motifs=sel_motifs,
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        pert_metrics = eval_actionrec_samples(
            model=model,
            samples=pert_samples,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            batch_size=batch_size,
            device=device,
            n_classes=n_classes,
        )
        delta = _summarize_action_delta(base_metrics, pert_metrics)
        rows.append(
            {
                "set_size_k": kk,
                "selection_mode": "top_k_global_motifs",
                "perturbation": "drop_motif_set_global",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                "selected_motif_keys": " || ".join(sel_keys),
                "importance_score": float(delta["primary_metric_drop"]),
                **delta,
                **pstats,
            }
        )
    return rows


def compute_retrieval_motif_set_perturbation(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    base_metrics: Dict[str, object],
    cached_text_emb: np.ndarray,
    motif_rows: Sequence[Dict[str, object]],
    top_k_values: Sequence[int],
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ranked = [r for r in motif_rows if str(r.get("motif_id_sequence", "")).strip()]
    motifs = [[int(x) for x in str(r.get("motif_id_sequence", "")).split() if str(x).strip()] for r in ranked]
    keys = [str(r.get("motif_key", _motif_to_key(m))) for r, m in zip(ranked, motifs)]
    max_k = len(motifs)
    if max_k <= 0:
        return rows
    for k in sorted(set(max(1, int(v)) for v in top_k_values if int(v) > 0)):
        kk = min(k, max_k)
        sel_motifs = motifs[:kk]
        sel_keys = keys[:kk]
        pert_samples, pstats = _perturb_samples_by_global_motif_set_drop(
            samples=samples,
            motifs=sel_motifs,
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        pert_metrics = evaluate_retrieval(
            model=model,
            samples=pert_samples,
            motiongpt_root=motiongpt_root,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            max_text_len=max_text_len,
            batch_size=batch_size,
            device=device,
            cached_text_emb=cached_text_emb,
        )
        delta = _summarize_retrieval_delta(base_metrics, pert_metrics)
        rows.append(
            {
                "set_size_k": kk,
                "selection_mode": "top_k_global_motifs",
                "perturbation": "drop_motif_set_global",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                "selected_motif_keys": " || ".join(sel_keys),
                "importance_score": float(delta["primary_metric_drop"]),
                **delta,
                **pstats,
            }
        )
    return rows


def compute_actionrec_budget_curves(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    base_metrics: Dict[str, float],
    per_sample_rows: Sequence[Dict[str, object]],
    top_p_values: Sequence[float],
    top_k_values: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample_attr_ranked = _build_mid_to_ranked_attr(per_sample_rows)
    for p in top_p_values:
        for mode, curve_kind in (("keep_top_p_mass", "sufficiency"), ("drop_top_p_mass", "necessity")):
            pert_samples, pstats = _perturb_samples_by_sample_attr(samples, sample_attr_ranked, mode, float(p))
            pert_metrics = eval_actionrec_samples(
                model=model,
                samples=pert_samples,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                batch_size=batch_size,
                device=device,
                n_classes=n_classes,
            )
            delta = _summarize_action_delta(base_metrics, pert_metrics)
            rows.append({
                "curve_kind": curve_kind,
                "budget_kind": "top_p_mass",
                "budget_value": float(p),
                "selection_mode": mode,
                **delta,
                **pstats,
            })
    for k in top_k_values:
        for mode, curve_kind in (("keep_top_k_ids", "sufficiency"), ("drop_top_k_ids", "necessity")):
            pert_samples, pstats = _perturb_samples_by_sample_attr(samples, sample_attr_ranked, mode, float(k))
            pert_metrics = eval_actionrec_samples(
                model=model,
                samples=pert_samples,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                batch_size=batch_size,
                device=device,
                n_classes=n_classes,
            )
            delta = _summarize_action_delta(base_metrics, pert_metrics)
            rows.append({
                "curve_kind": curve_kind,
                "budget_kind": "top_k_ids",
                "budget_value": int(k),
                "selection_mode": mode,
                **delta,
                **pstats,
            })
    return rows


def compute_retrieval_budget_curves(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    base_metrics: Dict[str, object],
    cached_text_emb: np.ndarray,
    per_sample_rows: Sequence[Dict[str, object]],
    top_p_values: Sequence[float],
    top_k_values: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample_attr_ranked = _build_mid_to_ranked_attr(per_sample_rows)
    for p in top_p_values:
        for mode, curve_kind in (("keep_top_p_mass", "sufficiency"), ("drop_top_p_mass", "necessity")):
            pert_samples, pstats = _perturb_samples_by_sample_attr(samples, sample_attr_ranked, mode, float(p))
            pert_metrics = evaluate_retrieval(
                model=model,
                samples=pert_samples,
                motiongpt_root=motiongpt_root,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                max_text_len=max_text_len,
                batch_size=batch_size,
                device=device,
                cached_text_emb=cached_text_emb,
            )
            delta = _summarize_retrieval_delta(base_metrics, pert_metrics)
            rows.append({
                "curve_kind": curve_kind,
                "budget_kind": "top_p_mass",
                "budget_value": float(p),
                "selection_mode": mode,
                **delta,
                **pstats,
            })
    for k in top_k_values:
        for mode, curve_kind in (("keep_top_k_ids", "sufficiency"), ("drop_top_k_ids", "necessity")):
            pert_samples, pstats = _perturb_samples_by_sample_attr(samples, sample_attr_ranked, mode, float(k))
            pert_metrics = evaluate_retrieval(
                model=model,
                samples=pert_samples,
                motiongpt_root=motiongpt_root,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                max_text_len=max_text_len,
                batch_size=batch_size,
                device=device,
                cached_text_emb=cached_text_emb,
            )
            delta = _summarize_retrieval_delta(base_metrics, pert_metrics)
            rows.append({
                "curve_kind": curve_kind,
                "budget_kind": "top_k_ids",
                "budget_value": int(k),
                "selection_mode": mode,
                **delta,
                **pstats,
            })
    return rows


def _build_mid_to_id_weight_map(per_sample_rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for r in per_sample_rows:
        mid = str(r.get("mid", "")).strip()
        if not mid:
            continue
        tid = int(r.get("item_id", 0))
        w = float(r.get("abs_attr_norm", abs(float(r.get("signed_attr_norm", 0.0)))))
        if w <= 0.0:
            continue
        out.setdefault(mid, {})
        out[mid][tid] = out[mid].get(tid, 0.0) + w
    return out


def _dedup_consecutive(seq: Sequence[int]) -> List[int]:
    out: List[int] = []
    prev = None
    for x in seq:
        xx = int(x)
        if prev is not None and xx == prev:
            continue
        out.append(xx)
        prev = xx
    return out


def _weighted_jaccard(a: Dict[int, float], b: Dict[int, float]) -> float:
    keys = set(int(k) for k in a.keys()) | set(int(k) for k in b.keys())
    if not keys:
        return 0.0
    inter = 0.0
    union = 0.0
    for k in keys:
        va = float(a.get(int(k), 0.0))
        vb = float(b.get(int(k), 0.0))
        inter += min(va, vb)
        union += max(va, vb)
    if union <= 1e-12:
        return 0.0
    return float(inter / union)


def _lcs_len(a: Sequence[int], b: Sequence[int]) -> int:
    na = len(a)
    nb = len(b)
    if na == 0 or nb == 0:
        return 0
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        ai = int(a[i - 1])
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, nb + 1):
            if ai == int(b[j - 1]):
                row[j] = prev_row[j - 1] + 1
            else:
                row[j] = row[j - 1] if row[j - 1] >= prev_row[j] else prev_row[j]
    return int(dp[na][nb])


def _sparse_seq_similarity(
    seq_a: Sequence[int],
    seq_b: Sequence[int],
    w_a: Dict[int, float],
    w_b: Dict[int, float],
    jaccard_mix: float,
) -> Tuple[float, float, float]:
    jac = _weighted_jaccard(w_a, w_b)
    denom = max(len(seq_a), len(seq_b), 1)
    lcs_n = float(_lcs_len(seq_a, seq_b) / float(denom))
    a = min(1.0, max(0.0, float(jaccard_mix)))
    sim = float(a * jac + (1.0 - a) * lcs_n)
    return sim, jac, lcs_n


def build_sparse_sequences_per_sample(
    samples: Sequence[Sample],
    per_sample_rows: Sequence[Dict[str, object]],
    select_mode: str,
    top_p_mass: float,
    top_k_ids: int,
    min_len: int,
    max_len: int,
    dedup_consecutive: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    sample_attr_ranked = _build_mid_to_ranked_attr(per_sample_rows)
    mid_to_w = _build_mid_to_id_weight_map(per_sample_rows)
    out_rows: List[Dict[str, object]] = []
    candidates: List[Dict[str, object]] = []

    lo = max(1, int(min_len))
    hi = int(max_len)

    for s in samples:
        mid = str(s.mid)
        ranked = sample_attr_ranked.get(mid, [])
        if not ranked:
            continue
        if str(select_mode) == "top_p_mass":
            keep_ids = _select_ids_by_top_p(ranked, float(top_p_mass), keep=True)
        elif str(select_mode) == "top_k_ids":
            keep_ids = _select_ids_by_top_k(ranked, int(top_k_ids), keep=True)
        else:
            raise ValueError(f"unsupported sparse_seq select_mode: {select_mode}")
        if not keep_ids:
            continue

        seq = [int(t) for t in s.token_ids if int(t) in keep_ids]
        if bool(dedup_consecutive):
            seq = _dedup_consecutive(seq)
        if hi > 0 and len(seq) > hi:
            seq = seq[:hi]
        if len(seq) < lo:
            continue

        wmap = mid_to_w.get(mid, {})
        per_id_weight: Dict[int, float] = {}
        token_weight_mass = 0.0
        for t in seq:
            w = float(wmap.get(int(t), 0.0))
            per_id_weight[int(t)] = max(per_id_weight.get(int(t), 0.0), w)
            token_weight_mass += w
        selected_weight_mass = float(sum(float(w) for tid, w in ranked if int(tid) in keep_ids))

        class_name = ""
        if s.label is not None and 0 <= int(s.label) < len(ar.CLASS_NAMES):
            class_name = str(ar.CLASS_NAMES[int(s.label)])
        verb = ""
        adjective = ""
        if "__" in class_name:
            verb, adjective = class_name.split("__", 1)

        rec = {
            "mid": mid,
            "class_name": class_name,
            "verb": verb,
            "adjective": adjective,
            "sparse_seq_id_sequence": _motif_to_key(seq),
            "sparse_seq_len": int(len(seq)),
            "n_selected_ids": int(len(keep_ids)),
            "token_weight_mass": float(token_weight_mass),
            "selected_id_weight_mass": float(selected_weight_mass),
            "select_mode": str(select_mode),
            "select_top_p_mass": float(top_p_mass),
            "select_top_k_ids": int(top_k_ids),
        }
        out_rows.append(rec)
        candidates.append(
            {
                "mid": mid,
                "seq": seq,
                "per_id_weight": per_id_weight,
                "token_weight_mass": float(token_weight_mass),
                "selected_id_weight_mass": float(selected_weight_mass),
                "class_name": class_name,
            }
        )
    return out_rows, candidates


def cluster_sparse_sequences(
    candidates: Sequence[Dict[str, object]],
    similarity_threshold: float,
    jaccard_mix: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    n = len(candidates)
    if n <= 0:
        return [], []
    thr = min(1.0, max(0.0, float(similarity_threshold)))
    sims = np.eye(n, dtype=np.float64)
    jacs = np.eye(n, dtype=np.float64)
    lcss = np.eye(n, dtype=np.float64)
    adj: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        ai = candidates[i]
        seq_i = ai["seq"]
        w_i = ai["per_id_weight"]
        for j in range(i + 1, n):
            bj = candidates[j]
            sim, jac, lcs_n = _sparse_seq_similarity(
                seq_i, bj["seq"], w_i, bj["per_id_weight"], jaccard_mix=float(jaccard_mix)
            )
            sims[i, j] = sims[j, i] = sim
            jacs[i, j] = jacs[j, i] = jac
            lcss[i, j] = lcss[j, i] = lcs_n
            if sim >= thr:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        clusters.append(comp)

    cluster_rows: List[Dict[str, object]] = []
    cluster_objs: List[Dict[str, object]] = []

    for cid, comp in enumerate(clusters, start=1):
        if not comp:
            continue
        support = len(comp)
        best_idx = comp[0]
        best_avg = -1.0
        for ii in comp:
            avg = float(np.mean([sims[ii, jj] for jj in comp])) if comp else 0.0
            if avg > best_avg:
                best_avg = avg
                best_idx = ii
        rep = candidates[best_idx]
        mids = [str(candidates[ii]["mid"]) for ii in comp]
        mass_list = [float(candidates[ii]["token_weight_mass"]) for ii in comp]
        sel_mass_list = [float(candidates[ii]["selected_id_weight_mass"]) for ii in comp]

        sim_vals: List[float] = []
        jac_vals: List[float] = []
        lcs_vals: List[float] = []
        for a in range(len(comp)):
            for b in range(a + 1, len(comp)):
                i = comp[a]
                j = comp[b]
                sim_vals.append(float(sims[i, j]))
                jac_vals.append(float(jacs[i, j]))
                lcs_vals.append(float(lcss[i, j]))

        cluster_rows.append(
            {
                "cluster_id": int(cid),
                "support_samples": int(support),
                "representative_mid": str(rep["mid"]),
                "representative_sparse_seq": _motif_to_key(rep["seq"]),
                "representative_seq_len": int(len(rep["seq"])),
                "member_mids": " || ".join(mids),
                "mean_pair_similarity": float(np.mean(sim_vals)) if sim_vals else 1.0,
                "mean_pair_weighted_jaccard": float(np.mean(jac_vals)) if jac_vals else 1.0,
                "mean_pair_lcs_similarity": float(np.mean(lcs_vals)) if lcs_vals else 1.0,
                "representative_avg_similarity": float(best_avg),
                "mean_token_weight_mass": float(np.mean(mass_list)) if mass_list else float("nan"),
                "mean_selected_id_weight_mass": float(np.mean(sel_mass_list)) if sel_mass_list else float("nan"),
            }
        )
        cluster_objs.append(
            {
                "cluster_id": int(cid),
                "representative_seq": [int(x) for x in rep["seq"]],
                "support_samples": int(support),
                "mean_pair_similarity": float(np.mean(sim_vals)) if sim_vals else 1.0,
                "mean_pair_weighted_jaccard": float(np.mean(jac_vals)) if jac_vals else 1.0,
                "mean_pair_lcs_similarity": float(np.mean(lcs_vals)) if lcs_vals else 1.0,
                "representative_mid": str(rep["mid"]),
                "mean_token_weight_mass": float(np.mean(mass_list)) if mass_list else float("nan"),
                "mean_selected_id_weight_mass": float(np.mean(sel_mass_list)) if sel_mass_list else float("nan"),
            }
        )

    cluster_rows = sorted(
        cluster_rows,
        key=lambda r: (
            -int(r.get("support_samples", 0)),
            -float(r.get("mean_token_weight_mass", float("-inf"))),
            str(r.get("representative_sparse_seq", "")),
        ),
    )
    rank_map: Dict[int, int] = {}
    for rank, r in enumerate(cluster_rows, start=1):
        r["rank"] = rank
        rank_map[int(r["cluster_id"])] = rank

    for obj in cluster_objs:
        cid = int(obj["cluster_id"])
        obj["rank"] = int(rank_map.get(cid, 0))

    cluster_objs = sorted(cluster_objs, key=lambda o: int(o.get("rank", 0)))
    return cluster_rows, cluster_objs


def compute_actionrec_sparse_sequence_cluster_perturbation(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    base_metrics: Dict[str, float],
    clusters: Sequence[Dict[str, object]],
    top_k_clusters: int,
    min_gap: int,
    max_gap: int,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ranked = list(clusters)
    if int(top_k_clusters) > 0:
        ranked = ranked[: int(top_k_clusters)]
    for c in ranked:
        seq = [int(x) for x in c.get("representative_seq", [])]
        if not seq:
            continue
        motifs = [[int(t)] for t in seq]
        pert_samples, pstats = _perturb_samples_by_global_motif_chain_drop(
            samples=samples,
            motifs=motifs,
            min_gap=int(min_gap),
            max_gap=int(max_gap),
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        pert_metrics = eval_actionrec_samples(
            model=model,
            samples=pert_samples,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            batch_size=batch_size,
            device=device,
            n_classes=n_classes,
        )
        delta = _summarize_action_delta(base_metrics, pert_metrics)
        rows.append(
            {
                "rank": int(c.get("rank", 0)),
                "cluster_id": int(c.get("cluster_id", 0)),
                "support_samples": int(c.get("support_samples", 0)),
                "representative_mid": str(c.get("representative_mid", "")),
                "representative_sparse_seq": _motif_to_key(seq),
                "representative_seq_len": len(seq),
                "mean_pair_similarity": float(c.get("mean_pair_similarity", float("nan"))),
                "mean_pair_weighted_jaccard": float(c.get("mean_pair_weighted_jaccard", float("nan"))),
                "mean_pair_lcs_similarity": float(c.get("mean_pair_lcs_similarity", float("nan"))),
                "perturbation": "drop_sparse_id_chain_global",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                "chain_min_gap": int(min_gap),
                "chain_max_gap": int(max_gap),
                "importance_score": float(delta["primary_metric_drop"]),
                **delta,
                **pstats,
            }
        )
    return rows


def compute_retrieval_sparse_sequence_cluster_perturbation(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    base_metrics: Dict[str, object],
    cached_text_emb: np.ndarray,
    clusters: Sequence[Dict[str, object]],
    top_k_clusters: int,
    min_gap: int,
    max_gap: int,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ranked = list(clusters)
    if int(top_k_clusters) > 0:
        ranked = ranked[: int(top_k_clusters)]
    for c in ranked:
        seq = [int(x) for x in c.get("representative_seq", [])]
        if not seq:
            continue
        motifs = [[int(t)] for t in seq]
        pert_samples, pstats = _perturb_samples_by_global_motif_chain_drop(
            samples=samples,
            motifs=motifs,
            min_gap=int(min_gap),
            max_gap=int(max_gap),
            drop_all_occurrences=bool(drop_all_occurrences),
        )
        pert_metrics = evaluate_retrieval(
            model=model,
            samples=pert_samples,
            motiongpt_root=motiongpt_root,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            max_text_len=max_text_len,
            batch_size=batch_size,
            device=device,
            cached_text_emb=cached_text_emb,
        )
        delta = _summarize_retrieval_delta(base_metrics, pert_metrics)
        rows.append(
            {
                "rank": int(c.get("rank", 0)),
                "cluster_id": int(c.get("cluster_id", 0)),
                "support_samples": int(c.get("support_samples", 0)),
                "representative_mid": str(c.get("representative_mid", "")),
                "representative_sparse_seq": _motif_to_key(seq),
                "representative_seq_len": len(seq),
                "mean_pair_similarity": float(c.get("mean_pair_similarity", float("nan"))),
                "mean_pair_weighted_jaccard": float(c.get("mean_pair_weighted_jaccard", float("nan"))),
                "mean_pair_lcs_similarity": float(c.get("mean_pair_lcs_similarity", float("nan"))),
                "perturbation": "drop_sparse_id_chain_global",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                "chain_min_gap": int(min_gap),
                "chain_max_gap": int(max_gap),
                "importance_score": float(delta["primary_metric_drop"]),
                **delta,
                **pstats,
            }
        )
    return rows


def _motif_to_key(motif: Sequence[int]) -> str:
    return " ".join(str(int(t)) for t in motif)


def _find_subsequence_occurrences(token_ids: Sequence[int], motif: Sequence[int]) -> List[int]:
    seq = [int(x) for x in token_ids]
    sub = [int(x) for x in motif]
    n = len(seq)
    m = len(sub)
    if m <= 0 or n < m:
        return []
    out: List[int] = []
    for i in range(0, n - m + 1):
        if seq[i : i + m] == sub:
            out.append(i)
    return out


def _drop_motif_occurrences(
    token_ids: Sequence[int],
    motif: Sequence[int],
    drop_all_occurrences: bool,
) -> Tuple[List[int], int]:
    seq = [int(x) for x in token_ids]
    sub = [int(x) for x in motif]
    n = len(seq)
    m = len(sub)
    if m <= 0 or n < m:
        return seq, 0
    out: List[int] = []
    i = 0
    removed = 0
    while i < n:
        if i + m <= n and seq[i : i + m] == sub:
            removed += 1
            i += m
            if not bool(drop_all_occurrences):
                out.extend(seq[i:])
                break
            continue
        out.append(seq[i])
        i += 1
    return out, removed


def extract_sample_motifs(
    samples: Sequence[Sample],
    per_sample_rows: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    min_len: int,
    max_len: int,
    top_k_per_sample: int,
    min_mean_weight: float,
) -> List[Dict[str, object]]:
    mid_to_w = _build_mid_to_id_weight_map(per_sample_rows)
    lo = max(1, int(min_len))
    hi = max(lo, int(max_len))
    rows: List[Dict[str, object]] = []

    for s in samples:
        seq = [int(t) for t in s.token_ids]
        if len(seq) < lo:
            continue
        wmap = mid_to_w.get(str(s.mid), {})
        total_weight = sum(float(wmap.get(t, 0.0)) for t in seq)
        cand: Dict[Tuple[int, ...], Dict[str, object]] = {}

        for st in range(len(seq)):
            for L in range(lo, hi + 1):
                ed = st + L
                if ed > len(seq):
                    break
                motif = tuple(seq[st:ed])
                token_weights = [float(wmap.get(t, 0.0)) for t in motif]
                mass = float(sum(token_weights))
                mean_w = float(mass / max(1, L))
                if mass <= 0.0 or mean_w < float(min_mean_weight):
                    continue
                rec = cand.get(motif)
                if rec is None:
                    cand[motif] = {
                        "best_start_idx": int(st),
                        "best_end_idx_exclusive": int(ed),
                        "motif_weight_mass": mass,
                        "mean_token_weight": mean_w,
                        "occurrences_in_sample": 1,
                    }
                else:
                    rec["occurrences_in_sample"] = int(rec["occurrences_in_sample"]) + 1
                    if (mean_w > float(rec["mean_token_weight"])) or (
                        abs(mean_w - float(rec["mean_token_weight"])) <= 1e-12 and mass > float(rec["motif_weight_mass"])
                    ):
                        rec["best_start_idx"] = int(st)
                        rec["best_end_idx_exclusive"] = int(ed)
                        rec["motif_weight_mass"] = mass
                        rec["mean_token_weight"] = mean_w

        ranked = sorted(
            cand.items(),
            key=lambda kv: (
                -float(kv[1]["mean_token_weight"]),
                -float(kv[1]["motif_weight_mass"]),
                -int(kv[1]["occurrences_in_sample"]),
                kv[0],
            ),
        )
        if int(top_k_per_sample) > 0:
            ranked = ranked[: int(top_k_per_sample)]

        class_name = ""
        if s.label is not None and 0 <= int(s.label) < len(class_names):
            class_name = str(class_names[int(s.label)])
        verb = ""
        adjective = ""
        if "__" in class_name:
            verb, adjective = class_name.split("__", 1)

        for rank, (motif, info) in enumerate(ranked, start=1):
            motif_key = _motif_to_key(motif)
            rows.append(
                {
                    "mid": str(s.mid),
                    "class_name": class_name,
                    "verb": verb,
                    "adjective": adjective,
                    "motif_rank_in_sample": rank,
                    "motif_len": len(motif),
                    "motif_id_sequence": motif_key,
                    "motif_key": motif_key,
                    "best_start_idx": int(info["best_start_idx"]),
                    "best_end_idx_exclusive": int(info["best_end_idx_exclusive"]),
                    "occurrences_in_sample": int(info["occurrences_in_sample"]),
                    "mean_token_weight": float(info["mean_token_weight"]),
                    "motif_weight_mass": float(info["motif_weight_mass"]),
                    "sample_total_token_weight": float(total_weight),
                    "motif_weight_share_in_sample": _safe_ratio(float(info["motif_weight_mass"]), float(total_weight)),
                }
            )
    return rows


def mine_frequent_contiguous_motifs(
    samples: Sequence[Sample],
    min_len: int,
    max_len: int,
    min_support: int,
    top_k: int,
) -> List[Dict[str, object]]:
    lo = max(1, int(min_len))
    hi = max(lo, int(max_len))
    support_by_motif: Dict[Tuple[int, ...], int] = Counter()
    occ_by_motif: Dict[Tuple[int, ...], int] = Counter()

    for s in samples:
        seq = [int(t) for t in s.token_ids]
        if len(seq) < lo:
            continue
        seen_in_sample: Set[Tuple[int, ...]] = set()
        for st in range(len(seq)):
            for L in range(lo, hi + 1):
                ed = st + L
                if ed > len(seq):
                    break
                motif = tuple(seq[st:ed])
                occ_by_motif[motif] += 1
                seen_in_sample.add(motif)
        for motif in seen_in_sample:
            support_by_motif[motif] += 1

    candidates: List[Tuple[Tuple[int, ...], int, int, float]] = []
    for motif, sup in support_by_motif.items():
        if int(sup) < max(1, int(min_support)):
            continue
        occ = int(occ_by_motif.get(motif, 0))
        # Prefer motifs that recur across many samples and appear repeatedly.
        cscore = float(sup) * math.log1p(float(occ))
        candidates.append((motif, int(sup), occ, cscore))

    candidates = sorted(
        candidates,
        key=lambda x: (-x[1], -x[2], -len(x[0]), x[0]),
    )
    if int(top_k) > 0:
        candidates = candidates[: int(top_k)]

    rows: List[Dict[str, object]] = []
    for rank, (motif, sup, occ, cscore) in enumerate(candidates, start=1):
        key = _motif_to_key(motif)
        rows.append(
            {
                "rank": rank,
                "motif_key": key,
                "motif_id_sequence": key,
                "motif_len": len(motif),
                "support_samples": int(sup),
                "total_occurrences_in_samples": int(occ),
                "candidate_score": float(cscore),
                "mining_method": "frequent_contiguous",
            }
        )
    return rows


def build_motif_summary_from_global_perturbation(
    candidate_rows: Sequence[Dict[str, object]],
    global_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    cmap = {str(r.get("motif_key", "")): dict(r) for r in candidate_rows}
    out: List[Dict[str, object]] = []
    for r in global_rows:
        key = str(r.get("motif_key", ""))
        c = cmap.get(key, {})
        merged = {
            "motif_key": key,
            "motif_id_sequence": str(r.get("motif_id_sequence", c.get("motif_id_sequence", key))),
            "motif_len": int(r.get("motif_len", c.get("motif_len", 0))),
            "support_samples": int(r.get("support_samples", c.get("support_samples", 0))),
            "total_occurrences_in_samples": int(c.get("total_occurrences_in_samples", 0)),
            "candidate_score": float(c.get("candidate_score", float("nan"))),
            "importance_score": float(r.get("importance_score", float("nan"))),
            "primary_metric_drop": float(r.get("primary_metric_drop", float("nan"))),
            "removed_token_ratio": float(r.get("removed_token_ratio", float("nan"))),
            "samples_with_motif": int(r.get("samples_with_motif", 0)),
            "total_matched_occurrences": int(r.get("total_matched_occurrences", 0)),
        }
        out.append(merged)

    out = sorted(
        out,
        key=lambda x: (-float(x.get("importance_score", float("-inf"))), -int(x.get("support_samples", 0)), str(x.get("motif_key", ""))),
    )
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out


def _build_chain_motif_pool(
    motif_rows: Sequence[Dict[str, object]],
    top_k_motifs: int,
) -> List[Tuple[str, List[int], int]]:
    motifs: List[Tuple[str, List[int]]] = []
    seen: Set[str] = set()
    for r in motif_rows:
        key = str(r.get("motif_key", r.get("motif_id_sequence", ""))).strip()
        if not key or key in seen:
            continue
        seq = [int(x) for x in str(r.get("motif_id_sequence", key)).split() if str(x).strip()]
        if not seq:
            continue
        motifs.append((key, seq))
        seen.add(key)
        if int(top_k_motifs) > 0 and len(motifs) >= int(top_k_motifs):
            break
    out: List[Tuple[str, List[int], int]] = []
    for rank, (k, seq) in enumerate(motifs, start=1):
        out.append((k, seq, rank))
    return out


def compute_actionrec_motif_chain_greedy(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    vocab_size: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    n_classes: int,
    base_metrics: Dict[str, float],
    motif_rows: Sequence[Dict[str, object]],
    top_k_motifs: int,
    max_chain_len: int,
    min_gap: int,
    max_gap: int,
    min_delta: float,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pool = _build_chain_motif_pool(motif_rows, top_k_motifs=top_k_motifs)
    selected: List[Tuple[str, List[int]]] = []
    used_keys: Set[str] = set()
    current_drop = 0.0
    for step in range(1, max(1, int(max_chain_len)) + 1):
        best = None
        best_delta = float("-inf")
        best_pstats = None
        for key, seq, _rank in pool:
            if key in used_keys:
                continue
            chain = [s for _k, s in selected] + [seq]
            pert_samples, pstats = _perturb_samples_by_global_motif_chain_drop(
                samples=samples,
                motifs=chain,
                min_gap=int(min_gap),
                max_gap=int(max_gap),
                drop_all_occurrences=bool(drop_all_occurrences),
            )
            pert_metrics = eval_actionrec_samples(
                model=model,
                samples=pert_samples,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                batch_size=batch_size,
                device=device,
                n_classes=n_classes,
            )
            delta = _summarize_action_delta(base_metrics, pert_metrics)
            new_drop = float(delta["primary_metric_drop"])
            gain = float(new_drop - current_drop)
            if gain > best_delta:
                best_delta = gain
                best = (key, seq, delta, new_drop)
                best_pstats = pstats
        if best is None or best_delta < float(min_delta):
            break
        bkey, bseq, bdelta, bnew_drop = best
        selected.append((bkey, bseq))
        used_keys.add(bkey)
        current_drop = bnew_drop
        rows.append(
            {
                "chain_step": step,
                "added_motif_key": bkey,
                "chain_keys": " -> ".join(k for k, _s in selected),
                "chain_size": len(selected),
                "chain_min_gap": int(min_gap),
                "chain_max_gap": int(max_gap),
                "gain_primary_metric_drop": float(best_delta),
                "importance_score": float(bdelta["primary_metric_drop"]),
                "perturbation": "drop_ordered_motif_chain_global",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                **bdelta,
                **(best_pstats or {}),
            }
        )
    return rows


def compute_retrieval_motif_chain_greedy(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motiongpt_root: str,
    vocab_size: int,
    max_tokens: int,
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    base_metrics: Dict[str, object],
    cached_text_emb: np.ndarray,
    motif_rows: Sequence[Dict[str, object]],
    top_k_motifs: int,
    max_chain_len: int,
    min_gap: int,
    max_gap: int,
    min_delta: float,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pool = _build_chain_motif_pool(motif_rows, top_k_motifs=top_k_motifs)
    selected: List[Tuple[str, List[int]]] = []
    used_keys: Set[str] = set()
    current_drop = 0.0
    for step in range(1, max(1, int(max_chain_len)) + 1):
        best = None
        best_delta = float("-inf")
        best_pstats = None
        for key, seq, _rank in pool:
            if key in used_keys:
                continue
            chain = [s for _k, s in selected] + [seq]
            pert_samples, pstats = _perturb_samples_by_global_motif_chain_drop(
                samples=samples,
                motifs=chain,
                min_gap=int(min_gap),
                max_gap=int(max_gap),
                drop_all_occurrences=bool(drop_all_occurrences),
            )
            pert_metrics = evaluate_retrieval(
                model=model,
                samples=pert_samples,
                motiongpt_root=motiongpt_root,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                max_text_len=max_text_len,
                batch_size=batch_size,
                device=device,
                cached_text_emb=cached_text_emb,
            )
            delta = _summarize_retrieval_delta(base_metrics, pert_metrics)
            new_drop = float(delta["primary_metric_drop"])
            gain = float(new_drop - current_drop)
            if gain > best_delta:
                best_delta = gain
                best = (key, seq, delta, new_drop)
                best_pstats = pstats
        if best is None or best_delta < float(min_delta):
            break
        bkey, bseq, bdelta, bnew_drop = best
        selected.append((bkey, bseq))
        used_keys.add(bkey)
        current_drop = bnew_drop
        rows.append(
            {
                "chain_step": step,
                "added_motif_key": bkey,
                "chain_keys": " -> ".join(k for k, _s in selected),
                "chain_size": len(selected),
                "chain_min_gap": int(min_gap),
                "chain_max_gap": int(max_gap),
                "gain_primary_metric_drop": float(best_delta),
                "importance_score": float(bdelta["primary_metric_drop"]),
                "perturbation": "drop_ordered_motif_chain_global",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                **bdelta,
                **(best_pstats or {}),
            }
        )
    return rows


@torch.no_grad()
def _score_actionrec_single_sample(
    model: ar.ActionClassifier,
    sample: Sample,
    vocab_size: int,
    max_tokens: int,
    device: torch.device,
    score_target: str,
) -> float:
    if sample.label is None:
        return float("nan")
    if score_target not in {"true_logit", "neg_ce"}:
        raise ValueError(f"unsupported actionrec motif score_target: {score_target}")
    x, mask, _ = collate_tokens_safe([sample.token_ids], vocab_size=vocab_size, max_len=max_tokens)
    x = x.to(device)
    mask = mask.to(device)
    y = torch.tensor([int(sample.label)], dtype=torch.long, device=device)
    logits = model(x, mask)
    if score_target == "true_logit":
        return float(logits[0, int(y.item())].detach().cpu().item())
    return float((-F.cross_entropy(logits, y)).detach().cpu().item())


@torch.no_grad()
def _score_retrieval_single_sample(
    model: rt.DualEncoder,
    sample: Sample,
    wvec,
    max_text_len: int,
    vocab_size: int,
    max_tokens: int,
    device: torch.device,
    score_target: str,
) -> float:
    if score_target != "diag_cosine":
        raise ValueError(f"unsupported retrieval motif score_target: {score_target}")
    x, mask, _ = collate_tokens_safe([sample.token_ids], vocab_size=vocab_size, max_len=max_tokens)
    x = x.to(device)
    mask = mask.to(device)
    w, p, L = build_word_pos_tensors(wvec, sample.text_tokens, max_text_len=max_text_len)
    word = torch.from_numpy(np.expand_dims(w, axis=0)).to(device)
    pos = torch.from_numpy(np.expand_dims(p, axis=0)).to(device)
    tlen = torch.tensor([int(L)], dtype=torch.long, device=device)
    m = model.motion_enc(x, mask)
    t = model.text_enc(word, pos, tlen)
    return float((t * m).sum(dim=-1)[0].detach().cpu().item())


def validate_actionrec_motifs_with_perturbation(
    model: ar.ActionClassifier,
    samples: Sequence[Sample],
    motif_rows: Sequence[Dict[str, object]],
    vocab_size: int,
    max_tokens: int,
    device: torch.device,
    score_target: str,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    mid_to_sample = {str(s.mid): s for s in samples}
    base_cache: Dict[str, float] = {}
    out: List[Dict[str, object]] = []

    for r in motif_rows:
        mid = str(r.get("mid", ""))
        sample = mid_to_sample.get(mid)
        if sample is None:
            continue
        motif = [int(x) for x in str(r.get("motif_id_sequence", "")).split() if str(x).strip()]
        if not motif:
            continue

        base_score = base_cache.get(mid)
        if base_score is None:
            base_score = _score_actionrec_single_sample(
                model=model,
                sample=sample,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                device=device,
                score_target=score_target,
            )
            base_cache[mid] = base_score
        new_tokens, n_removed_occ = _drop_motif_occurrences(sample.token_ids, motif, drop_all_occurrences=drop_all_occurrences)
        if not new_tokens:
            new_tokens = list(sample.token_ids[:1]) if sample.token_ids else [0]
        pert_sample = Sample(mid=sample.mid, token_ids=new_tokens, label=sample.label, text_tokens=list(sample.text_tokens))
        pert_score = _score_actionrec_single_sample(
            model=model,
            sample=pert_sample,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            device=device,
            score_target=score_target,
        )
        out.append(
            {
                **dict(r),
                "score_target": score_target,
                "perturbation": "drop_motif_sequence",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                "matched_occurrences": int(n_removed_occ),
                "tokens_before": len(sample.token_ids),
                "tokens_after": len(new_tokens),
                "removed_token_ratio": _safe_ratio(float(len(sample.token_ids) - len(new_tokens)), float(max(1, len(sample.token_ids)))),
                "base_sample_score": float(base_score),
                "perturbed_sample_score": float(pert_score),
                "sample_score_drop": float(base_score - pert_score),
                "sample_score_ratio": _safe_ratio(float(pert_score), float(base_score)),
            }
        )
    return out


def validate_retrieval_motifs_with_perturbation(
    model: rt.DualEncoder,
    samples: Sequence[Sample],
    motif_rows: Sequence[Dict[str, object]],
    wvec,
    max_text_len: int,
    vocab_size: int,
    max_tokens: int,
    device: torch.device,
    score_target: str,
    drop_all_occurrences: bool,
) -> List[Dict[str, object]]:
    mid_to_sample = {str(s.mid): s for s in samples}
    base_cache: Dict[str, float] = {}
    out: List[Dict[str, object]] = []

    for r in motif_rows:
        mid = str(r.get("mid", ""))
        sample = mid_to_sample.get(mid)
        if sample is None:
            continue
        motif = [int(x) for x in str(r.get("motif_id_sequence", "")).split() if str(x).strip()]
        if not motif:
            continue

        base_score = base_cache.get(mid)
        if base_score is None:
            base_score = _score_retrieval_single_sample(
                model=model,
                sample=sample,
                wvec=wvec,
                max_text_len=max_text_len,
                vocab_size=vocab_size,
                max_tokens=max_tokens,
                device=device,
                score_target=score_target,
            )
            base_cache[mid] = base_score
        new_tokens, n_removed_occ = _drop_motif_occurrences(sample.token_ids, motif, drop_all_occurrences=drop_all_occurrences)
        if not new_tokens:
            new_tokens = list(sample.token_ids[:1]) if sample.token_ids else [0]
        pert_sample = Sample(mid=sample.mid, token_ids=new_tokens, label=sample.label, text_tokens=list(sample.text_tokens))
        pert_score = _score_retrieval_single_sample(
            model=model,
            sample=pert_sample,
            wvec=wvec,
            max_text_len=max_text_len,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            device=device,
            score_target=score_target,
        )
        out.append(
            {
                **dict(r),
                "score_target": score_target,
                "perturbation": "drop_motif_sequence",
                "drop_all_occurrences": int(bool(drop_all_occurrences)),
                "matched_occurrences": int(n_removed_occ),
                "tokens_before": len(sample.token_ids),
                "tokens_after": len(new_tokens),
                "removed_token_ratio": _safe_ratio(float(len(sample.token_ids) - len(new_tokens)), float(max(1, len(sample.token_ids)))),
                "base_sample_score": float(base_score),
                "perturbed_sample_score": float(pert_score),
                "sample_score_drop": float(base_score - pert_score),
                "sample_score_ratio": _safe_ratio(float(pert_score), float(base_score)),
            }
        )
    return out


def summarize_recurring_motifs(
    motif_rows: Sequence[Dict[str, object]],
    motif_perturb_rows: Sequence[Dict[str, object]],
    min_support: int,
    top_k: int,
) -> List[Dict[str, object]]:
    by_key: Dict[str, Dict[str, object]] = {}
    for r in motif_rows:
        key = str(r.get("motif_key", "")).strip()
        if not key:
            continue
        rec = by_key.setdefault(
            key,
            {
                "motif_key": key,
                "motif_id_sequence": str(r.get("motif_id_sequence", key)),
                "motif_len": int(r.get("motif_len", 0)),
                "samples": set(),
                "total_occ": 0,
                "motif_scores": [],
                "motif_weight_shares": [],
            },
        )
        rec["samples"].add(str(r.get("mid", "")))
        rec["total_occ"] = int(rec["total_occ"]) + int(r.get("occurrences_in_sample", 0))
        rec["motif_scores"].append(float(r.get("mean_token_weight", 0.0)))
        rec["motif_weight_shares"].append(float(r.get("motif_weight_share_in_sample", 0.0)))

    by_key_drop: Dict[str, List[float]] = {}
    by_key_drop_pos: Dict[str, int] = {}
    by_key_removed_ratio: Dict[str, List[float]] = {}
    for r in motif_perturb_rows:
        key = str(r.get("motif_key", "")).strip()
        if not key:
            continue
        drop = float(r.get("sample_score_drop", 0.0))
        by_key_drop.setdefault(key, []).append(drop)
        by_key_removed_ratio.setdefault(key, []).append(float(r.get("removed_token_ratio", 0.0)))
        if drop > 0.0:
            by_key_drop_pos[key] = int(by_key_drop_pos.get(key, 0)) + 1

    rows: List[Dict[str, object]] = []
    for key, rec in by_key.items():
        support = len(rec["samples"])
        if support < max(1, int(min_support)):
            continue
        drops = by_key_drop.get(key, [])
        pos = int(by_key_drop_pos.get(key, 0))
        rows.append(
            {
                "motif_key": key,
                "motif_id_sequence": rec["motif_id_sequence"],
                "motif_len": int(rec["motif_len"]),
                "support_samples": support,
                "total_occurrences_in_samples": int(rec["total_occ"]),
                "mean_token_weight": float(np.mean(rec["motif_scores"])) if rec["motif_scores"] else float("nan"),
                "mean_motif_weight_share_in_sample": float(np.mean(rec["motif_weight_shares"])) if rec["motif_weight_shares"] else float("nan"),
                "mean_score_drop": float(np.mean(drops)) if drops else float("nan"),
                "median_score_drop": float(np.median(np.asarray(drops, dtype=np.float64))) if drops else float("nan"),
                "positive_drop_rate": _safe_ratio(float(pos), float(len(drops))) if drops else float("nan"),
                "mean_removed_token_ratio": float(np.mean(by_key_removed_ratio.get(key, []))) if by_key_removed_ratio.get(key) else float("nan"),
            }
        )

    def _score_or_neginf(v: object) -> float:
        x = float(v)
        return x if not math.isnan(x) else float("-inf")

    rows = sorted(
        rows,
        key=lambda r: (
            float(r.get("support_samples", 0)),
            _score_or_neginf(r.get("mean_score_drop", float("nan"))),
            _score_or_neginf(r.get("mean_token_weight", float("nan"))),
            str(r.get("motif_key", "")),
        ),
        reverse=True,
    )
    if int(top_k) > 0:
        rows = rows[: int(top_k)]
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def merge_attribution_into_overall(
    overall_rows: Sequence[Dict[str, object]],
    attribution_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    pmap = {int(r["item_id"]): r for r in attribution_rows}
    out: List[Dict[str, object]] = []
    for r in overall_rows:
        rr = dict(r)
        pid = int(rr["item_id"])
        pr = pmap.get(pid)
        if pr is not None:
            for k, v in pr.items():
                if k not in {"item_id", "item_name"}:
                    rr[k] = v
        out.append(rr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["actionrec", "retrieval"], required=True)
    ap.add_argument("--hml_root", type=str, required=True)
    ap.add_argument("--token_root", type=str, required=True)
    ap.add_argument("--pretrained_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--top_k_ids", type=int, default=50)
    ap.add_argument("--top_k_types", type=int, default=16)
    ap.add_argument("--id_type_mode", type=str, default="bucket", choices=["none", "bucket", "modulo"])
    ap.add_argument("--id_type_bucket_size", type=int, default=8192)
    ap.add_argument("--id_type_modulo", type=int, default=4)
    ap.add_argument("--attr_top_k", type=int, default=0,
                    help="Top-K IDs from direct attribution to force-include in id_class_stats/top_ids. <=0 uses --top_k_ids.")
    ap.add_argument("--attr_score_target", type=str, default="",
                    help="actionrec: true_logit/neg_ce, retrieval: diag_cosine/neg_nce. empty uses task default.")

    ap.add_argument("--motiongpt_root", type=str, default="")
    ap.add_argument("--max_text_len", type=int, default=20)
    ap.add_argument("--perturb_top_k_ids", type=int, default=50, help="Top-K globally important IDs for single-ID deletion test.")
    ap.add_argument("--curve_top_p_values", type=str, default="0.1,0.2,0.3,0.5,0.7,0.9,1.0")
    ap.add_argument("--curve_top_k_values", type=str, default="1,2,3,5,10,20")
    ap.add_argument("--motif_min_len", type=int, default=2)
    ap.add_argument("--motif_max_len", type=int, default=6)
    ap.add_argument("--motif_mining_method", type=str, default="attr_contiguous",
                    choices=["attr_contiguous", "frequent_contiguous"],
                    help="How to generate motif candidates before perturbation scoring.")
    ap.add_argument("--motif_candidate_top_k", type=int, default=500,
                    help="Top-K motif candidates before global perturbation scoring.")
    ap.add_argument("--motif_top_k_per_sample", type=int, default=5)
    ap.add_argument("--motif_min_mean_weight", type=float, default=0.02)
    ap.add_argument("--motif_drop_all_occurrences", type=int, default=1, choices=[0, 1])
    ap.add_argument("--motif_score_target", type=str, default="",
                    help="actionrec: true_logit/neg_ce, retrieval: diag_cosine (single-sample validation).")
    ap.add_argument("--motif_min_support", type=int, default=2)
    ap.add_argument("--motif_summary_top_k", type=int, default=200)
    ap.add_argument("--motif_perturb_top_k", type=int, default=50,
                    help="Top-K recurring motifs (from motif summary) for global perturbation.")
    ap.add_argument("--motif_set_top_k_values", type=str, default="1,2,3,5,10,20",
                    help="Top-k values for cumulative motif-set perturbation (drop top-k motifs together).")
    ap.add_argument("--motif_chain_enable", type=int, default=1, choices=[0, 1])
    ap.add_argument("--motif_chain_top_k_motifs", type=int, default=40,
                    help="Use top-k motifs from motif summary as chain candidate pool.")
    ap.add_argument("--motif_chain_max_len", type=int, default=8)
    ap.add_argument("--motif_chain_min_gap", type=int, default=0)
    ap.add_argument("--motif_chain_max_gap", type=int, default=8)
    ap.add_argument("--motif_chain_min_delta", type=float, default=0.0,
                    help="Minimum incremental primary_metric_drop gain to keep greedy chain expansion.")
    ap.add_argument("--sparse_seq_enable", type=int, default=1, choices=[0, 1])
    ap.add_argument("--sparse_seq_select_mode", type=str, default="top_p_mass", choices=["top_p_mass", "top_k_ids"])
    ap.add_argument("--sparse_seq_top_p_mass", type=float, default=0.3)
    ap.add_argument("--sparse_seq_top_k_ids", type=int, default=5)
    ap.add_argument("--sparse_seq_min_len", type=int, default=2)
    ap.add_argument("--sparse_seq_max_len", type=int, default=12,
                    help="<=0 means no max length limit.")
    ap.add_argument("--sparse_seq_dedup_consecutive", type=int, default=1, choices=[0, 1])
    ap.add_argument("--sparse_seq_similarity_threshold", type=float, default=0.65)
    ap.add_argument("--sparse_seq_similarity_jaccard_mix", type=float, default=0.5,
                    help="Similarity = a*weighted_jaccard + (1-a)*normalized_lcs")
    ap.add_argument("--sparse_seq_cluster_top_k", type=int, default=50)
    ap.add_argument("--sparse_seq_chain_min_gap", type=int, default=0)
    ap.add_argument("--sparse_seq_chain_max_gap", type=int, default=16)
    ap.add_argument("--sparse_seq_drop_all_occurrences", type=int, default=1, choices=[0, 1])

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    type_mapper = IdTypeMapper(args.id_type_mode, args.id_type_bucket_size, args.id_type_modulo)
    curve_top_p_values = _parse_float_list(args.curve_top_p_values)
    curve_top_k_values = _parse_int_list(args.curve_top_k_values)
    motif_set_top_k_values = _parse_int_list(args.motif_set_top_k_values)

    all_samples, labeled_samples, dataset_key = load_samples(
        args.hml_root, args.token_root, args.split, args.max_tokens
    )
    class_names = list(ar.CLASS_NAMES)
    n_classes = len(class_names)

    run_samples: List[Sample]
    if args.task == "actionrec":
        run_samples = labeled_samples
    else:
        run_samples = all_samples

    if not run_samples:
        raise RuntimeError("no valid samples to evaluate")

    attr_id_rows: List[Dict[str, object]] = []
    attr_sample_rows: List[Dict[str, object]] = []
    perturb_rows: List[Dict[str, object]] = []
    budget_curve_rows: List[Dict[str, object]] = []
    attr_coverage_per_sample_rows: List[Dict[str, object]] = []
    attr_coverage_summary_rows: List[Dict[str, object]] = []
    motif_per_sample_rows: List[Dict[str, object]] = []
    motif_perturb_per_sample_rows: List[Dict[str, object]] = []
    motif_perturb_global_rows: List[Dict[str, object]] = []
    motif_set_perturb_rows: List[Dict[str, object]] = []
    motif_chain_perturb_rows: List[Dict[str, object]] = []
    motif_summary_rows: List[Dict[str, object]] = []
    sparse_seq_per_sample_rows: List[Dict[str, object]] = []
    sparse_seq_cluster_rows: List[Dict[str, object]] = []
    sparse_seq_cluster_perturb_rows: List[Dict[str, object]] = []
    motif_score_target_used = ""
    base_metrics: Dict[str, object]

    if args.task == "actionrec":
        model = load_actionrec_model(
            ckpt_path=Path(args.pretrained_ckpt),
            vocab_size=args.vocab_size,
            n_classes=n_classes,
            max_tokens=args.max_tokens,
            device=device,
        )
        base = eval_actionrec_samples(
            model=model,
            samples=run_samples,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=device,
            n_classes=n_classes,
        )
        base_metrics = {"task": "actionrec", **base}
        attr_top_k = int(args.attr_top_k) if int(args.attr_top_k) > 0 else int(args.top_k_ids)
        attr_score_target = str(args.attr_score_target).strip() or "true_logit"
        attr_id_rows, attr_sample_rows = compute_actionrec_id_attribution(
            model=model,
            samples=run_samples,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=device,
            score_target=attr_score_target,
            top_k=attr_top_k,
            class_names=class_names,
        )
        attr_coverage_per_sample_rows, attr_coverage_summary_rows = compute_attr_coverage_rows(attr_sample_rows)
        if int(args.sparse_seq_enable) == 1:
            sparse_seq_per_sample_rows, sparse_candidates = build_sparse_sequences_per_sample(
                samples=run_samples,
                per_sample_rows=attr_sample_rows,
                select_mode=str(args.sparse_seq_select_mode),
                top_p_mass=float(args.sparse_seq_top_p_mass),
                top_k_ids=int(args.sparse_seq_top_k_ids),
                min_len=int(args.sparse_seq_min_len),
                max_len=int(args.sparse_seq_max_len),
                dedup_consecutive=bool(int(args.sparse_seq_dedup_consecutive)),
            )
            sparse_seq_cluster_rows, sparse_clusters = cluster_sparse_sequences(
                candidates=sparse_candidates,
                similarity_threshold=float(args.sparse_seq_similarity_threshold),
                jaccard_mix=float(args.sparse_seq_similarity_jaccard_mix),
            )
            sparse_seq_cluster_perturb_rows = compute_actionrec_sparse_sequence_cluster_perturbation(
                model=model,
                samples=run_samples,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                device=device,
                n_classes=n_classes,
                base_metrics=base,
                clusters=sparse_clusters,
                top_k_clusters=int(args.sparse_seq_cluster_top_k),
                min_gap=int(args.sparse_seq_chain_min_gap),
                max_gap=int(args.sparse_seq_chain_max_gap),
                drop_all_occurrences=bool(int(args.sparse_seq_drop_all_occurrences)),
            )
        if str(args.motif_mining_method) == "frequent_contiguous":
            motif_candidates = mine_frequent_contiguous_motifs(
                samples=run_samples,
                min_len=args.motif_min_len,
                max_len=args.motif_max_len,
                min_support=args.motif_min_support,
                top_k=args.motif_candidate_top_k,
            )
            motif_score_target_used = "global_primary_metric"
            motif_perturb_global_rows = compute_actionrec_motif_perturbation(
                model=model,
                samples=run_samples,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                device=device,
                n_classes=n_classes,
                base_metrics=base,
                motif_rows=motif_candidates,
                top_k_motifs=args.motif_perturb_top_k,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
            motif_summary_rows = build_motif_summary_from_global_perturbation(
                candidate_rows=motif_candidates,
                global_rows=motif_perturb_global_rows,
            )
        else:
            motif_per_sample_rows = extract_sample_motifs(
                samples=run_samples,
                per_sample_rows=attr_sample_rows,
                class_names=class_names,
                min_len=args.motif_min_len,
                max_len=args.motif_max_len,
                top_k_per_sample=args.motif_top_k_per_sample,
                min_mean_weight=args.motif_min_mean_weight,
            )
            motif_score_target_used = str(args.motif_score_target).strip() or "true_logit"
            motif_perturb_per_sample_rows = validate_actionrec_motifs_with_perturbation(
                model=model,
                samples=run_samples,
                motif_rows=motif_per_sample_rows,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                device=device,
                score_target=motif_score_target_used,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
            motif_summary_rows = summarize_recurring_motifs(
                motif_rows=motif_per_sample_rows,
                motif_perturb_rows=motif_perturb_per_sample_rows,
                min_support=args.motif_min_support,
                top_k=args.motif_summary_top_k,
            )
            motif_perturb_global_rows = compute_actionrec_motif_perturbation(
                model=model,
                samples=run_samples,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                device=device,
                n_classes=n_classes,
                base_metrics=base,
                motif_rows=motif_summary_rows,
                top_k_motifs=args.motif_perturb_top_k,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
        motif_set_perturb_rows = compute_actionrec_motif_set_perturbation(
            model=model,
            samples=run_samples,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=device,
            n_classes=n_classes,
            base_metrics=base,
            motif_rows=motif_summary_rows,
            top_k_values=motif_set_top_k_values,
            drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
        )
        if int(args.motif_chain_enable) == 1:
            motif_chain_perturb_rows = compute_actionrec_motif_chain_greedy(
                model=model,
                samples=run_samples,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                device=device,
                n_classes=n_classes,
                base_metrics=base,
                motif_rows=motif_summary_rows,
                top_k_motifs=args.motif_chain_top_k_motifs,
                max_chain_len=args.motif_chain_max_len,
                min_gap=args.motif_chain_min_gap,
                max_gap=args.motif_chain_max_gap,
                min_delta=args.motif_chain_min_delta,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
        perturb_rows = compute_actionrec_id_perturbation(
            model=model,
            samples=run_samples,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=device,
            n_classes=n_classes,
            base_metrics=base,
            attr_rows=attr_id_rows,
            top_k_ids=args.perturb_top_k_ids,
        )
        budget_curve_rows = compute_actionrec_budget_curves(
            model=model,
            samples=run_samples,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=device,
            n_classes=n_classes,
            base_metrics=base,
            per_sample_rows=attr_sample_rows,
            top_p_values=curve_top_p_values,
            top_k_values=curve_top_k_values,
        )
        forced_ids = [int(r["item_id"]) for r in attr_id_rows]
        stats = collect_stats(
            labeled_samples=labeled_samples,
            class_names=class_names,
            top_k_ids=args.top_k_ids,
            top_k_types=args.top_k_types,
            type_mapper=type_mapper,
            forced_ids=forced_ids,
        )

    else:
        if not str(args.motiongpt_root).strip():
            raise ValueError("--motiongpt_root is required for retrieval task")

        model, _cargs = load_retrieval_model(
            ckpt_path=Path(args.pretrained_ckpt),
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            device=device,
        )
        text_emb = encode_text_embeddings(
            model=model,
            samples=run_samples,
            motiongpt_root=args.motiongpt_root,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
            device=device,
        )
        base = evaluate_retrieval(
            model=model,
            samples=run_samples,
            motiongpt_root=args.motiongpt_root,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
            device=device,
            cached_text_emb=text_emb,
        )
        base_metrics = {"task": "retrieval", **base}
        attr_top_k = int(args.attr_top_k) if int(args.attr_top_k) > 0 else int(args.top_k_ids)
        attr_score_target = str(args.attr_score_target).strip() or "diag_cosine"
        attr_id_rows, attr_sample_rows = compute_retrieval_id_attribution(
            model=model,
            samples=run_samples,
            motiongpt_root=args.motiongpt_root,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
            device=device,
            score_target=attr_score_target,
            top_k=attr_top_k,
            class_names=class_names,
        )
        attr_coverage_per_sample_rows, attr_coverage_summary_rows = compute_attr_coverage_rows(attr_sample_rows)
        if int(args.sparse_seq_enable) == 1:
            sparse_seq_per_sample_rows, sparse_candidates = build_sparse_sequences_per_sample(
                samples=run_samples,
                per_sample_rows=attr_sample_rows,
                select_mode=str(args.sparse_seq_select_mode),
                top_p_mass=float(args.sparse_seq_top_p_mass),
                top_k_ids=int(args.sparse_seq_top_k_ids),
                min_len=int(args.sparse_seq_min_len),
                max_len=int(args.sparse_seq_max_len),
                dedup_consecutive=bool(int(args.sparse_seq_dedup_consecutive)),
            )
            sparse_seq_cluster_rows, sparse_clusters = cluster_sparse_sequences(
                candidates=sparse_candidates,
                similarity_threshold=float(args.sparse_seq_similarity_threshold),
                jaccard_mix=float(args.sparse_seq_similarity_jaccard_mix),
            )
            sparse_seq_cluster_perturb_rows = compute_retrieval_sparse_sequence_cluster_perturbation(
                model=model,
                samples=run_samples,
                motiongpt_root=args.motiongpt_root,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                max_text_len=args.max_text_len,
                batch_size=args.batch_size,
                device=device,
                base_metrics=base,
                cached_text_emb=text_emb,
                clusters=sparse_clusters,
                top_k_clusters=int(args.sparse_seq_cluster_top_k),
                min_gap=int(args.sparse_seq_chain_min_gap),
                max_gap=int(args.sparse_seq_chain_max_gap),
                drop_all_occurrences=bool(int(args.sparse_seq_drop_all_occurrences)),
            )
        if str(args.motif_mining_method) == "frequent_contiguous":
            motif_candidates = mine_frequent_contiguous_motifs(
                samples=run_samples,
                min_len=args.motif_min_len,
                max_len=args.motif_max_len,
                min_support=args.motif_min_support,
                top_k=args.motif_candidate_top_k,
            )
            motif_score_target_used = "global_primary_metric"
            motif_perturb_global_rows = compute_retrieval_motif_perturbation(
                model=model,
                samples=run_samples,
                motiongpt_root=args.motiongpt_root,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                max_text_len=args.max_text_len,
                batch_size=args.batch_size,
                device=device,
                base_metrics=base,
                cached_text_emb=text_emb,
                motif_rows=motif_candidates,
                top_k_motifs=args.motif_perturb_top_k,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
            motif_summary_rows = build_motif_summary_from_global_perturbation(
                candidate_rows=motif_candidates,
                global_rows=motif_perturb_global_rows,
            )
        else:
            motif_per_sample_rows = extract_sample_motifs(
                samples=run_samples,
                per_sample_rows=attr_sample_rows,
                class_names=class_names,
                min_len=args.motif_min_len,
                max_len=args.motif_max_len,
                top_k_per_sample=args.motif_top_k_per_sample,
                min_mean_weight=args.motif_min_mean_weight,
            )
            motif_score_target_used = "diag_cosine"
            wvec = load_word_vectorizer(Path(args.motiongpt_root))
            motif_perturb_per_sample_rows = validate_retrieval_motifs_with_perturbation(
                model=model,
                samples=run_samples,
                motif_rows=motif_per_sample_rows,
                wvec=wvec,
                max_text_len=args.max_text_len,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                device=device,
                score_target=motif_score_target_used,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
            motif_summary_rows = summarize_recurring_motifs(
                motif_rows=motif_per_sample_rows,
                motif_perturb_rows=motif_perturb_per_sample_rows,
                min_support=args.motif_min_support,
                top_k=args.motif_summary_top_k,
            )
            motif_perturb_global_rows = compute_retrieval_motif_perturbation(
                model=model,
                samples=run_samples,
                motiongpt_root=args.motiongpt_root,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                max_text_len=args.max_text_len,
                batch_size=args.batch_size,
                device=device,
                base_metrics=base,
                cached_text_emb=text_emb,
                motif_rows=motif_summary_rows,
                top_k_motifs=args.motif_perturb_top_k,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
        motif_set_perturb_rows = compute_retrieval_motif_set_perturbation(
            model=model,
            samples=run_samples,
            motiongpt_root=args.motiongpt_root,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
            device=device,
            base_metrics=base,
            cached_text_emb=text_emb,
            motif_rows=motif_summary_rows,
            top_k_values=motif_set_top_k_values,
            drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
        )
        if int(args.motif_chain_enable) == 1:
            motif_chain_perturb_rows = compute_retrieval_motif_chain_greedy(
                model=model,
                samples=run_samples,
                motiongpt_root=args.motiongpt_root,
                vocab_size=args.vocab_size,
                max_tokens=args.max_tokens,
                max_text_len=args.max_text_len,
                batch_size=args.batch_size,
                device=device,
                base_metrics=base,
                cached_text_emb=text_emb,
                motif_rows=motif_summary_rows,
                top_k_motifs=args.motif_chain_top_k_motifs,
                max_chain_len=args.motif_chain_max_len,
                min_gap=args.motif_chain_min_gap,
                max_gap=args.motif_chain_max_gap,
                min_delta=args.motif_chain_min_delta,
                drop_all_occurrences=bool(int(args.motif_drop_all_occurrences)),
            )
        perturb_rows = compute_retrieval_id_perturbation(
            model=model,
            samples=run_samples,
            motiongpt_root=args.motiongpt_root,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
            device=device,
            base_metrics=base,
            cached_text_emb=text_emb,
            attr_rows=attr_id_rows,
            top_k_ids=args.perturb_top_k_ids,
        )
        budget_curve_rows = compute_retrieval_budget_curves(
            model=model,
            samples=run_samples,
            motiongpt_root=args.motiongpt_root,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            max_text_len=args.max_text_len,
            batch_size=args.batch_size,
            device=device,
            base_metrics=base,
            cached_text_emb=text_emb,
            per_sample_rows=attr_sample_rows,
            top_p_values=curve_top_p_values,
            top_k_values=curve_top_k_values,
        )
        forced_ids = [int(r["item_id"]) for r in attr_id_rows]
        stats = collect_stats(
            labeled_samples=labeled_samples,
            class_names=class_names,
            top_k_ids=args.top_k_ids,
            top_k_types=args.top_k_types,
            type_mapper=type_mapper,
            forced_ids=forced_ids,
        )


    id_overall_merged = merge_attribution_into_overall(stats["id_overall_rows"], attr_id_rows)
    tp_overall_merged = list(stats["type_overall_rows"])

    write_csv(out_dir / "id_overall.csv", id_overall_merged)
    write_csv(out_dir / "id_class_stats.csv", stats["id_class_rows"])
    write_csv(out_dir / "id_type_overall.csv", tp_overall_merged)
    write_csv(out_dir / "id_type_class_stats.csv", stats["type_class_rows"])
    write_csv(out_dir / "id_attribution.csv", attr_id_rows)
    write_csv(out_dir / "id_attribution_per_sample.csv", attr_sample_rows)
    write_csv(out_dir / "id_perturbation.csv", perturb_rows)
    write_csv(out_dir / "id_budget_curves.csv", budget_curve_rows)
    write_csv(out_dir / "id_attr_coverage_per_sample.csv", attr_coverage_per_sample_rows)
    write_csv(out_dir / "id_attr_coverage_summary.csv", attr_coverage_summary_rows)
    write_csv(out_dir / "id_sparse_sequences_per_sample.csv", sparse_seq_per_sample_rows)
    write_csv(out_dir / "id_sparse_sequence_clusters.csv", sparse_seq_cluster_rows)
    write_csv(out_dir / "id_sparse_sequence_cluster_perturbation.csv", sparse_seq_cluster_perturb_rows)
    write_csv(out_dir / "id_sequence_motifs_per_sample.csv", motif_per_sample_rows)
    write_csv(out_dir / "id_sequence_motif_perturbation_per_sample.csv", motif_perturb_per_sample_rows)
    write_csv(out_dir / "id_sequence_motif_perturbation.csv", motif_perturb_global_rows)
    write_csv(out_dir / "id_sequence_motif_set_perturbation.csv", motif_set_perturb_rows)
    write_csv(out_dir / "id_sequence_motif_chain_perturbation.csv", motif_chain_perturb_rows)
    write_csv(out_dir / "id_sequence_motif_summary.csv", motif_summary_rows)

    summary = {
        "task": args.task,
        "dataset_key": dataset_key,
        "split": args.split,
        "hml_root": args.hml_root,
        "token_root": args.token_root,
        "pretrained_ckpt": args.pretrained_ckpt,
        "vocab_size": int(args.vocab_size),
        "max_tokens": int(args.max_tokens),
        "id_type_mode": args.id_type_mode,
        "id_type_bucket_size": int(args.id_type_bucket_size),
        "id_type_modulo": int(args.id_type_modulo),
        "attr_top_k": int(args.attr_top_k),
        "attr_score_target": str(args.attr_score_target),
        "perturb_top_k_ids": int(args.perturb_top_k_ids),
        "curve_top_p_values": curve_top_p_values,
        "curve_top_k_values": curve_top_k_values,
        "motif_mining_method": str(args.motif_mining_method),
        "motif_candidate_top_k": int(args.motif_candidate_top_k),
        "motif_min_len": int(args.motif_min_len),
        "motif_max_len": int(args.motif_max_len),
        "motif_top_k_per_sample": int(args.motif_top_k_per_sample),
        "motif_min_mean_weight": float(args.motif_min_mean_weight),
        "motif_drop_all_occurrences": int(args.motif_drop_all_occurrences),
        "motif_score_target": motif_score_target_used,
        "motif_min_support": int(args.motif_min_support),
        "motif_summary_top_k": int(args.motif_summary_top_k),
        "motif_perturb_top_k": int(args.motif_perturb_top_k),
        "motif_set_top_k_values": motif_set_top_k_values,
        "motif_chain_enable": int(args.motif_chain_enable),
        "motif_chain_top_k_motifs": int(args.motif_chain_top_k_motifs),
        "motif_chain_max_len": int(args.motif_chain_max_len),
        "motif_chain_min_gap": int(args.motif_chain_min_gap),
        "motif_chain_max_gap": int(args.motif_chain_max_gap),
        "motif_chain_min_delta": float(args.motif_chain_min_delta),
        "sparse_seq_enable": int(args.sparse_seq_enable),
        "sparse_seq_select_mode": str(args.sparse_seq_select_mode),
        "sparse_seq_top_p_mass": float(args.sparse_seq_top_p_mass),
        "sparse_seq_top_k_ids": int(args.sparse_seq_top_k_ids),
        "sparse_seq_min_len": int(args.sparse_seq_min_len),
        "sparse_seq_max_len": int(args.sparse_seq_max_len),
        "sparse_seq_dedup_consecutive": int(args.sparse_seq_dedup_consecutive),
        "sparse_seq_similarity_threshold": float(args.sparse_seq_similarity_threshold),
        "sparse_seq_similarity_jaccard_mix": float(args.sparse_seq_similarity_jaccard_mix),
        "sparse_seq_cluster_top_k": int(args.sparse_seq_cluster_top_k),
        "sparse_seq_chain_min_gap": int(args.sparse_seq_chain_min_gap),
        "sparse_seq_chain_max_gap": int(args.sparse_seq_chain_max_gap),
        "sparse_seq_drop_all_occurrences": int(args.sparse_seq_drop_all_occurrences),
        "n_samples_all": len(all_samples),
        "n_samples_labeled": len(labeled_samples),
        "n_samples_eval": len(run_samples),
        "n_classes": n_classes,
        "classes": class_names,
        "base_metrics": base_metrics,
        "top_ids": stats["top_ids"],
        "top_types": stats["top_types"],
        "files": {
            "id_overall": str(out_dir / "id_overall.csv"),
            "id_class_stats": str(out_dir / "id_class_stats.csv"),
            "id_type_overall": str(out_dir / "id_type_overall.csv"),
            "id_type_class_stats": str(out_dir / "id_type_class_stats.csv"),
            "id_attribution": str(out_dir / "id_attribution.csv"),
            "id_attribution_per_sample": str(out_dir / "id_attribution_per_sample.csv"),
            "id_perturbation": str(out_dir / "id_perturbation.csv"),
            "id_budget_curves": str(out_dir / "id_budget_curves.csv"),
            "id_attr_coverage_per_sample": str(out_dir / "id_attr_coverage_per_sample.csv"),
            "id_attr_coverage_summary": str(out_dir / "id_attr_coverage_summary.csv"),
            "id_sparse_sequences_per_sample": str(out_dir / "id_sparse_sequences_per_sample.csv"),
            "id_sparse_sequence_clusters": str(out_dir / "id_sparse_sequence_clusters.csv"),
            "id_sparse_sequence_cluster_perturbation": str(out_dir / "id_sparse_sequence_cluster_perturbation.csv"),
            "id_sequence_motifs_per_sample": str(out_dir / "id_sequence_motifs_per_sample.csv"),
            "id_sequence_motif_perturbation": str(out_dir / "id_sequence_motif_perturbation.csv"),
            "id_sequence_motif_perturbation_per_sample": str(out_dir / "id_sequence_motif_perturbation_per_sample.csv"),
            "id_sequence_motif_set_perturbation": str(out_dir / "id_sequence_motif_set_perturbation.csv"),
            "id_sequence_motif_chain_perturbation": str(out_dir / "id_sequence_motif_chain_perturbation.csv"),
            "id_sequence_motif_summary": str(out_dir / "id_sequence_motif_summary.csv"),
        },
    }
    (out_dir / "id_contrib_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
