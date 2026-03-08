#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute per-ID within-style / between-style statistics from token list files.

Modes:
  - variance: variance-based decomposition (existing behavior)
  - distance: distance-based decomposition

Example:
  python bandai_id_variance_within_between_style.py ^
    --root fig/Bandai-Namco-Research-Motiondataset-1 ^
    --out out_id_var ^
    --binary 0 ^
    --var-normalization id_mean ^
    --score-mode distance
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


# -----------------------------
# parse / load
# -----------------------------
def parse_action_style_from_filename(filename: str) -> Tuple[str, str]:
    # dataset-1_{action}_{style}_{index}.txt
    stem = Path(filename).stem
    if not stem.startswith("dataset-1_"):
        raise ValueError(filename)
    core = stem[len("dataset-1_") :]
    parts = core.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(filename)
    action, style, index = parts
    if not action or not style or not index.isdigit():
        raise ValueError(filename)
    return action, style


def discover_latent_lists(root: Path) -> List[Tuple[str, str, Path]]:
    out = []
    for p in root.rglob("*.txt"):
        try:
            a, s = parse_action_style_from_filename(p.name)
        except ValueError:
            continue
        out.append((a, s, p))
    return sorted(out, key=lambda x: (x[0], x[1], str(x[2])))


def read_id_column(latent_list_path: Path) -> List[str]:
    """
    Parse first column as ID.
    - Skip empty lines and comment lines starting with '#'
    - Accept comma-separated or whitespace-separated formats
    - Skip a header-like first token such as 'id' / 'latent_id'
    """
    ids: List[str] = []
    lines = latent_list_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        if "," in s:
            first = s.split(",", 1)[0].strip()
        else:
            first = s.split(None, 1)[0].strip()

        if not ids:
            low = first.lower()
            if low in {"id", "ids", "latent_id", "latentid"}:
                continue
            if low.startswith("id:"):
                first = first.split(":", 1)[1].strip()

        if first:
            ids.append(first)
    return ids


# -----------------------------
# build sparse probability matrix P (groups x IDs)
# -----------------------------
def build_sparse_counts(
    latent_lists: List[Tuple[str, str, Path]],
    binary: bool,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    returns:
      C: (n_groups, n_ids) counts (int)
      actions: (n_groups,)
      styles: (n_groups,)
      id_vocab: list[str] (col -> id)
      denom: (n_groups,) denominator used for probability (total count or unique count)
    """
    vocab: Dict[str, int] = {}
    id_vocab: List[str] = []

    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []

    actions = []
    styles = []
    denom = []

    for gi, (action, style, path) in enumerate(latent_lists):
        ids = read_id_column(path)
        if binary:
            ids_use = list(set(ids))
            c = Counter(ids_use)
            denom.append(len(ids_use))
        else:
            c = Counter(ids)
            denom.append(len(ids))

        actions.append(action)
        styles.append(style)

        for _id, cnt in c.items():
            _id = str(_id)
            j = vocab.get(_id)
            if j is None:
                j = len(id_vocab)
                vocab[_id] = j
                id_vocab.append(_id)
            rows.append(gi)
            cols.append(j)
            data.append(int(cnt if not binary else 1))

    n_groups = len(latent_lists)
    n_ids = len(id_vocab)
    C = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.int32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(n_groups, n_ids),
        dtype=np.float64,
    )

    actions = np.asarray(actions, dtype=str)
    styles = np.asarray(styles, dtype=str)
    denom = np.asarray(denom, dtype=np.float64)
    denom = np.maximum(denom, 1.0)  # safety

    return C, actions, styles, id_vocab, denom


def make_style_indicator(styles: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    styles: (n_groups,)
    returns:
      S: (n_styles, n_groups) indicator
      uniq_styles: (n_styles,)
      style_idx: (n_groups,) code
    """
    style_idx, uniq_styles = pd.factorize(pd.Index(styles.astype(str)))
    style_idx = style_idx.astype(np.int64, copy=False)
    n_groups = len(styles)
    n_styles = len(uniq_styles)

    S = sp.csr_matrix(
        (np.ones(n_groups, dtype=np.float64), (style_idx, np.arange(n_groups, dtype=np.int64))),
        shape=(n_styles, n_groups),
    )
    return S, np.asarray(uniq_styles, dtype=str), style_idx


def build_probability_matrix(
    C: sp.csr_matrix,
    denom: np.ndarray,
    mode: str,
    eps: float = 1e-12,
) -> sp.csr_matrix:
    """
    Build probability matrix P (groups x IDs).

    mode:
      - per_file:               P = C / len(latent_list.txt)
      - per_file_then_id_l1:    per_file after that each ID column sums to 1 across groups
    """
    P = (sp.diags(1.0 / np.maximum(denom, 1.0)) @ C).tocsr()

    if mode == "per_file":
        return P
    if mode == "per_file_then_id_l1":
        col_sum = np.asarray(P.sum(axis=0)).ravel()
        inv = 1.0 / np.maximum(col_sum, eps)
        return (P @ sp.diags(inv)).tocsr()

    raise ValueError(f"Unsupported probability mode: {mode}")


def normalize_probability_for_score(
    P: sp.csr_matrix,
    mode: str,
    eps: float = 1e-12,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Prepare matrix used for score computation.

    mode:
      - none: keep original probabilities
      - id_mean: divide each ID column by its global mean probability
    returns:
      P_score: matrix used in computation
      mean_prob_overall: original global mean probability per ID
    """
    mean_prob_overall = np.asarray(P.mean(axis=0)).ravel()

    if mode == "none":
        return P, mean_prob_overall
    if mode == "id_mean":
        inv = 1.0 / np.maximum(mean_prob_overall, eps)
        P_score = (P @ sp.diags(inv)).tocsr()
        return P_score, mean_prob_overall

    raise ValueError(f"Unsupported var-normalization mode: {mode}")


# -----------------------------
# score computations
# -----------------------------
def compute_variance_scores(P: sp.csr_matrix, styles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Variance-based within/between scores.

    returns:
      within_score: (n_ids,)
      between_score: (n_ids,)
      mean_overall: (n_ids,)
      n_g_per_style: (n_styles,)
    """
    n_groups, n_ids = P.shape
    S, uniq_styles, style_idx = make_style_indicator(styles)
    n_styles = len(uniq_styles)

    n_g_per_style = np.asarray(S.sum(axis=1)).ravel()
    n_g_per_style = np.maximum(n_g_per_style, 1.0)
    Dinv = sp.diags(1.0 / n_g_per_style)

    sum_s = (S @ P).tocsr()
    mean_s = (Dinv @ sum_s).tocsr()

    P2 = P.multiply(P)
    sum2_s = (S @ P2).tocsr()
    mean2_s = (Dinv @ sum2_s).tocsr()

    var_s = (mean2_s - mean_s.multiply(mean_s)).tocsr()

    within_score = np.asarray(var_s.sum(axis=0)).ravel() / float(n_styles)

    Em = np.asarray(mean_s.sum(axis=0)).ravel() / float(n_styles)
    Em2 = np.asarray(mean_s.multiply(mean_s).sum(axis=0)).ravel() / float(n_styles)
    between_score = Em2 - Em * Em
    between_score = np.maximum(between_score, 0.0)

    mean_overall = Em
    return within_score, between_score, mean_overall, n_g_per_style


def mean_pairwise_abs_1d(x: np.ndarray) -> float:
    """mean_{i<j} |x_i - x_j| for 1D vector x."""
    n = int(x.size)
    if n < 2:
        return 0.0
    xs = np.sort(x.astype(np.float64, copy=False))
    coeff = (2.0 * np.arange(n, dtype=np.float64)) - float(n) + 1.0
    total_abs = float(np.dot(coeff, xs))
    return total_abs / (n * (n - 1) / 2.0)


def mean_cross_abs_1d(a: np.ndarray, b: np.ndarray) -> float:
    """mean_{i,j} |a_i - b_j| for 1D vectors a and b."""
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.abs(a[:, None] - b[None, :]).mean())


def compute_distance_scores(P: sp.csr_matrix, styles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Distance-based within/between scores with d(x,y)=|x-y| for scalar probabilities.

    within(style)   = mean_{i<j} d(a_i, a_j)
    between(A, B)   = mean_{i,j} d(a_i, b_j)

    returns:
      within_score: mean of within(style) over styles, per ID
      between_score: mean of between(style_a, style_b) over style pairs, per ID
      mean_overall: global mean over groups, per ID
      n_g_per_style: number of groups in each style
    """
    n_groups, n_ids = P.shape
    style_idx, uniq_styles = pd.factorize(pd.Index(styles.astype(str)))
    style_idx = style_idx.astype(np.int64, copy=False)
    n_styles = len(uniq_styles)

    style_rows = [np.where(style_idx == si)[0] for si in range(n_styles)]
    n_g_per_style = np.asarray([len(rows) for rows in style_rows], dtype=np.float64)

    style_pair_indices = [(i, j) for i in range(n_styles) for j in range(i + 1, n_styles)]
    n_style_pairs = len(style_pair_indices)

    within_score = np.zeros(n_ids, dtype=np.float64)
    between_score = np.zeros(n_ids, dtype=np.float64)

    P_csc = P.tocsc()
    p_dense = np.zeros(n_groups, dtype=np.float64)

    for j in range(n_ids):
        p_dense.fill(0.0)
        col = P_csc.getcol(j)
        if col.nnz > 0:
            p_dense[col.indices] = col.data

        w_sum = 0.0
        for rows in style_rows:
            w_sum += mean_pairwise_abs_1d(p_dense[rows])
        within_score[j] = w_sum / float(max(n_styles, 1))

        if n_style_pairs > 0:
            b_sum = 0.0
            for si, sj in style_pair_indices:
                b_sum += mean_cross_abs_1d(p_dense[style_rows[si]], p_dense[style_rows[sj]])
            between_score[j] = b_sum / float(n_style_pairs)
        else:
            between_score[j] = 0.0

    mean_overall = np.asarray(P.mean(axis=0)).ravel()
    return within_score, between_score, mean_overall, n_g_per_style


# -----------------------------
# plotting
# -----------------------------
def plot_scatter(
    df: pd.DataFrame,
    out_png: Path,
    score_mode: str,
    annotate_top: int = 0,
    eps: float = 1e-18,
) -> None:
    if 0:
        x = np.log10(df["within_score"].to_numpy(dtype=float) + eps)
        y = np.log10(df["between_score"].to_numpy(dtype=float) + eps)
    else:
        x = df["within_score"].to_numpy(dtype=float)
        y = df["between_score"].to_numpy(dtype=float)

    plt.figure()
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel(f"log10(within_{score_mode} + eps)")
    plt.ylabel(f"log10(between_{score_mode} + eps)")
    plt.title(f"ID distribution: within vs between ({score_mode})")
    plt.tight_layout()

    if annotate_top and annotate_top > 0:
        top = df.sort_values("between_over_within", ascending=False).head(int(annotate_top))
        if 0:
            xt = np.log10(top["within_score"].to_numpy(dtype=float) + eps)
            yt = np.log10(top["between_score"].to_numpy(dtype=float) + eps)
        else:
            xt = top["within_score"].to_numpy(dtype=float)
            yt = top["between_score"].to_numpy(dtype=float)
        for (_id, xi, yi) in zip(top["id"].astype(str).tolist(), xt, yt):
            plt.text(xi, yi, _id, fontsize=7)

    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument(
        "--binary",
        type=int,
        default=0,
        help="1: binary occurrence per file, 0: count frequency",
    )
    ap.add_argument(
        "--probability-mode",
        type=str,
        default="per_file",
        choices=["per_file", "per_file_then_id_l1"],
        help="How to build probability matrix before score normalization",
    )
    ap.add_argument(
        "--var-normalization",
        type=str,
        default="none",
        choices=["none", "id_mean"],
        help="Normalization mode applied before score computation",
    )
    ap.add_argument(
        "--score-mode",
        type=str,
        default="variance",
        choices=["variance", "distance"],
        help="Use variance-based or distance-based within/between score",
    )

    ap.add_argument(
        "--min-total-count",
        type=int,
        default=10,
        help="Filter out IDs with global_count smaller than this value",
    )
    ap.add_argument(
        "--max-plot",
        type=int,
        default=50000,
        help="Maximum number of IDs in plot (take top by between_score)",
    )
    ap.add_argument(
        "--annotate-top",
        type=int,
        default=0,
        help="Annotate top IDs by between_over_within on the scatter plot",
    )

    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    latent_lists = discover_latent_lists(root)
    if not latent_lists:
        raise SystemExit(f"dataset-1_*_*.txt not found under: {root}")

    # counts -> probability
    C, actions, styles, id_vocab, denom = build_sparse_counts(latent_lists, binary=bool(args.binary))

    # probability matrix
    P = build_probability_matrix(C, denom, mode=str(args.probability_mode))

    # global count per ID
    global_count = np.asarray(C.sum(axis=0)).ravel()

    # optional normalization before score computation
    if str(args.probability_mode) == "per_file_then_id_l1":
        P_for_score = P
        mean_prob_overall = np.asarray(P.mean(axis=0)).ravel()
    else:
        P_for_score, mean_prob_overall = normalize_probability_for_score(
            P,
            mode=str(args.var_normalization),
        )

    if args.score_mode == "variance":
        within_score, between_score, mean_overall_for_score, n_g_per_style = compute_variance_scores(P_for_score, styles)
    else:
        within_score, between_score, mean_overall_for_score, n_g_per_style = compute_distance_scores(P_for_score, styles)

    # dataframe
    df = pd.DataFrame(
        {
            "id": id_vocab,
            "global_count": global_count.astype(np.int64),
            "mean_prob_overall": mean_prob_overall,
            "mean_prob_overall_for_score": mean_overall_for_score,
            "within_score": within_score,
            "between_score": between_score,
            # backward-compatible aliases
            "within_var": within_score,
            "between_var": between_score,
            "between_over_within": between_score / (within_score + 1e-18),
            "within_over_between": within_score / (between_score + 1e-18),
            "max_ratio": np.maximum(
                between_score / (within_score + 1e-18),
                within_score / (between_score + 1e-18),
            ),
            "score_mode": str(args.score_mode),
            "probability_mode": str(args.probability_mode),
            "var_normalization": str(args.var_normalization),
        }
    )

    print(f"[INFO] total IDs before filtering by min_total_count={args.min_total_count}: {len(df)}")

    # filter rare IDs
    df = df[df["global_count"] >= int(args.min_total_count)].copy()

    # save metrics
    df.sort_values("between_score", ascending=False).to_csv(
        out / "id_within_between_style_variance.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"[INFO] total IDs after filtering by min_total_count={args.min_total_count}: {len(df)}")

    # plot subset (optional cap)
    df_plot = df
    if args.max_plot and len(df_plot) > int(args.max_plot):
        df_plot = df_plot.sort_values("between_score", ascending=False).head(int(args.max_plot)).copy()

    plot_scatter(
        df_plot,
        out / "plot_id_variance_scatter.png",
        score_mode=str(args.score_mode),
        annotate_top=int(args.annotate_top),
    )

    # also save manifest of groups (for traceability)
    manifest = pd.DataFrame(
        [{"action": a, "style": s, "latent_list_path": str(p)} for (a, s, p) in latent_lists]
    )
    manifest.to_csv(out / "groups_manifest.csv", index=False, encoding="utf-8")

    print(f"[DONE] wrote: {out / 'id_within_between_style_variance.csv'}")
    print(f"[DONE] wrote: {out / 'plot_id_variance_scatter.png'}")
    print(f"[DONE] wrote: {out / 'groups_manifest.csv'}")


if __name__ == "__main__":
    main()
