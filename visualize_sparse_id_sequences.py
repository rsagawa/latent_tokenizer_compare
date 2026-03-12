#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sparse_id_sequences.py

Visualize sparse high-importance ID sequence outputs produced by
analyze_retrieval_actionrec_id_contrib.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def _safe_float(v, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]


def _clip_label(s: str, limit: int = 40) -> str:
    t = str(s).strip()
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 3)] + "..."


def _prepare_top_cluster_rows(rows: List[Dict[str, str]], top_k: int) -> List[Dict[str, object]]:
    out = []
    for r in rows:
        out.append(
            {
                "cluster_id": _safe_int(r.get("cluster_id", 0)),
                "support_samples": _safe_int(r.get("support_samples", 0)),
                "importance_score": _safe_float(r.get("importance_score", float("nan"))),
                "mean_pair_similarity": _safe_float(r.get("mean_pair_similarity", float("nan"))),
                "representative_sparse_seq": str(r.get("representative_sparse_seq", "")),
            }
        )
    out = sorted(
        out,
        key=lambda x: (
            -float(x["importance_score"]) if not math.isnan(float(x["importance_score"])) else float("inf"),
            -int(x["support_samples"]),
            int(x["cluster_id"]),
        ),
    )
    if int(top_k) > 0:
        out = out[: int(top_k)]
    return out


def plot_top_cluster_importance(rows: List[Dict[str, object]], out_png: Path, title_prefix: str) -> None:
    if not rows:
        return
    labels = [
        f"C{int(r['cluster_id'])} | n={int(r['support_samples'])}\n{_clip_label(str(r['representative_sparse_seq']), 32)}"
        for r in rows
    ]
    vals = [float(r["importance_score"]) for r in rows]
    fig_h = max(4.0, 0.45 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = list(range(len(rows)))
    ax.barh(y, vals, color="#1f77b4", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("importance_score (primary_metric_drop)")
    ax.set_title(f"{title_prefix} Sparse Sequence Cluster Importance (Top)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_support_vs_importance(rows: List[Dict[str, object]], out_png: Path, title_prefix: str) -> None:
    if not rows:
        return
    xs = [float(r["support_samples"]) for r in rows]
    ys = [float(r["importance_score"]) for r in rows]
    cs = [float(r["mean_pair_similarity"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=50, alpha=0.9, edgecolors="black", linewidths=0.3)
    for r in rows[: min(20, len(rows))]:
        x = float(r["support_samples"])
        y = float(r["importance_score"])
        cid = int(r["cluster_id"])
        ax.annotate(f"C{cid}", (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("support_samples")
    ax.set_ylabel("importance_score (primary_metric_drop)")
    ax.set_title(f"{title_prefix} Sparse Cluster Support vs Importance")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("mean_pair_similarity")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_sparse_seq_len_vs_mass(rows: List[Dict[str, str]], out_png: Path, title_prefix: str) -> None:
    if not rows:
        return
    xs = [_safe_int(r.get("sparse_seq_len", 0)) for r in rows]
    ys = [_safe_float(r.get("token_weight_mass", float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, s=18, alpha=0.6, color="#2ca02c")
    ax.set_xlabel("sparse_seq_len")
    ax.set_ylabel("token_weight_mass")
    ax.set_title(f"{title_prefix} Sparse Sequence Length vs Weight Mass")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id_contrib_dir", type=str, required=True, help="directory containing id_contrib CSV outputs")
    ap.add_argument("--out_dir", type=str, default="", help="output dir for sparse visualizations (default: <id_contrib_dir>/sparse_viz)")
    ap.add_argument("--top_k_clusters", type=int, default=25)
    ap.add_argument("--title_prefix", type=str, default="")
    args = ap.parse_args()

    base_dir = Path(args.id_contrib_dir)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base_dir / "sparse_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = base_dir / "id_sparse_sequences_per_sample.csv"
    cluster_csv = base_dir / "id_sparse_sequence_clusters.csv"
    cluster_pert_csv = base_dir / "id_sparse_sequence_cluster_perturbation.csv"

    per_sample_rows = read_csv_rows(per_sample_csv)
    cluster_rows = read_csv_rows(cluster_csv)
    cluster_pert_rows = read_csv_rows(cluster_pert_csv)

    merged_rows_by_cluster: Dict[int, Dict[str, object]] = {}
    for r in cluster_rows:
        cid = _safe_int(r.get("cluster_id", 0))
        merged_rows_by_cluster[cid] = {
            "cluster_id": cid,
            "support_samples": _safe_int(r.get("support_samples", 0)),
            "mean_pair_similarity": _safe_float(r.get("mean_pair_similarity", float("nan"))),
            "representative_sparse_seq": str(r.get("representative_sparse_seq", "")),
            "importance_score": float("nan"),
        }
    for r in cluster_pert_rows:
        cid = _safe_int(r.get("cluster_id", 0))
        rec = merged_rows_by_cluster.setdefault(
            cid,
            {
                "cluster_id": cid,
                "support_samples": _safe_int(r.get("support_samples", 0)),
                "mean_pair_similarity": _safe_float(r.get("mean_pair_similarity", float("nan"))),
                "representative_sparse_seq": str(r.get("representative_sparse_seq", "")),
                "importance_score": float("nan"),
            },
        )
        rec["importance_score"] = _safe_float(r.get("importance_score", float("nan")))

    merged_rows = list(merged_rows_by_cluster.values())
    top_rows = _prepare_top_cluster_rows(
        [{k: str(v) for k, v in r.items()} for r in merged_rows],
        top_k=int(args.top_k_clusters),
    )

    summary = {
        "id_contrib_dir": str(base_dir),
        "n_sparse_sequences_per_sample": len(per_sample_rows),
        "n_sparse_clusters": len(cluster_rows),
        "n_sparse_cluster_perturb_rows": len(cluster_pert_rows),
        "top_k_clusters": int(args.top_k_clusters),
        "files": {
            "per_sample_csv": str(per_sample_csv),
            "cluster_csv": str(cluster_csv),
            "cluster_perturb_csv": str(cluster_pert_csv),
        },
    }
    (out_dir / "sparse_viz_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not HAS_MATPLOTLIB:
        print("[warn] matplotlib not found: skip PNG plots")
        print("[info] install with: pip install matplotlib")
        print(f"[done] summary only: {out_dir / 'sparse_viz_summary.json'}")
        return

    title_prefix = str(args.title_prefix).strip()
    if not title_prefix:
        title_prefix = base_dir.name

    plot_top_cluster_importance(
        top_rows,
        out_dir / "sparse_cluster_top_importance.png",
        title_prefix=title_prefix,
    )
    plot_support_vs_importance(
        top_rows,
        out_dir / "sparse_cluster_support_vs_importance.png",
        title_prefix=title_prefix,
    )
    plot_sparse_seq_len_vs_mass(
        per_sample_rows,
        out_dir / "sparse_seq_len_vs_weight_mass.png",
        title_prefix=title_prefix,
    )

    print(f"[done] wrote sparse visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
