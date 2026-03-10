#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required to run this script") from exc


DEFAULT_NAMES = ["MotionGPT", "m2dm", "proposed", "latent_tokenizer"]
DEFAULT_TASKS = ["actionrec", "retrieval"]
DEFAULT_SPLITS = ["test", "val", "train"]


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_token_file(base_dir: Path, name: str, mid: str, split: str) -> Tuple[Path, str]:
    token_root = base_dir / f"tokens_out2_{name}"
    splits = [split] if split != "auto" else list(DEFAULT_SPLITS)
    for sp in splits:
        direct = token_root / sp / f"{mid}.txt"
        nested = token_root / "tokens" / sp / f"{mid}.txt"
        if direct.is_file():
            return direct, sp
        if nested.is_file():
            return nested, sp
    raise FileNotFoundError(f"token file not found: tokenizer={name}, mid={mid}, split={split}")


def _load_token_ids(base_dir: Path, name: str, mid: str, split: str) -> Tuple[List[int], str, Path]:
    token_path, resolved_split = _find_token_file(base_dir, name, mid, split)
    text = token_path.read_text(encoding="utf-8").strip()
    token_ids = [int(x) for x in text.split()] if text else []
    return token_ids, resolved_split, token_path


def _lookup_frame_count_from_index(hml_root: Path, mid: str) -> int | None:
    index_path = hml_root / "index.csv"
    if not index_path.is_file():
        return None
    target = f"{mid}.npy"
    with index_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("new_name", "")).strip() != target:
                continue
            start = int(_to_float(row.get("start_frame"), 0.0))
            end = int(_to_float(row.get("end_frame"), 0.0))
            if end > start:
                return end - start
    return None


def _lookup_frame_count_from_npy(hml_root: Path, mid: str) -> int | None:
    if np is None:
        return None
    candidates = [
        hml_root / "new_joints" / f"{mid}.npy",
        hml_root / "new_joint_vecs" / f"{mid}.npy",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if arr.ndim >= 1 and int(arr.shape[0]) > 0:
            return int(arr.shape[0])
    return None


def resolve_frame_count(hml_root: Path, mid: str, fallback_tokens: int) -> int:
    frame_count = _lookup_frame_count_from_npy(hml_root, mid)
    if frame_count is not None:
        return frame_count
    frame_count = _lookup_frame_count_from_index(hml_root, mid)
    if frame_count is not None:
        return frame_count
    return max(1, int(fallback_tokens))


def _sample_attr_path(base_dir: Path, task: str, name: str, out_suffix: str) -> Path:
    return base_dir / f"{task}2_{name}" / out_suffix / "id_attribution_per_sample.csv"


def load_sample_importance_rows(
    base_dir: Path,
    task: str,
    name: str,
    out_suffix: str,
    mid: str,
) -> List[Dict[str, str]]:
    path = _sample_attr_path(base_dir, task, name, out_suffix)
    if not path.is_file():
        return []
    rows = [row for row in _read_csv_rows(path) if str(row.get("mid", "")).strip() == mid]
    rows.sort(key=lambda r: (-_to_float(r.get("abs_attr_norm"), 0.0), int(_to_float(r.get("item_id"), 0.0))))
    return rows


def build_token_occurrence_matrix(
    token_ids: Sequence[int],
    sample_rows: Sequence[Dict[str, str]],
    frame_count: int,
    score_mode: str,
    max_ids: int,
) -> Tuple[List[List[float]], List[int], Dict[int, float]]:
    if not token_ids or frame_count <= 0:
        return [], [], {}

    importance_by_id: Dict[int, float] = {}
    for row in sample_rows:
        tid = int(_to_float(row.get("item_id"), 0.0))
        if score_mode == "signed":
            score = _to_float(row.get("signed_attr_norm"), 0.0)
        else:
            score = _to_float(row.get("abs_attr_norm"), 0.0)
        importance_by_id[tid] = score

    ids_in_sample = [tid for tid in token_ids if tid in importance_by_id]
    unique_ids = sorted(set(ids_in_sample), key=lambda tid: (-abs(float(importance_by_id[tid])), tid))
    if int(max_ids) > 0:
        unique_ids = unique_ids[: int(max_ids)]
    if not unique_ids:
        return [], [], importance_by_id
    selected_ids = set(unique_ids)

    counts = defaultdict(int)
    for tid in token_ids:
        if tid in selected_ids:
            counts[tid] += 1

    mat = [[0.0 for _ in range(frame_count)] for _ in range(len(unique_ids))]
    row_of = {tid: idx for idx, tid in enumerate(unique_ids)}
    n_tokens = len(token_ids)
    for tok_idx, tid in enumerate(token_ids):
        if tid not in selected_ids:
            continue
        left = int(math.floor(tok_idx * frame_count / n_tokens))
        right = int(math.floor((tok_idx + 1) * frame_count / n_tokens))
        left = max(0, min(left, frame_count - 1))
        right = max(left + 1, min(frame_count, right if right > left else left + 1))
        share = float(importance_by_id[tid]) / float(max(1, counts[tid]))
        row = mat[row_of[tid]]
        for frame_idx in range(left, right):
            row[frame_idx] += share
    return mat, unique_ids, importance_by_id


def export_matrix_csv(
    out_path: Path,
    mid: str,
    task: str,
    name: str,
    item_ids: Sequence[int],
    matrix: Sequence[Sequence[float]],
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mid", "task", "tokenizer", "item_id", "frame", "importance"],
        )
        writer.writeheader()
        for row_idx, item_id in enumerate(item_ids):
            row = matrix[row_idx] if row_idx < len(matrix) else []
            for frame_idx, val in enumerate(row):
                if abs(val) <= 1e-12:
                    continue
                writer.writerow(
                    {
                        "mid": mid,
                        "task": task,
                        "tokenizer": name,
                        "item_id": int(item_id),
                        "frame": int(frame_idx),
                        "importance": val,
                    }
                )


def _title_suffix(class_name: str) -> str:
    return f" | {class_name}" if class_name else ""


def plot_heatmaps(
    panels: Dict[str, Dict[str, Dict[str, object]]],
    mid: str,
    out_path: Path,
    score_mode: str,
) -> None:
    tasks = list(panels.keys())
    if not tasks:
        raise ValueError("no panels to plot")
    names = []
    for task in tasks:
        for name in panels[task].keys():
            if name not in names:
                names.append(name)

    fig, axes = plt.subplots(
        len(tasks),
        len(names),
        figsize=(4.4 * max(1, len(names)), 3.2 * max(1, len(tasks))),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{mid} | frame x token-ID importance ({score_mode})", fontsize=13)

    for row_idx, task in enumerate(tasks):
        for col_idx, name in enumerate(names):
            ax = axes[row_idx][col_idx]
            panel = panels.get(task, {}).get(name)
            if not panel:
                ax.axis("off")
                ax.set_title(f"{task} | {name}\nmissing")
                continue

            matrix = panel["matrix"]
            item_ids = panel["item_ids"]
            class_name = str(panel.get("class_name", ""))
            n_tokens = int(panel.get("n_tokens", 0))
            frame_count = int(panel.get("frame_count", 0))
            if not matrix:
                ax.axis("off")
                ax.set_title(f"{task} | {name}\nno attributed IDs")
                continue

            flat = [abs(v) if score_mode == "signed" else v for row in matrix for v in row]
            vmax = max(flat) if flat else 0.0
            if vmax <= 0.0:
                vmax = 1.0
            cmap = "coolwarm" if score_mode == "signed" else "viridis"
            vmin = -vmax if score_mode == "signed" else 0.0
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{task} | {name}{_title_suffix(class_name)}")
            ax.set_xlabel(f"frame (n={frame_count})")
            ax.set_ylabel("item_id")
            ax.set_yticks(list(range(len(item_ids))))
            ax.set_yticklabels([str(x) for x in item_ids], fontsize=8)
            if frame_count > 8:
                if frame_count == 1:
                    xticks = [0]
                else:
                    xticks = sorted(set(int(round(i * (frame_count - 1) / 5.0)) for i in range(6)))
                ax.set_xticks(xticks)
            ax.text(
                0.99,
                0.01,
                f"tokens={n_tokens}\nids={len(item_ids)}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="white" if score_mode == "signed" else "w",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 3},
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Bandai2 sample-specific frame x token-ID importance heatmaps for four tokenizers."
    )
    ap.add_argument("--mid", type=str, required=True, help="Bandai2 sample ID without extension.")
    ap.add_argument("--base_dir", type=str, default="experiments/bandai")
    ap.add_argument("--hml_root", type=str, default="../Bandai/HumanML3D_Bandai2_20FPS")
    ap.add_argument("--out_suffix", type=str, default="id_contrib_test")
    ap.add_argument("--out_dir", type=str, default="experiments/bandai/sample_id_importance")
    ap.add_argument("--names", type=str, nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--split", type=str, default="auto", help="train/val/test or auto")
    ap.add_argument("--score_mode", type=str, default="abs", choices=["abs", "signed"])
    ap.add_argument("--max_ids", type=int, default=0, help="If >0, show only the top-N IDs per task/tokenizer.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    hml_root = Path(args.hml_root)
    out_dir = Path(args.out_dir)

    panels: Dict[str, Dict[str, Dict[str, object]]] = {}
    summary_rows: List[Dict[str, object]] = []
    missing_attr_messages: List[str] = []
    found_any_attr = False

    for task in args.tasks:
        panels[task] = {}
        for name in args.names:
            sample_rows = load_sample_importance_rows(base_dir, task, name, args.out_suffix, args.mid)
            token_ids, resolved_split, token_path = _load_token_ids(base_dir, name, args.mid, args.split)
            if sample_rows:
                found_any_attr = True
            else:
                missing_attr_messages.append(
                    f"{task}/{name}: no attribution rows for mid={args.mid} in "
                    f"{_sample_attr_path(base_dir, task, name, args.out_suffix)} "
                    f"(token split={resolved_split})"
                )
            frame_count = resolve_frame_count(hml_root, args.mid, fallback_tokens=len(token_ids))
            matrix, item_ids, importance_by_id = build_token_occurrence_matrix(
                token_ids=token_ids,
                sample_rows=sample_rows,
                frame_count=frame_count,
                score_mode=args.score_mode,
                max_ids=args.max_ids,
            )
            class_name = str(sample_rows[0].get("class_name", "")) if sample_rows else ""
            panels[task][name] = {
                "matrix": matrix,
                "item_ids": item_ids,
                "class_name": class_name,
                "n_tokens": len(token_ids),
                "frame_count": frame_count,
            }
            for rank, item_id in enumerate(item_ids, start=1):
                summary_rows.append(
                    {
                        "mid": args.mid,
                        "task": task,
                        "tokenizer": name,
                        "class_name": class_name,
                        "item_id": item_id,
                        "rank": rank,
                        "sample_importance": float(importance_by_id.get(item_id, 0.0)),
                        "token_count_in_sample": sum(1 for tid in token_ids if tid == item_id),
                        "resolved_split": resolved_split,
                        "token_path": str(token_path),
                        "frame_count": frame_count,
                    }
                )

    if not found_any_attr:
        details = "\n".join(missing_attr_messages)
        raise SystemExit(
            "No attribution rows were found for this sample.\n"
            f"mid={args.mid}\n"
            f"out_suffix={args.out_suffix}\n"
            "This usually means the sample belongs to a different split than the attribution outputs.\n"
            "Generate attribution for the matching split, or point --out_suffix to that output.\n"
            f"{details}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        for name in args.names:
            panel = panels.get(task, {}).get(name, {})
            export_matrix_csv(
                out_dir / f"{args.mid}_{task}_{name}_frame_id_importance.csv",
                args.mid,
                task,
                name,
                panel.get("item_ids", []),
                panel.get("matrix", []),
            )

    with (out_dir / f"{args.mid}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mid",
                "task",
                "tokenizer",
                "class_name",
                "item_id",
                "rank",
                "sample_importance",
                "token_count_in_sample",
                "resolved_split",
                "token_path",
                "frame_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_heatmaps(
        panels=panels,
        mid=args.mid,
        out_path=out_dir / f"{args.mid}_frame_x_id_importance.png",
        score_mode=args.score_mode,
    )


if __name__ == "__main__":
    main()
