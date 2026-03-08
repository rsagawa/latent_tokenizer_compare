#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_recon_humanml3d_motiongpt_metrics.py

Compute MotionGPT-style evaluation metrics for *reconstructed* HumanML3D motions.

This script is meant to be used after training a tokenizer / autoencoder that
reconstructs HumanML3D feature sequences (new_joint_vecs, 263-D).

It computes two groups of metrics (the same implementations as MotionGPT):

1) Motion reconstruction metrics (joint-space)
   - MPJPE
   - PAMPJPE (Procrustes Aligned MPJPE)
   - ACCL (joint acceleration error)

2) Motion distribution metrics (T2M-evaluator embedding space)
   - FID
   - DIV (Diversity)

Expected reconstructed outputs
------------------------------

By default the script expects a directory that contains files:

  <recon_dir>/<keyid>.npy

where each file is a float array of shape [T, 263] in *raw HumanML3D feature*
space (same space as <data_root>/new_joint_vecs/<keyid>.npy).

If you pass --recon_root instead of --recon_dir, the script will look for:

  <recon_root>/recon/<split>/<keyid>.npy

which is the directory layout used by the provided baseline scripts.

Requirements
------------

Run inside a MotionGPT environment (so that `import mGPT` works). For example:

  cd /path/to/MotionGPT
  pip install -e .

You also need the T2M evaluator checkpoint (finest.tar) that MotionGPT uses.
This is pointed to by cfg.METRIC.TM2T.t2m_path, or can be overridden by
--t2m_path.

Example
-------

  python eval_recon_humanml3d_motiongpt_metrics.py \
    --cfg_assets ./configs/assets.yaml \
    --cfg ./configs/config_h3d_t2m.yaml \
    --data_root /path/to/HumanML3D \
    --recon_root /path/to/my_baseline_outputs \
    --split test \
    --t2m_path ./deps/t2m_evaluators \
    --out_json recon_eval.json

Notes
-----

* FID/DIV are computed using MotionGPT's TM2TMetrics (T2M-evaluator embeddings).
  We disable text-related matching metrics by forcing cfg.TRAIN.STAGE to a
  non-'lm' stage.
* MPJPE/PAMPJPE/ACCL are computed using MotionGPT's MRMetrics.
  We evaluate per-sequence (batch size 1) to avoid any padding artifacts in
  ACCL.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../MotionGPT_100FPS")))



def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pad_3d(seqs: Sequence[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """Pad a list of [T_i, D] arrays into [B, T_max, D]."""
    if len(seqs) == 0:
        raise ValueError("No sequences to pad")
    max_len = max(int(s.shape[0]) for s in seqs)
    feat_dim = int(seqs[0].shape[1])
    out = np.full((len(seqs), max_len, feat_dim), pad_value, dtype=np.float32)
    for i, s in enumerate(seqs):
        t = int(s.shape[0])
        out[i, :t, :] = s.astype(np.float32, copy=False)
    return out


def _safe_load_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"np.load returned non-ndarray for {path}")
    return arr


def _auto_detect_cfg(defaults: List[str]) -> Optional[str]:
    for p in defaults:
        if os.path.isfile(p):
            return p
    return None


def load_motiongpt_cfg(cfg_assets_path: str, cfg_path: str):
    """Load MotionGPT OmegaConf cfg in the same style as mGPT.config.parse_args()."""
    from omegaconf import OmegaConf
    from mGPT.config import get_module_config

    OmegaConf.register_new_resolver("eval", eval)

    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not bool(getattr(cfg_exp, "FULL_CONFIG", False)):
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    if not hasattr(cfg, "DEBUG"):
        cfg.DEBUG = False
    return cfg


def _read_split_ids(data_root: Path, split: str) -> List[str]:
    split_file = data_root / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    ids: List[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            mid = line.strip()
            if mid:
                ids.append(mid)
    return ids



def _make_base_name(amass_rel: str, start_frame: int, end_frame: int, frame_skip: int, latent_len_ratio: str) -> str:
    """
    Build clip name from (amass_rel, start_frame, end_frame) in the same way as the joints-txt pipeline.
    This is optional and only used when the CSV does not contain explicit motion ids.
    """
    s = amass_rel.strip().replace("\\", "/")
    s = s.replace("/", "_")
    if s.endswith("_poses.npz"):
        s = s[: -len("_poses.npz")]
    elif s.endswith(".npz"):
        s = s[: -len(".npz")]
    s = f"{s}_{start_frame}_{end_frame}_sk{frame_skip}_r{latent_len_ratio}.npy"
    return s


def _read_pairs_from_csv(
    csv_path: Path,
    frame_skip: int = 1,
    latent_len_ratio: str = "1.000",
) -> List[tuple[str, str]]:
    """
    Read evaluation key pairs from a CSV.

    Expected columns:
      - amass_rel
      - start_frame
      - end_frame
      - new_name

    Returns:
      List of (pred_keyid, gt_keyid).

    pred_keyid is built by applying _make_base_name() to amass_rel with the
    (start_frame, end_frame, frame_skip, latent_len_ratio).
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["amass_rel", "start_frame", "end_frame", "new_name"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV is missing required columns: {missing_cols}. Available: {list(df.columns)}")

    pairs: List[tuple[str, str]] = []
    for _, row in df.iterrows():
        amass_rel = row["amass_rel"]
        humanml3d_id = row["new_name"]

        if pd.isna(amass_rel) or pd.isna(humanml3d_id):
            continue
        amass_rel_s = str(amass_rel).strip()
        humanml3d_id_s = str(humanml3d_id).strip()
        if not amass_rel_s or not humanml3d_id_s:
            continue

        try:
            sf = int(row["start_frame"])
            ef = int(row["end_frame"])
        except Exception:
            continue

        pred_keyid = _make_base_name(amass_rel_s, sf, ef, frame_skip, latent_len_ratio)
        pairs.append((pred_keyid, humanml3d_id_s))

    if len(pairs) == 0:
        raise ValueError(
            f"No valid rows found from CSV: {csv_path}. "
            "Required columns: amass_rel, start_frame, end_frame, new_name."
        )
    return pairs


def _resolve_recon_dir(recon_dir: Optional[str], recon_root: Optional[str], split: str) -> Path:
    if recon_dir is not None:
        p = Path(recon_dir)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"--recon_dir not found: {p}")

    if recon_root is None:
        raise SystemExit("ERROR: Either --recon_dir or --recon_root must be provided")

    # root = Path(recon_root)
    # cand = root / "recon" / split
    cand = Path(recon_root)
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"--recon_root not found or layout mismatch: expected {cand}")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_assets", type=str, default="./configs/assets.yaml",
                    help="Path to MotionGPT configs/assets.yaml")
    ap.add_argument("--cfg", type=str, default=None,
                    help="Path to a MotionGPT experiment config yaml (must include METRIC.TM2T.*).")
    ap.add_argument("--data_root", type=str, required=True,
                    help="HumanML3D dataset root (contains new_joint_vecs/, texts/, test.txt, etc.)")
    ap.add_argument("--recon_dir", type=str, default=None,
                    help="Directory containing reconstructed features <keyid>.npy (shape [T,263]).")
    ap.add_argument("--recon_root", type=str, default=None,
                    help="Root directory that contains recon/<split>/<keyid>.npy (baseline script layout).")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                    help="Which split txt to use.")
    ap.add_argument("--id_csv", type=str, default=None,
                    help="Optional CSV to enumerate evaluation pairs instead of <split>.txt. Required columns: amass_rel,start_frame,end_frame,humanml3d_id.")
    ap.add_argument("--frame_skip", type=int, default=1,
                    help="Used only when building ids from (amass_rel,start_frame,end_frame).")
    ap.add_argument("--latent_len_ratio", type=str, default="1.000",
                    help="Used only when building ids from (amass_rel,start_frame,end_frame).")
    ap.add_argument("--t2m_path", type=str, default=None,
                    help="Override cfg.METRIC.TM2T.t2m_path (folder containing evaluators).")
    ap.add_argument("--meta_dir", type=str, default="./assets/meta",
                    help="Directory containing mean_eval.npy/std_eval.npy (MotionGPT assets/meta).")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cpu. default: auto")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for FID/DIV embedding computation (TM2TMetrics.update).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_motion_len", type=int, default=196,
                    help="Hard cap for evaluation (HumanML3D commonly uses <200 frames at 20fps).")
    ap.add_argument("--length_mode", type=str, default="ref", choices=["ref", "pred", "min"],
                    help="How to set reconstructed length for FID/DIV when output length differs from GT.")
    ap.add_argument("--diversity_times", type=int, default=300,
                    help="Number of pairs for Diversity (MotionGPT default: 300).")
    ap.add_argument("--out_json", type=str, default=None,
                    help="If set, save metrics to this json file.")
    ap.add_argument("--quiet", action="store_true",
                    help="Reduce per-sample logging.")

    args = ap.parse_args()

    _seed_everything(args.seed)

    # Auto-detect cfg if omitted
    if args.cfg is None:
        guess = _auto_detect_cfg([
            "./configs/config_h3d_t2m.yaml",
            "./configs/config_h3d_stage3.yaml",
            "./configs/config_h3d.yaml",
            "./configs/humanml3d.yaml",
        ])
        if guess is None:
            raise SystemExit("ERROR: --cfg is required (could not auto-detect).")
        print(f"[INFO] Auto-detected cfg: {guess}")
        args.cfg = guess

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise SystemExit(f"ERROR: data_root not found: {data_root}")

    recon_dir = _resolve_recon_dir(args.recon_dir, args.recon_root, args.split)
    if not recon_dir.is_dir():
        raise SystemExit(f"ERROR: recon_dir not found: {recon_dir}")

    # Load cfg (for TM2TMetrics instantiation, evaluator paths, UNIT_LEN, etc.)
    cfg = load_motiongpt_cfg(args.cfg_assets, args.cfg)

    # Force non-LM stage so TM2TMetrics only computes FID/DIV (no text metrics).
    # TM2TMetrics.text is: ('lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 't2m')
    cfg.DATASET.HUMANML3D.ROOT = str(data_root)
    if hasattr(cfg, "TRAIN") and hasattr(cfg.TRAIN, "STAGE"):
        cfg.TRAIN.STAGE = "t2m"
    if args.t2m_path is not None:
        cfg.METRIC.TM2T.t2m_path = args.t2m_path

    unit_len = int(getattr(cfg.DATASET.HUMANML3D, "UNIT_LEN", 4))
    njoints = 22

    # Load evaluator normalization stats
    meta_dir = Path(args.meta_dir)
    mean_eval = np.load(str(meta_dir / "mean_eval.npy")).astype(np.float32)
    std_eval = np.load(str(meta_dir / "std_eval.npy")).astype(np.float32)

    # Motion processing: feats -> joints
    from mGPT.data.humanml.scripts.motion_process import recover_from_ric

    # Metrics
    from mGPT.metrics.t2m import TM2TMetrics
    from mGPT.metrics.mr import MRMetrics

    tm2t = TM2TMetrics(cfg, dataname="humanml3d", diversity_times=int(args.diversity_times)).to(device)

    # MRMetrics uses CPU internally for pampjpe. Keep it on CPU.
    mr = MRMetrics(njoints=njoints, jointstype="humanml3d", force_in_meter=True, align_root=True,
                   dist_sync_on_step=False)

    # Evaluate list
    if args.id_csv is not None:
        pairs = _read_pairs_from_csv(
            Path(args.id_csv),
            frame_skip=int(args.frame_skip),
            latent_len_ratio=str(args.latent_len_ratio),
        )
    else:
        ids = _read_split_ids(data_root, args.split)
        pairs = [(mid, mid) for mid in ids]

    missing = 0
    used = 0
    skipped_nan = 0
    skipped_shape = 0

    # TM2T batch buffers
    batch_ref: List[np.ndarray] = []
    batch_rst: List[np.ndarray] = []
    batch_len_ref: List[int] = []
    batch_len_rst: List[int] = []

    def flush_tm2t_batch():
        nonlocal batch_ref, batch_rst, batch_len_ref, batch_len_rst
        if len(batch_ref) == 0:
            return
        ref_pad = _pad_3d(batch_ref, pad_value=0.0)
        rst_pad = _pad_3d(batch_rst, pad_value=0.0)
        tm2t.update(
            feats_ref=torch.from_numpy(ref_pad).to(device),
            feats_rst=torch.from_numpy(rst_pad).to(device),
            lengths_ref=list(batch_len_ref),
            lengths_rst=list(batch_len_rst),
        )
        batch_ref, batch_rst = [], []
        batch_len_ref, batch_len_rst = [], []

    motion_dir = data_root / "new_joint_vecs"
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"new_joint_vecs not found: {motion_dir}")

    for pred_id, gt_id in pairs:
        gt_path = motion_dir / gt_id
        pred_path = recon_dir / pred_id

        if not pred_path.is_file():
            missing += 1
            continue
        if not gt_path.is_file():
            # Should not happen if split is consistent
            missing += 1
            continue

        gt_raw = _safe_load_npy(gt_path)
        pred_raw = _safe_load_npy(pred_path)

        if gt_raw.ndim != 2 or pred_raw.ndim != 2 or gt_raw.shape[1] != 263 or pred_raw.shape[1] != 263:
            skipped_shape += 1
            if not args.quiet:
                print(f"[WARN] Skip gt={gt_id} pred={pred_id}: bad shape gt={gt_raw.shape} pred={pred_raw.shape}")
            continue

        # Cap lengths and enforce multiple of unit_len (same logic used in HumanML3D loaders)
        ref_len = int(min(int(gt_raw.shape[0]), int(args.max_motion_len)))
        rst_len_full = int(min(int(pred_raw.shape[0]), int(args.max_motion_len)))

        ref_len = (ref_len // unit_len) * unit_len
        rst_len_full = (rst_len_full // unit_len) * unit_len

        if ref_len <= 0 or rst_len_full <= 0:
            skipped_shape += 1
            continue

        # Decide length for TM2T features
        if args.length_mode == "pred":
            rst_len = rst_len_full
        elif args.length_mode == "min":
            rst_len = min(ref_len, rst_len_full)
            ref_len = rst_len
        else:  # "ref"
            rst_len = min(ref_len, rst_len_full)

        # Slice
        gt_raw_ref = gt_raw[:ref_len]
        pred_raw_rst = pred_raw[:rst_len]

        # If lengths differ, also cap MR to common length
        mr_len = min(ref_len, rst_len)
        gt_raw_mr = gt_raw[:mr_len]
        pred_raw_mr = pred_raw[:mr_len]

        if (np.isnan(gt_raw_ref).any() or np.isnan(pred_raw_rst).any() or
                np.isnan(gt_raw_mr).any() or np.isnan(pred_raw_mr).any()):
            skipped_nan += 1
            if not args.quiet:
                print(f"[WARN] Skip gt={gt_id} pred={pred_id}: NaN found")
            continue

        # ===== TM2T: eval-normalized features for FID/DIV =====
        gt_eval = (gt_raw_ref.astype(np.float32, copy=False) - mean_eval) / std_eval
        pred_eval = (pred_raw_rst.astype(np.float32, copy=False) - mean_eval) / std_eval
        batch_ref.append(gt_eval)
        batch_rst.append(pred_eval)
        batch_len_ref.append(ref_len)
        batch_len_rst.append(rst_len)

        # ===== MR: joints-space MPJPE/PAMPJPE/ACCL =====
        # recover_from_ric supports torch tensors; use CPU to avoid GPU memory spikes.
        with torch.no_grad():
            gt_j = recover_from_ric(torch.from_numpy(gt_raw_mr.astype(np.float32)), njoints)
            pr_j = recover_from_ric(torch.from_numpy(pred_raw_mr.astype(np.float32)), njoints)

        if gt_j.ndim != 3 or pr_j.ndim != 3:
            skipped_shape += 1
            continue

        # Shape to (bs=1, seq, njoints, 3)
        gt_j = gt_j.unsqueeze(0)
        pr_j = pr_j.unsqueeze(0)

        mr.update(joints_rst=pr_j, joints_ref=gt_j, lengths=[mr_len])

        used += 1
        if len(batch_ref) >= int(args.batch_size):
            flush_tm2t_batch()

    flush_tm2t_batch()

    if used == 0:
        raise SystemExit("ERROR: No valid reconstructions found (0 matched ids).")

    # Guard: MotionGPT's TM2TMetrics asserts count_seq > diversity_times
    if used <= tm2t.diversity_times:
        tm2t.diversity_times = max(1, used - 1)

    tm2t_res = tm2t.compute(sanity_flag=False)
    # tm2t_res = {k: float(v.detach().cpu().item()) for k, v in tm2t_res.items()}
    tm2t_res = {k: float(v) for k, v in tm2t_res.items()}

    mr_res = mr.compute(sanity_flag=False)
    # mr_res = {k: float(v.detach().cpu().item()) for k, v in mr_res.items()}
    mr_res = {k: float(v) for k, v in mr_res.items()}

    # Rename to user-requested keys
    out_metrics: Dict[str, float] = {
        "MPJPE": float(mr_res.get("MPJPE", float("nan"))),
        "PAMPJPE": float(mr_res.get("PAMPJPE", float("nan"))),
        "ACCL": float(mr_res.get("ACCEL", float("nan"))),
        "FID": float(tm2t_res.get("FID", float("nan"))),
        "DIV": float(tm2t_res.get("Diversity", float("nan"))),
        "gt_DIV": float(tm2t_res.get("gt_Diversity", float("nan"))),
    }

    print("\n=== Motion Reconstruction (MR) ===")
    print(f"MPJPE:   {out_metrics['MPJPE']:.6f}")
    print(f"PAMPJPE: {out_metrics['PAMPJPE']:.6f}")
    print(f"ACCL:    {out_metrics['ACCL']:.6f}")

    print("\n=== Motion Distribution (T2M embeddings) ===")
    print(f"FID: {out_metrics['FID']:.6f}")
    print(f"DIV: {out_metrics['DIV']:.6f}")
    print(f"gt_DIV: {out_metrics['gt_DIV']:.6f}")

    print("\n=== Summary ===")
    print(f"eval entries: {len(pairs)}")
    print(f"used:      {used}")
    print(f"missing:   {missing}")
    print(f"skip_nan:  {skipped_nan}")
    print(f"skip_shape:{skipped_shape}")
    print(f"recon_dir: {recon_dir}")

    out: Dict[str, object] = {
        "metrics": out_metrics,
        "details": {
            "split": args.split,
            "num_ids": len(pairs),
            "used": used,
            "missing": missing,
            "skipped_nan": skipped_nan,
            "skipped_shape": skipped_shape,
            "recon_dir": str(recon_dir),
            "data_root": str(data_root),
            "length_mode": args.length_mode,
            "max_motion_len": int(args.max_motion_len),
            "unit_len": unit_len,
            "seed": int(args.seed),
            "device": str(device),
        },
        "raw_tm2t": tm2t_res,
        "raw_mr": mr_res,
    }

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        print(
            "ERROR: MotionGPT modules were not found.\n"
            "Run this script from inside the MotionGPT repository root, or make sure MotionGPT is installed.\n"
            f"Details: {e}"
        )
        raise
