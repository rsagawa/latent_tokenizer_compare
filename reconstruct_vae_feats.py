#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../MotionGPT_100FPS")))

def _add_motiongpt_to_syspath(motiongpt_root: Path) -> None:
    root = motiongpt_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_cfg(cfg_assets_path: str, cfg_path: str):
    from mGPT.config import get_module_config

    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    return cfg


def read_split_ids(dataset_root: Path, split: str) -> List[str]:
    split_txt = dataset_root / f"{split}.txt"
    if split_txt.exists():
        ids: List[str] = []
        for line in split_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            ids.append(s.split()[0])
        return ids

    njv_dir = dataset_root / "new_joint_vecs"
    if not njv_dir.exists():
        raise FileNotFoundError(f"new_joint_vecs not found: {njv_dir}")
    return sorted([p.stem for p in njv_dir.glob("*.npy")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motiongpt_root", type=str, default="../MotionGPT")
    ap.add_argument("--cfg_assets", type=str, default="configs/assets.yaml")
    ap.add_argument("--cfg", type=str, default="configs/config_h3d_stage1.yaml")
    ap.add_argument("--ckpt_tar", type=str, default="experiments/mgpt/VQVAE_HumanML3D/checkpoints/last.ckpt")
    ap.add_argument("--dataset_root", type=str, default="../Bandai/HumanML3D_Bandai_20FPS")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    motiongpt_root = Path(args.motiongpt_root)
    _add_motiongpt_to_syspath(motiongpt_root)

    from mGPT.data.HumanML3D import HumanML3DDataModule
    from mGPT.models.build_model import build_model

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dataset_root = Path(args.dataset_root)
    njv_root = dataset_root / "new_joint_vecs"
    out_root = Path(args.out_dir) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_assets_path = Path(args.cfg_assets)
    if not cfg_assets_path.is_absolute():
        cfg_assets_path = motiongpt_root / cfg_assets_path
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = motiongpt_root / cfg_path
    ckpt_path = Path(args.ckpt_tar)
    if not ckpt_path.is_absolute():
        ckpt_path = motiongpt_root / ckpt_path

    cfg = load_cfg(str(cfg_assets_path), str(cfg_path))
    cfg.DATASET.HUMANML3D.ROOT = str(dataset_root)

    datamodule = HumanML3DDataModule(cfg)
    model = build_model(cfg, datamodule)

    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)

    model = model.to(device).eval()
    if not hasattr(model, "vae"):
        raise RuntimeError("model.vae not found")
    vae = model.vae

    ids = read_split_ids(dataset_root, args.split)
    missing = 0
    reconstructed = 0

    for seq_id in tqdm(ids, desc=f"reconstruct:{args.split}"):
        fpath = njv_root / f"{seq_id}.npy"
        if not fpath.exists():
            missing += 1
            continue

        feats_raw = np.load(fpath)
        feats_raw_t = torch.from_numpy(feats_raw).float().unsqueeze(0).to(device)
        feats_norm = datamodule.normalize(feats_raw_t)

        with torch.no_grad():
            code_idx, _ = vae.encode(feats_norm)
            feats_rec_norm = vae.decode(code_idx)
            feats_rec = datamodule.denormalize(feats_rec_norm)

        rec_np = feats_rec[0].detach().cpu().numpy().astype(np.float32)
        np.save(str(out_root / f"{seq_id}.npy"), rec_np)
        reconstructed += 1

    print(f"done: split={args.split} reconstructed={reconstructed} missing={missing} out={out_root}")


if __name__ == "__main__":
    main()
