#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert reconstructed root+vposer vectors (36D) to HumanML3D 263D vectors.

Input feature per frame:
  [root(4), vposer_latent(32)]
Output feature per frame:
  [root(4), ric(63), rot6d(126), local_vel(66), foot_contact(4)] = 263

Notes:
- This script keeps root(4) from input as-is.
- vposer latent is decoded with VPoser + SMPL-H BodyModel, then converted to
  HumanML3D-style components.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

# Compatibility shim for legacy HumanML3D code that references np.float
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


def _ensure_humanml3d_import_path() -> None:
    here = Path(__file__).resolve().parent
    cand = (here / "../HumanML3D").resolve()
    if cand.exists():
        cstr = str(cand)
        if cstr not in sys.path:
            sys.path.insert(0, cstr)


_ensure_humanml3d_import_path()

from common.skeleton import Skeleton
from common.quaternion import qfix, qinv_np, qrot_np, quaternion_to_cont6d_np
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model as hbp_load_model
from paramUtil import t2m_kinematic_chain, t2m_raw_offsets


FACE_JOINT_INDX = [2, 1, 17, 16]
JOINTS_NUM = 22
FID_R, FID_L = [8, 11], [7, 10]
N_RAW_OFFSETS = torch.from_numpy(t2m_raw_offsets).float()


class VPoserDecoder:
    def __init__(
        self,
        *,
        vposer_expr_dir: str,
        body_models_root: str,
        device: torch.device,
        gender: str = "male",
        num_betas: int = 10,
        num_dmpls: int = 8,
    ) -> None:
        vposer, _ = hbp_load_model(
            vposer_expr_dir,
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        self.vposer = vposer.to(device)
        self.vposer.eval()

        gender = str(gender).lower().strip()
        if gender not in {"male", "female"}:
            gender = "male"

        bm_path = Path(body_models_root) / "smplh" / gender / "model.npz"
        dmpl_path = Path(body_models_root) / "dmpls" / gender / "model.npz"
        if not bm_path.exists():
            raise FileNotFoundError(f"SMPL-H model not found: {bm_path}")
        if not dmpl_path.exists():
            raise FileNotFoundError(f"DMPL model not found: {dmpl_path}")

        self.body_model = BodyModel(
            bm_fname=str(bm_path),
            num_betas=int(num_betas),
            num_dmpls=int(num_dmpls),
            dmpl_fname=str(dmpl_path),
        ).to(device)
        self.body_model.eval()

        self.device = device
        self.latent_dim = int(getattr(self.vposer, "latentD", 32))

    @torch.no_grad()
    def decode_to_joints22(self, z: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Decode vposer latent [T,D] -> joints [T,22,3] using SMPL-H Jtr."""
        if z.ndim != 2:
            raise ValueError(f"z must be 2D [T,D], got {z.shape}")
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"latent dim mismatch: input={z.shape[1]} model={self.latent_dim}")
        if z.shape[0] == 0:
            return np.zeros((0, JOINTS_NUM, 3), dtype=np.float32)

        outs: List[np.ndarray] = []
        zero_root_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        zero_pose_hand = torch.zeros((1, 90), dtype=torch.float32, device=self.device)
        zero_trans = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        zero_betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device)

        for s in range(0, z.shape[0], batch_size):
            e = min(s + batch_size, z.shape[0])
            z_t = torch.from_numpy(z[s:e]).to(self.device)

            dec = self.vposer.decode(z_t)
            pose_body = dec["pose_body"] if isinstance(dec, dict) else dec
            pose_body = pose_body.reshape(z_t.shape[0], -1).to(dtype=torch.float32)

            body = self.body_model(
                root_orient=zero_root_orient.expand(z_t.shape[0], -1),
                pose_body=pose_body,
                pose_hand=zero_pose_hand.expand(z_t.shape[0], -1),
                trans=zero_trans.expand(z_t.shape[0], -1),
                betas=zero_betas.expand(z_t.shape[0], -1),
            )
            jtr = body.Jtr[:, :JOINTS_NUM, :].detach().cpu().numpy().astype(np.float32)
            outs.append(jtr)

        return np.concatenate(outs, axis=0)


def recover_root_rot_pos_np(root4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """root4 [L,4] -> (root_quat [L,4], root_pos [L,3])."""
    if root4.ndim != 2 or root4.shape[1] != 4:
        raise ValueError(f"root4 must be [L,4], got {root4.shape}")

    rot_vel = root4[:, 0]
    r_rot_ang = np.zeros_like(rot_vel, dtype=np.float32)
    if len(rot_vel) > 1:
        r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang, axis=0)

    r_rot_quat = np.zeros((len(root4), 4), dtype=np.float32)
    r_rot_quat[:, 0] = np.cos(r_rot_ang)
    r_rot_quat[:, 2] = np.sin(r_rot_ang)

    r_pos = np.zeros((len(root4), 3), dtype=np.float32)
    if len(root4) > 1:
        r_pos[1:, [0, 2]] = root4[:-1, 1:3]
    r_pos = qrot_np(qinv_np(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=0)
    r_pos[:, 1] = root4[:, 3]

    return r_rot_quat, r_pos


def _pad_last_frame(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return x
    return np.concatenate([x, x[-1:]], axis=0)


def _foot_detect(positions: np.ndarray, thres: float) -> tuple[np.ndarray, np.ndarray]:
    """positions [T,J,3] -> feet_l, feet_r of shape [T-1,2] each."""
    feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
    feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
    feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < np.array([thres, thres], dtype=np.float32)).astype(np.float32)

    feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
    feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
    feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < np.array([thres, thres], dtype=np.float32)).astype(np.float32)

    return feet_l, feet_r


def build_humanml263_from_root_and_joints(root4: np.ndarray, joints22: np.ndarray, feet_thre: float = 0.002) -> np.ndarray:
    """
    root4: [L,4]
    joints22: [L,22,3] decoded from vposer latent
    returns: [L,263]
    """
    if root4.shape[0] != joints22.shape[0]:
        raise ValueError(f"length mismatch: root={root4.shape[0]}, joints={joints22.shape[0]}")
    L = root4.shape[0]
    if L == 0:
        return np.zeros((0, 263), dtype=np.float32)

    # Make decoded joints root-relative, then place them on reconstructed root trajectory.
    local_j = joints22.astype(np.float32) - joints22[:, 0:1, :].astype(np.float32)
    r_rot_quat, r_pos = recover_root_rot_pos_np(root4.astype(np.float32))

    global_j = qrot_np(np.repeat(qinv_np(r_rot_quat)[:, None, :], JOINTS_NUM, axis=1), local_j)
    global_j = global_j + r_pos[:, None, :]

    # Pad one frame to produce T-1 style derivatives while keeping output length L.
    global_pad = _pad_last_frame(global_j)
    r_rot_pad = _pad_last_frame(r_rot_quat)

    # RIC (rotation-invariant coords)
    ric_pos = global_pad.copy()
    ric_pos[..., 0] -= ric_pos[:, 0:1, 0]
    ric_pos[..., 2] -= ric_pos[:, 0:1, 2]
    ric_pos = qrot_np(np.repeat(r_rot_pad[:, None, :], JOINTS_NUM, axis=1), ric_pos)
    ric_data = ric_pos[:, 1:].reshape(global_pad.shape[0], -1)[:-1]

    # Joint rotation 6D from IK on 22-joint skeleton
    skel = Skeleton(N_RAW_OFFSETS, t2m_kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(global_pad, FACE_JOINT_INDX, smooth_forward=True)
    quat_params = qfix(quat_params)
    cont6d = quaternion_to_cont6d_np(quat_params)
    rot_data = cont6d[:, 1:].reshape(global_pad.shape[0], -1)[:-1]

    # Local joint velocities
    local_vel = qrot_np(
        np.repeat(r_rot_pad[:-1, None, :], JOINTS_NUM, axis=1),
        global_pad[1:] - global_pad[:-1],
    ).reshape(L, -1)

    # Foot contacts
    feet_l, feet_r = _foot_detect(global_pad, float(feet_thre))

    out = np.concatenate([
        root4.astype(np.float32),
        ric_data.astype(np.float32),
        rot_data.astype(np.float32),
        local_vel.astype(np.float32),
        feet_l.astype(np.float32),
        feet_r.astype(np.float32),
    ], axis=-1)

    if out.shape[1] != 263:
        raise RuntimeError(f"Unexpected output dim: got {out.shape[1]}, expected 263")
    return out


def iter_input_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if recursive:
        for p in sorted(input_path.rglob("*.npy")):
            if p.is_file():
                yield p
    else:
        for p in sorted(input_path.glob("*.npy")):
            if p.is_file():
                yield p


def resolve_output_path(input_root: Path, in_file: Path, out_root: Path) -> Path:
    if input_root.is_file():
        return out_root / in_file.name
    rel = in_file.relative_to(input_root)
    return out_root / rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Input .npy file or directory (root+vposer 36D)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for HumanML3D 263D .npy")
    ap.add_argument("--vposer_expr_dir", type=str, default="../HumanML3D/human_body_prior/train/V02_05")
    ap.add_argument("--body_models_root", type=str, default="../HumanML3D/body_models")
    ap.add_argument("--gender", type=str, default="male", choices=["male", "female"])
    ap.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:0")
    ap.add_argument("--feet_thre", type=float, default=0.002)
    ap.add_argument("--decode_batch_size", type=int, default=1024)
    ap.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    ap.add_argument("--latent_dim", type=int, default=32, help="Expected latent dim in input (default: 32)")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    decoder = VPoserDecoder(
        vposer_expr_dir=args.vposer_expr_dir,
        body_models_root=args.body_models_root,
        device=device,
        gender=args.gender,
    )

    if int(args.latent_dim) != int(decoder.latent_dim):
        raise ValueError(
            f"--latent_dim({args.latent_dim}) != VPoser latentD({decoder.latent_dim}). "
            "Set --latent_dim to match your input data."
        )

    files = list(iter_input_files(input_path, recursive=bool(args.recursive)))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files found under: {input_path}")

    ok = 0
    skipped = 0
    for p in files:
        try:
            arr = np.load(str(p))
            if arr.ndim != 2:
                raise ValueError(f"Expected [T,D], got {arr.shape}")
            need_dim = 4 + int(args.latent_dim)
            if arr.shape[1] != need_dim:
                raise ValueError(f"Expected feature dim {need_dim}, got {arr.shape[1]}")

            root4 = arr[:, :4].astype(np.float32)
            z = arr[:, 4:].astype(np.float32)

            joints22 = decoder.decode_to_joints22(z, batch_size=int(args.decode_batch_size))
            out263 = build_humanml263_from_root_and_joints(root4, joints22, feet_thre=float(args.feet_thre))

            out_path = resolve_output_path(input_path, p, out_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), out263.astype(np.float32))
            ok += 1
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
            skipped += 1

    print(f"Done. converted={ok}, skipped={skipped}, out_dir={out_root}")


if __name__ == "__main__":
    main()
