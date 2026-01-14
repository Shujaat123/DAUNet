#!/usr/bin/env python3
import os
import json
import math
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from dataset_Pubic_Symphysis import PUBIC
from utils import _load_config, setup_logger, make_model, load_ckpt


# ---------------------------
# Args
# ---------------------------
def get_args():
    p = argparse.ArgumentParser("Comprehensive Evaluation (Multiclass + PSFH)")
    p.add_argument("--log_dir", required=True, type=str)
    p.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--split", type=str, default="test", choices=["test", "train"])
    p.add_argument("--save-csv", type=str, default="eval_metrics_comprehensive.csv")
    return p.parse_args()


# ---------------------------
# Metric helpers
# ---------------------------
def _boundary(mask):
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    er = binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
    return mask & (~er)

def _directed_surface_dists_mm(A, B, spacing):
    if not A.any() and not B.any():
        return np.array([0.0]), np.array([0.0])
    if not A.any() or not B.any():
        return np.array([math.inf]), np.array([math.inf])

    Ab, Bb = _boundary(A), _boundary(B)
    ya, xa = np.nonzero(Ab)
    yb, xb = np.nonzero(Bb)

    ptsA = np.column_stack([ya * spacing[0], xa * spacing[1]])
    ptsB = np.column_stack([yb * spacing[0], xb * spacing[1]])

    dAB, _ = cKDTree(ptsB).query(ptsA, k=1)
    dBA, _ = cKDTree(ptsA).query(ptsB, k=1)
    return dAB, dBA

def _hd95_asd(dAB, dBA):
    if np.isinf(dAB).all() and np.isinf(dBA).all():
        return math.inf, math.inf
    both = np.concatenate([dAB, dBA])
    return np.percentile(both, 95), both.mean()

def _dice(gt, pr):
    if not gt.any() and not pr.any():
        return 1.0
    inter = np.logical_and(gt, pr).sum()
    denom = gt.sum() + pr.sum()
    return (2.0 * inter) / denom if denom > 0 else 0.0

def _to_psfh(mask):
    return np.isin(mask, [1, 2])


def _read_spacing(dataroot, split, case):
    path = os.path.join(dataroot, f"{split}dataset", "label_mha", f"{case}.mha")
    img = sitk.ReadImage(path)
    sp = img.GetSpacing()
    return (sp[1], sp[0]), sitk.GetArrayFromImage(img).shape[:2]


# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()
    log_dir = Path(args.log_dir)
    cfg = _load_config(str(log_dir))

    logger = setup_logger(log_dir, filename="eval_comprehensive.log")
    logger.info("========== COMPREHENSIVE EVAL ==========")
    logger.info(json.dumps(cfg, indent=2))

    device = args.device

    dataset_name = cfg.get("dataset", "PSFH")
    if dataset_name == "PSFH":
        dataroot = "/home/muhammad_jabbar/diffusion/code/Fast-DDPM/data/Pubic_Symphysis_Fetal_Head_Segmentation_and_Angle_of_Progression"
        ds = PUBIC(dataroot, cfg["img_size"], split=args.split)
        n_classes = cfg["n_classes"]
    else:
        raise ValueError(dataset_name)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    daunet_att_mod=cfg.get("daunet_att_mod", ["enc", "btn", "dec"])

    model = make_model(
        kind=cfg["model"],
        n_channels=cfg["n_channels"],
        n_classes=n_classes,
        base=cfg["base"],
        bilinear=cfg["bilinear"],
        use_se=cfg.get("use_se", False),
        rg_reparam_bn=cfg.get("rg_reparam_bn", True),
        rg_reparam_identity=cfg.get("rg_reparam_identity", False),
        daunet_btlnk_kernel=cfg.get("btlnk_kernel", 3),
    ).to(device)

    load_ckpt(log_dir / "ckpt_best.pth", model, map_location=device)
    model.eval()

    # Model parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    logger.info(f"Model parameters: {n_params:,}")

    results = []
    sums = {k: [] for k in [
        "dice_c0","dice_c1","dice_c2",
        "hd95_c0","hd95_c1","hd95_c2",
        "asd_c0","asd_c1","asd_c2",
        "dice_psfh","hd95_psfh","asd_psfh"
    ]}

    with torch.no_grad():
        for batch in dl:
            x = batch["LD"].to(device)
            y = batch["FD"].cpu().numpy()
            names = batch["case_name"]

            pred = torch.argmax(model(x), dim=1).cpu().numpy()

            for i, case in enumerate(names):
                spacing, orig_hw = _read_spacing(dataroot, args.split, case)
                row = {"case_name": case}

                for c in [0,1,2]:
                    gt = (y[i] == c)
                    pr = (pred[i] == c)
                    d = _dice(gt, pr)
                    dAB, dBA = _directed_surface_dists_mm(gt, pr, spacing)
                    hd, asd = _hd95_asd(dAB, dBA)

                    row[f"dice_c{c}"] = d
                    row[f"hd95_c{c}"] = hd
                    row[f"asd_c{c}"] = asd

                    sums[f"dice_c{c}"].append(d)
                    sums[f"hd95_c{c}"].append(hd)
                    sums[f"asd_c{c}"].append(asd)

                gt_p = _to_psfh(y[i])
                pr_p = _to_psfh(pred[i])
                d = _dice(gt_p, pr_p)
                dAB, dBA = _directed_surface_dists_mm(gt_p, pr_p, spacing)
                hd, asd = _hd95_asd(dAB, dBA)

                row["dice_psfh"] = d
                row["hd95_psfh"] = hd
                row["asd_psfh"] = asd

                sums["dice_psfh"].append(d)
                sums["hd95_psfh"].append(hd)
                sums["asd_psfh"].append(asd)

                results.append(row)

    # Save CSV
    csv_path = log_dir / args.save_csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Log averages
    logger.info("========== DATASET AVERAGES ==========")
    for k, v in sums.items():
        v = [x for x in v if np.isfinite(x)]
        logger.info(f"{k}: {np.mean(v):.6f}")

    logger.info(f"Saved CSV → {csv_path}")


if __name__ == "__main__":
    main()
