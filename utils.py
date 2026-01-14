import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from models_DAUNet import DAUNet

# ---------------------------
# Config helpers (your style)
# ---------------------------
def _to_jsonable(v):
    if isinstance(v, Path):           return str(v)
    if isinstance(v, torch.device):   return str(v)
    if isinstance(v, (np.integer, np.floating)): return v.item()
    return v

def _save_config(path, cfg: dict):
    path = Path(path)
    cfg_jsonable = {k: _to_jsonable(v) for k, v in cfg.items()}
    with open(path, 'w') as f:
        json.dump(cfg_jsonable, f, indent=2, sort_keys=True)

def _load_config(log_dir: str) -> dict:
    cfg_path = os.path.join(log_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {log_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)


# ---------------------------
# Logging
# ---------------------------
def setup_logger(log_dir: Path, filename: str = "train.log"):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_dir / filename, mode="a")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); sh.setLevel(logging.INFO)

    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def create_log_directory(base_dir):
    # Ensure the base log directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Get a list of existing directories in the base_dir
    existing_dirs = os.listdir(base_dir)
    # Find the highest current log number
    log_numbers = []
    for d in existing_dirs:
        if d.startswith('log_'):
            try:
                # Extract the numeric part and check for format
                num_str = d.split('_')[1]
                log_numbers.append(int(num_str))
            except ValueError:
                pass
    # Determine the next log number
    next_num = 1 if not log_numbers else max(log_numbers) + 1
    log_num = f"{next_num:03d}"  # Ensure a 3-digit number
    # Get current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the new log directory name
    new_log_dir = os.path.join(base_dir, f'log_{log_num}_{timestamp}')
    # Create the new directory
    os.makedirs(new_log_dir)
    return new_log_dir

# ---------------------------
# Loss / metrics
# ---------------------------
def dice_loss_mc(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    C = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, C).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(0,2,3))
    denom = (probs + one_hot).sum(dim=(0,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self, ce_w=1.0, dice_w=1.0, class_weights=None, ignore_index: int = -100):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.ce_w, self.dice_w = ce_w, dice_w
    def forward(self, logits, target):
        return self.ce_w * self.ce(logits, target) + self.dice_w * dice_loss_mc(logits, target)

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor):
    C = logits.shape[1]
    pred = torch.argmax(logits, dim=1)
    pix_acc = (pred == target).float().mean().item()

    dices = []
    eps = 1e-6
    for c in range(C):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p*t).sum()
        denom = p.sum() + t.sum()
        d = (2*inter + eps) / (denom + eps)
        dices.append(d.item())
    mean_dice = float(np.mean(dices)) if dices else 0.0
    return {"pixel_acc": pix_acc, "mean_dice": mean_dice, "per_class_dice": dices}


@torch.no_grad()
def dice_per_class_per_sample(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    logits: [B, C, H, W], target: [B, H, W] (Long)
    returns: [B, C] per-sample Dice (includes background as class 0)
    """
    B, C, H, W = logits.shape
    pred = logits.argmax(dim=1)                       # [B,H,W]
    dices = torch.empty(B, C, device=logits.device)   # [B,C]
    for c in range(C):
        p = (pred == c).view(B, -1).float()
        t = (target == c).view(B, -1).float()
        inter = (p * t).sum(dim=1)                    # [B]
        denom = p.sum(dim=1) + t.sum(dim=1)           # [B]
        d = (2 * inter + eps) / (denom + eps)         # [B]
        # (optional) if you prefer dice=1 when both empty exactly (no eps effect):
        # d = torch.where(denom == 0, torch.ones_like(d), d)
        dices[:, c] = d
    return dices


# ---------------------------
# Model factory
# ---------------------------
def make_model(kind: str, n_channels: int, n_classes: int, base: int, bilinear: bool,
               use_se: bool, rg_reparam_bn: bool, rg_reparam_identity: bool,
               # DAUNet-JBHI-specific args
               daunet_btlnk_kernel: int, daunet_bottleneck_type: str = "deform", attn_type: str = "simam", attn_mode: str = "skip",
               attn_enc: bool = True, attn_btn: bool = True, attn_dec: bool = True,
               deform_enc_first: bool = False, deform_dec_last: bool = False,
               ):
    kind = kind.lower()

    if kind == "daunet":
        return DAUNet(in_channels=n_channels, num_classes=n_classes, btlnk_kernel=daunet_btlnk_kernel)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, log_interval: int, logger):
    model.train()
    running_tot = running_ce = running_dice = 0.0
    n_seen = 0
    for it, batch in enumerate(loader, 1):
        x = batch["LD"].to(device)
        y = batch["FD"].to(device).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        # loss = loss_fn(logits, y)
        ce   = loss_fn.ce(logits, y)           # CE with class weights
        dsc  = dice_loss_mc(logits, y)         # dice in utils.py
        loss = loss_fn.ce_w * ce + loss_fn.dice_w * dsc

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_tot  += loss.item()  * bs
        running_ce   += ce.item()    * bs
        running_dice += dsc.item()   * bs
        n_seen       += bs

        if it % max(1, log_interval) == 0:
            logger.info(
                f"iter {it:05d} | loss {running_tot/n_seen:.4f} | ce {running_ce/n_seen:.4f} | dice {running_dice/n_seen:.4f}"
            )
    return running_tot / max(1, n_seen)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_pixels  = 0

    per_class_sum = None
    n_samples = 0

    for batch in loader:
        x = batch["LD"].to(device)         # [B,3,H,W]
        y = batch["FD"].to(device).long()  # [B,H,W]
        logits = model(x)                   # [B,C,H,W]
        pred = logits.argmax(dim=1)         # [B,H,W]

        # Pixel accuracy as global micro-average
        total_correct += (pred == y).sum().item()
        total_pixels  += y.numel()

        # Per-sample Dice macro-average
        dices_bs = dice_per_class_per_sample(logits, y)   # [B,C]
        if per_class_sum is None:
            per_class_sum = dices_bs.sum(dim=0).detach().cpu().double().numpy()
        else:
            per_class_sum += dices_bs.sum(dim=0).detach().cpu().double().numpy()
        n_samples += x.size(0)

    pixel_acc = total_correct / max(1, total_pixels)
    per_class = (per_class_sum / max(1, n_samples)).tolist() if per_class_sum is not None else []
    mean_dice = float(np.mean(per_class)) if per_class else 0.0
    return {"pixel_acc": pixel_acc, "mean_dice": mean_dice, "per_class_dice": per_class}


# ---------------------------
# Checkpoint
# ---------------------------
def save_ckpt(path: Path, model, optimizer, epoch, best_metric, args_dict):
    ckpt = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
        "args": args_dict,
    }
    torch.save(ckpt, path)

def load_ckpt(path: Path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ------------- class weights helper -------------
def compute_class_weights_from_dataset(dataset, n_classes=3, smooth=1e-6):
    """
    Iterate once over the *training* dataset to count pixels per class.
    Returns torch.float32 weights ∝ 1/freq, normalized so weights.sum() = n_classes.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for i in range(len(dataset)):
        y = dataset[i]['FD'].cpu().numpy().ravel()
        c = np.bincount(y, minlength=n_classes)
        counts += c
    freq = counts.astype(np.float64) + smooth
    inv = freq.sum() / freq
    inv *= (n_classes / inv.sum())  # normalize so sum = n_classes
    return torch.tensor(inv, dtype=torch.float32)