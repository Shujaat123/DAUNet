import os, sys, json, time, logging, argparse
from pathlib import Path
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset_Pubic_Symphysis import PUBIC

from utils import (
    create_log_directory,
    setup_logger,
    _save_config,
    _load_config,
    make_model,
    train_one_epoch,
    evaluate,
    save_ckpt,
    load_ckpt,
    DiceCELoss,
    compute_class_weights_from_dataset,
)

import math
import torchvision.transforms.functional as TF


############### Args ##########################

def str2bool(v):
    return str(v).lower() in {"1","true","t","yes","y"}

def parse_args():
    p = argparse.ArgumentParser(description="Train DAUNet on Dataset: Pubic_Symphysis")

    # ------------------- Data -------------------
    p.add_argument("--dataset", type=str, default='PSFH', choices=["PSFH",],
        help="Which dataset to use: 'PSFH' (Pubic Symphysis Fetal Head).")

    p.add_argument("--val-split", type=str, default='train8020', choices=["train", "test", "train8020"],
        help="For PUBIC (PSFH) dataset: Which split to use for validation: 'train', 'test', 'train8020': Split train dataset into 80/20 for train/val. (Note: No Val split provided in Pubic_Symphysis dataset)")

    p.add_argument("--img-size", type=int, default=256,
        help="Input image size (images and masks will be resized to [img-size,img-size]).")
    p.add_argument("--num-workers", type=int, default=4,
        help="Number of worker processes for the DataLoader.")

    p.add_argument("--btlnk_kernel", type=int, default=3,
        help="For DAUNet only: Kernal size in bottleneck Conv (default 3 for early log_(1-3)).")

    # ------------------- Model -------------------
    p.add_argument("--model", type=str, default="daunet", choices=["daunet",],
        help="Model variant to use: 'daunet' (Deformable-UNet).")
    p.add_argument("--n-channels", type=int, default=3,
        help="Number of input channels (default 3 for 3-slice input).")
    p.add_argument("--n-classes", type=int, default=3,
        help="For PUBIC (PSFH) dataset only: Number of segmentation classes (e.g., 3 for {0,1,2}).")
    p.add_argument("--base", type=int, default=64,
        help="Base number of channels in U-Net encoder. Default: 64 for UNet, 32 for Ghost/RepGhost.")
    p.add_argument("--bilinear", type=str2bool, default=True,
        help="Use bilinear upsampling in decoder instead of transposed conv.")
    p.add_argument("--use-se", type=str2bool, default=True,
        help="Enable Squeeze-and-Excitation (SE) modules in RepGhost blocks.")
    p.add_argument("--rg-no-bn-id", type=str2bool, default=False,
        help="Disable BN identity branch in RepGhostModule (default is enabled).")
    p.add_argument("--rg-use-pure-id", type=str2bool, default=False,
        help="Enable pure identity branch (no BN) in RepGhostModule.")
    # ------------------- Training -------------------
    p.add_argument("--gpu", type=int, default=0, help="CUDA GPU to be used.")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation.")
    p.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")

    p.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"], help="LR scheduler type.")
    p.add_argument("--lr-factor", type=float, default=0.5, help="Multiply LR by this on plateau.")
    p.add_argument("--lr-patience", type=int, default=10, help="Epochs without improvement before LR drop.")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Lower LR bound.")
    p.add_argument("--cosine-tmax", type=int, default=0, help="Cosine T_max in epochs (0 -> use args.epochs).")
    p.add_argument("--cosine-eta-min", type=float, default=1e-6, help="Cosine minimum LR.")

    p.add_argument("--augment", type=str2bool, default=True, help="Enable mild, mask-safe augmentations for training.")
    p.add_argument("--use-class-weights", type=str2bool, default=True, help="Use inverse-frequency class weights for CE (sum=n_classes).")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization) coefficient.")
    p.add_argument("--ce-w", type=float, default=0.5, help="Weight for CrossEntropy component in loss.")
    p.add_argument("--dice-w", type=float, default=1.0, help="Weight for Dice component in loss.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--patience", type=int, default=40, help="Early stopping patience (epochs without val improvement before stopping). 0 disables.")
    p.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement in mean Dice to reset patience.")    
    # ------------------- IO / Logging -------------------
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from.")
    p.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs (0 disables).")
    p.add_argument("--log-interval", type=int, default=50, help="Logging interval (in training iterations).")

    p.add_argument("--eval", type=str2bool, default=True, help="Perform Evaluation at the end of training")
    return p.parse_args()

args = parse_args()
base_dir = f'/home/muhammad_jabbar/diffusion/code/ghost_unet/logs_{args.dataset}/{args.model}'
args.out_dir = create_log_directory(base_dir=base_dir)
args.out_dir = Path(args.out_dir)

device = torch.device(f'cuda:{args.gpu}')
logger = setup_logger(args.out_dir)

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Save and log full args (to file and terminal)
args_dict = vars(args).copy()
for k, v in list(args_dict.items()): # sanitize anything not JSON-serializable
    if isinstance(v, Path):
        args_dict[k] = str(v)
    elif isinstance(v, torch.device):
        args_dict[k] = str(v)
_save_config(args.out_dir / "config.json", args_dict)
logger.info("========== Config (args) ==========")
logger.info(json.dumps(args_dict, indent=2, sort_keys=True))
logger.info("===================================")
logger.info(f"Device: {device}")

### Data ###

if args.dataset == "PSFH":
    args.dataroot = '/home/muhammad_jabbar/diffusion/code/Fast-DDPM/data/Pubic_Symphysis_Fetal_Head_Segmentation_and_Angle_of_Progression'
    if args.val_split == "train":
        train_ds = PUBIC(args.dataroot, args.img_size, split="train", augment=args.augment)
        val_ds   = PUBIC(args.dataroot, args.img_size, split="train")
        logger.info("Using 'train' split for both training and validation.")
    elif args.val_split == "test":
        train_ds = PUBIC(args.dataroot, args.img_size, split="train", augment=args.augment)
        val_ds   = PUBIC(args.dataroot, args.img_size, split="test")
        logger.info("Using 'test' split for validation.")
    elif args.val_split == "train8020":
        base_train = PUBIC(args.dataroot, args.img_size, split="train", augment=False) # base dataset WITHOUT augmentation for stable indexing
        n = len(base_train)
        n_val = max(1, int(round(0.2 * n)))
        # Deterministic shuffle
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=g).tolist()
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]
        # Two *separate* dataset instances from train, with no val augmentation:
        train_ds = Subset(PUBIC(args.dataroot, args.img_size, split="train", augment=args.augment), train_idx)
        val_ds   = Subset(PUBIC(args.dataroot, args.img_size, split="train"), val_idx)
        logger.info("Using 'train' sub-split for training (80pc) and validation(20pc).")
    # elif args.val_split == "nl_train8020": # no data leakage train8020: train split into 80pc for train & 20pc for validation with split by case/patient
    else:
        raise ValueError(f"Invalid --val-split '{args.val_split}'")
else:
    raise ValueError(f"{args.dataset} - Dataset not implemented!")

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

# Model / Optim / Loss
model = make_model(
    kind=args.model,
    n_channels=args.n_channels,
    n_classes=args.n_classes,
    base=args.base,
    bilinear=args.bilinear,
    use_se=args.use_se,
    rg_reparam_bn=(not args.rg_no_bn_id),
    rg_reparam_identity=args.rg_use_pure_id,
    daunet_btlnk_kernel=args.btlnk_kernel,
).to(device)

# Model parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")
logger.info(f"Model parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = None
if args.scheduler == "plateau":
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  factor=args.lr_factor, patience=args.lr_patience,
                                  min_lr=args.min_lr, verbose=True)
elif args.scheduler == "cosine":
    from torch.optim.lr_scheduler import CosineAnnealingLR
    T_max = args.cosine_tmax or args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.cosine_eta_min)

class_weights = None

# loss_fn = DiceCELoss(ce_w=args.ce_w, dice_w=args.dice_w)
if args.use_class_weights:
    if args.dataset == "PSFH":
        cw_ds = PUBIC(args.dataroot, args.img_size, split="train", augment=False)
    class_weights = compute_class_weights_from_dataset(cw_ds, n_classes=args.n_classes).to(device)
    logger.info(f"CE class weights (sum={class_weights.sum().item():.3f}): {class_weights.tolist()}")
loss_fn = DiceCELoss(ce_w=args.ce_w, dice_w=args.dice_w, class_weights=class_weights).to(device)


# Ckpt paths
latest = args.out_dir / "ckpt_latest.pth"
best   = args.out_dir / "ckpt_best.pth"
start_epoch = 1
best_metric = -1.0
epochs_since_improve = 0

# Resume
if args.resume and os.path.isfile(args.resume):
    ckpt = load_ckpt(Path(args.resume), model, optimizer, map_location=device)
    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", -1.0)
    logger.info(f"Resumed from {args.resume} @ epoch {start_epoch-1} | best={best_metric:.4f}")

# Train loop
for epoch in range(start_epoch, args.epochs + 1):
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, args.log_interval, logger)
    val = evaluate(model, val_loader, device)

    # per_class_dice = val['per_class_dice']
    # fg_mean = float(np.mean(per_class_dice[1:])) # foreground-only mean Dice

    fg_mean = None
    if "per_class_dice" in val and len(val["per_class_dice"]) >= (args.n_classes-1):
        fg_mean = float(np.mean(val["per_class_dice"][1:]))
    monitor_metric = fg_mean if fg_mean is not None else float(val["mean_dice"])

    if scheduler is not None:
        if args.scheduler == "plateau":
            scheduler.step(monitor_metric)
        else:  # cosine
            scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    logger.info(
        f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} | "
        f"val_pixAcc {val['pixel_acc']:.4f} | val_meanDice {val['mean_dice']:.4f} | "
        f"per_class {['%.3f' % d for d in val['per_class_dice']]} | val_meanDice_foreground {fg_mean:.4f} | lr={current_lr:.6f}"
    )

    # Save latest
    save_ckpt(latest, model, optimizer, epoch, best_metric, vars(args))

    # improved = (val["mean_dice"] > best_metric + args.min_delta) # based on all-class mean
    improved = (monitor_metric > best_metric + args.min_delta) # based on val_meanDice_foreground
    if improved:
        # best_metric = val["mean_dice"]
        best_metric = monitor_metric
        epochs_since_improve = 0
        save_ckpt(best, model, optimizer, epoch, best_metric, vars(args))
        logger.info(f"★ New best (meanDice={best_metric:.4f}) → {best}")
    else:
        epochs_since_improve += 1
        logger.info(f"No improvement. patience {epochs_since_improve}/{args.patience}")

    logger.info(f"Epoch {epoch:03d} time: {time.time()-t0:.1f}s | lr={optimizer.param_groups[0]['lr']:.6f}")

    # Optional per-epoch snapshot
    if args.save_every and (epoch % args.save_every == 0):
        snap = args.out_dir / f"ckpt_epoch{epoch:04d}.pth"
        save_ckpt(snap, model, optimizer, epoch, best_metric, vars(args))

    # Early stopping check
    if args.patience > 0 and epochs_since_improve >= args.patience:
        logger.info(f"Early stopping triggered (no improvement for {args.patience} epochs).")
        break

logger.info("Training complete.")

# Optional: run evaluate.py on BEST checkpoint only
if bool(args.eval):
    logger.info("Running evaluate_PFSH_comprehensive.py on best checkpoint...")
    try:
        subprocess.run(
            [sys.executable, "evaluate_PFSH_comprehensive.py", "--log-dir", str(args.out_dir)],
            check=True
        )
    except Exception as e:
        logger.exception(f"evaluate_PFSH_comprehensive.py failed: {e}")


