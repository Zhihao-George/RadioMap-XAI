"""
Ablation training: Physics-weighted L1 loss.

Same backbone, same hyperparameters, only change the loss function
to include physics-based spatial weighting (LoS + distance decay).

Usage:
    conda run -n pytorch python3 training/train_physics.py --config configs/config_ablation.yaml --gpus "0,1,2,3"
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import yaml
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from model.radio_map_model import Restormer
from datasets.radiomapseer_dataset import get_dataloaders
from losses.loss import PhysicsWeightedL1Loss
from training.validate import validate
from priors.los_mask import compute_los_mask_fast


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation: Physics-weighted training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--full_resume", action="store_true",
                        help="Resume from checkpoint including optimizer/scheduler/epoch (not warmstart)")
    return parser.parse_args()


def compute_physics_weight_map(building, tx_pos, img_size=256):
    """
    Compute a physics-based weight map for each sample.
    Combines LoS mask + distance decay from Tx.
    Returns: (H, W) numpy array, values in [1, max_weight].
    """
    H, W = building.shape

    # LoS mask
    los = compute_los_mask_fast(building, tx_pos, n_directions=360, max_radius=200)

    # Distance decay from Tx
    tx_x, tx_y = tx_pos
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dist = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2) + 1.0
    distance_weight = 1.0 / (1.0 + 0.01 * dist)

    # Combine: LoS regions and near-Tx get higher weight
    weight_map = 1.0 + 2.0 * los + distance_weight
    return weight_map.astype(np.float32)


def train_physics(config, resume_path=None, gpu_ids=None):
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)

    n_gpus = torch.cuda.device_count()
    use_multi_gpu = n_gpus > 1

    if gpu_ids is not None:
        gpu_ids = [int(x) for x in gpu_ids.split(",")]
        use_multi_gpu = len(gpu_ids) > 1
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataloaders
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Model
    model_cfg = config["model"]
    model = Restormer(
        inp_channels=model_cfg["inp_channels"],
        out_channels=model_cfg["out_channels"],
        dim=model_cfg["dim"],
        num_blocks=model_cfg["num_blocks"],
        num_refinement_blocks=model_cfg["num_refinement_blocks"],
        heads=model_cfg["heads"],
        ffn_expansion_factor=model_cfg["ffn_expansion_factor"],
        bias=model_cfg["bias"],
        LayerNorm_type=model_cfg["LayerNorm_type"],
    )

    if use_multi_gpu:
        if gpu_ids:
            model = nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs")
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss: physics-weighted L1
    alpha = config["loss"].get("physics_alpha", 0.3)
    criterion = PhysicsWeightedL1Loss(alpha=alpha)
    print(f"Loss: PhysicsWeightedL1 (alpha={alpha})")

    # Optimizer & scheduler
    base_lr = float(config["training"]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["T_max"]), eta_min=float(config["training"]["eta_min"])
    )

    # Directories
    ckpt_dir = config["output"]["checkpoint_dir"].replace("checkpoints", "improved_checkpoints")
    log_dir = config["output"]["log_dir"] + "_physics"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Resume: warmstart from baseline (load weights only, reset optimizer/scheduler)
    start_epoch = 0
    best_val_loss = float("inf")
    full_resume = config["training"].get("full_resume", False)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        if use_multi_gpu and not any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not use_multi_gpu and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        if full_resume and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"Full resume from {resume_path} (epoch {start_epoch}, best_val_loss={best_val_loss:.6f})")
        else:
            print(f"Warm-started from {resume_path} (fresh optimizer, epoch 0)")

    scaler = GradScaler("cuda")
    grad_accum_steps = config["training"].get("grad_accum_steps", 1)
    epochs = int(config["training"]["epochs"])
    grad_clip = float(config["training"]["grad_clip"])

    print(f"Batch size: {config['training']['batch_size']} x GPU={n_gpus if use_multi_gpu else 1} x accum={grad_accum_steps}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Compute physics weight maps for this batch
            batch_size = inputs.shape[0]
            weight_maps = []
            for i in range(batch_size):
                building = batch["building"][i].numpy()
                tx_pos = batch["tx_position"][i].numpy()
                wm = compute_physics_weight_map(building, tx_pos)
                weight_maps.append(wm)
            weight_tensor = torch.from_numpy(np.stack(weight_maps)).unsqueeze(1).to(device)

            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets, weight_map=weight_tensor) / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % 500 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item() * grad_accum_steps:.6f}"
                )

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Validate
        val_loss, val_rmse, val_mae = validate(model, val_loader, nn.L1Loss(), device)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_rmse", val_rmse, epoch)
        writer.add_scalar("Metrics/val_mae", val_mae, epoch)
        writer.add_scalar("LR", lr, epoch)

        print(
            f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) "
            f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
            f"RMSE: {val_rmse:.6f} ({val_rmse*139:.2f} dB) | MAE: {val_mae:.6f} | LR: {lr:.2e}"
        )

        # Save checkpoints
        is_ddp = use_multi_gpu
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  -> New best (val_loss: {best_val_loss:.6f})")

        if (epoch + 1) % 5 == 0:
            state = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(state, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        scheduler.step()

    # Final save
    state = {
        "epoch": epochs - 1,
        "model_state_dict": model.module.state_dict() if use_multi_gpu else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(state, os.path.join(ckpt_dir, "final_model.pth"))
    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_physics(config, resume_path=args.resume, gpu_ids=args.gpus)
