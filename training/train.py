"""
Training pipeline for radio map prediction.
"""

import sys
import os

# Ensure imports work when launched from the repository root or this file's directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import yaml
import time
import argparse
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from model.radio_map_model import Restormer
from datasets.radiomapseer_dataset import get_dataloaders
from losses import build_loss
from training.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description="Train a radio map prediction model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--subset", type=float, default=1.0, help="Use fraction of training data")
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs to use, e.g. '0,1,2,3'")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path, is_ddp=False):
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(state, path)


def train(config, resume_path=None, subset_frac=1.0, gpu_ids=None):
    # ---- Setup ----
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np_rng = __import__("numpy").random.RandomState(seed)

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

    # ---- Dataloaders ----
    train_loader, val_loader, _ = get_dataloaders(config)

    if subset_frac < 1.0:
        n_train = int(len(train_loader.dataset) * subset_frac)
        train_indices = torch.randperm(len(train_loader.dataset))[:n_train]
        train_loader = DataLoader(
            Subset(train_loader.dataset, train_indices.tolist()),
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
            drop_last=True,
        )
        n_val = int(len(val_loader.dataset) * subset_frac)
        val_indices = torch.randperm(len(val_loader.dataset))[:n_val]
        val_loader = DataLoader(
            Subset(val_loader.dataset, val_indices.tolist()),
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
        )

    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # ---- Model ----
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

    # Multi-GPU
    if use_multi_gpu:
        if gpu_ids:
            model = nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs")
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ---- Loss, optimizer, scheduler ----
    criterion = build_loss(config)
    base_lr = float(config["training"]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["T_max"]), eta_min=float(config["training"]["eta_min"])
    )

    # ---- Tensorboard & checkpoint dirs ----
    log_dir = config["output"]["log_dir"]
    ckpt_dir = config["output"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        # Handle DataParallel state_dict wrapping
        state_dict = ckpt["model_state_dict"]
        if use_multi_gpu and not any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not use_multi_gpu and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        # Only restore optimizer/scheduler if not warm-starting from a different setup
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
            except Exception as e:
                print(f"Could not restore optimizer state ({e}), starting fresh optimizer")
        else:
            print("Warm-starting from checkpoint (fresh optimizer)")

    # ---- AMP ----
    scaler = GradScaler("cuda")

    # ---- Gradient accumulation ----
    grad_accum_steps = config["training"].get("grad_accum_steps", 1)
    effective_batch = config["training"]["batch_size"] * grad_accum_steps * (n_gpus if use_multi_gpu else 1)
    print(f"Batch size: {config['training']['batch_size']} x GPU={n_gpus if use_multi_gpu else 1} x accum={grad_accum_steps} = effective {effective_batch}")

    # ---- Training loop ----
    epochs = int(config["training"]["epochs"])
    grad_clip = float(config["training"]["grad_clip"])

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / grad_accum_steps

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
                    f"Loss: {loss.item() * grad_accum_steps:.6f} "
                    f"({(batch_idx+1)*config['training']['batch_size']*(n_gpus if use_multi_gpu else 1)}/{len(train_loader.dataset)})"
                )

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # ---- Validate ----
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion, device)

        # ---- Log ----
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_rmse", val_rmse, epoch)
        writer.add_scalar("Metrics/val_mae", val_mae, epoch)
        writer.add_scalar("LR", lr, epoch)

        print(
            f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s) "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} | "
            f"Val MAE: {val_mae:.6f} | LR: {lr:.2e}"
        )

        # ---- Save checkpoints ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                            os.path.join(ckpt_dir, "best_model.pth"), is_ddp=use_multi_gpu)
            print(f"  -> New best model saved (val_loss: {best_val_loss:.6f})")

        # Save epoch checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                            os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth"), is_ddp=use_multi_gpu)

        scheduler.step()

    # ---- Final save ----
    save_checkpoint(model, optimizer, scheduler, epochs - 1, best_val_loss,
                    os.path.join(ckpt_dir, "final_model.pth"), is_ddp=use_multi_gpu)
    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.resume:
        config["training"]["resume"] = args.resume
    train(
        config,
        resume_path=config["training"].get("resume"),
        subset_frac=args.subset,
        gpu_ids=args.gpus,
    )
