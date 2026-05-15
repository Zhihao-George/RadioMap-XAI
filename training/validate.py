"""
Validation function for Radio Map Prediction model.
Supports both single-GPU and DataParallel models.
"""

import torch
import torch.nn as nn
from torch.amp import autocast


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Run validation and return loss, RMSE, MAE."""
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    num_batches = 0

    for batch in val_loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        total_rmse += torch.sqrt(nn.functional.mse_loss(outputs, targets)).item()
        total_mae += nn.functional.l1_loss(outputs, targets).item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_rmse = total_rmse / max(num_batches, 1)
    avg_mae = total_mae / max(num_batches, 1)

    return avg_loss, avg_rmse, avg_mae
