"""
Loss functions for Radio Map Prediction.

Extensible interface for adding new loss components.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Standard MSE loss for radio map prediction."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss(pred, target)


class L1Loss(nn.Module):
    """L1 loss."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss(pred, target)


class CombinedLoss(nn.Module):
    """Combined loss with configurable weights."""

    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            self.weights = weights

    def forward(self, pred, target):
        total = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(pred, target)
        return total


class PhysicsWeightedL1Loss(nn.Module):
    """
    L1 loss weighted by a physics-based spatial mask.
    Upweights LoS corridor and near-Tx regions.

    Args:
        alpha: weight for the physics-weighted component (0~1).
               total_loss = (1-alpha)*L1 + alpha*weighted_L1
    """

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, weight_map=None):
        l1 = torch.mean(torch.abs(pred - target))

        if weight_map is None:
            return l1

        weighted_l1 = torch.mean(weight_map * torch.abs(pred - target))
        return (1 - self.alpha) * l1 + self.alpha * weighted_l1


def build_loss(config):
    """Build loss function from config."""
    loss_name = config.get("loss", {}).get("primary", "mse")

    if loss_name == "mse":
        return MSELoss()
    elif loss_name == "l1":
        return L1Loss()
    elif loss_name == "mse_l1":
        return CombinedLoss([MSELoss(), L1Loss()], weights=[1.0, 0.1])
    elif loss_name == "physics_weighted_l1":
        alpha = config.get("loss", {}).get("physics_alpha", 0.3)
        return PhysicsWeightedL1Loss(alpha=alpha)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
