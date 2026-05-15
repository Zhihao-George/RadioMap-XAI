"""
Integrated Gradients explainer for Radio Map Prediction model.

Computes pixel-level attribution by integrating gradients along a path
from a baseline (zero input) to the actual input.
"""

import torch
import torch.nn as nn
import numpy as np


class IntegratedGradients:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def _compute_gradients(self, scaled_input, target_channel=0):
        """Compute gradients of output w.r.t. input."""
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        outputs = self.model(scaled_input)
        score = outputs[:, target_channel].sum()
        score.backward()
        return scaled_input.grad.clone()

    def explain(self, inputs, n_steps=50, target_channel=0):
        """
        Compute Integrated Gradients attribution map.

        Args:
            inputs: (B, C, H, W) input tensor
            n_steps: number of integration steps
            target_channel: output channel to explain

        Returns:
            attribution: (B, C, H, W) attribution map
        """
        self.model.eval()

        # Baseline: zero input
        baseline = torch.zeros_like(inputs).to(self.device)

        # Compute gradients at each interpolation step
        all_grads = []
        for i in range(n_steps + 1):
            alpha = float(i) / n_steps
            scaled_input = baseline + alpha * (inputs - baseline)
            grads = self._compute_gradients(scaled_input, target_channel)
            all_grads.append(grads)

        # Average gradients (trapezoidal approximation)
        avg_grads = torch.stack(all_grads).mean(dim=0)

        # IG = (input - baseline) * avg_gradients
        attribution = (inputs - baseline) * avg_grads

        return attribution

    def explain_sample(self, inputs, n_steps=50, target_channel=0):
        """
        Explain a single sample, return absolute attribution summed over input channels.

        Returns:
            attribution_map: (H, W) numpy array
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        # Note: do NOT use torch.no_grad() here - IG needs gradients
        attribution = self.explain(inputs, n_steps, target_channel)

        # Sum absolute attribution over input channels
        attribution_map = attribution.abs().sum(dim=1).squeeze(0).cpu().numpy()
        return attribution_map
