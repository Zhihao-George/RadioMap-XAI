"""
Occlusion Sensitivity explainer for Radio Map Prediction model.

Systematically occludes regions of the input and measures the change
in model output to determine the importance of each region.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class OcclusionSensitivity:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def explain(
        self,
        inputs,
        window_size=16,
        stride=8,
        occlusion_value=0.0,
        target_channel=0,
    ):
        """
        Compute Occlusion Sensitivity map.

        Args:
            inputs: (B, C, H, W) input tensor
            window_size: size of occlusion window
            stride: stride for sliding window
            occlusion_value: value to fill occluded region
            target_channel: output channel to measure

        Returns:
            sensitivity: (B, 1, H, W) attribution map
        """
        self.model.eval()
        B, C, H, W = inputs.shape

        # Get baseline output
        with torch.no_grad():
            baseline_output = self.model(inputs)
            baseline_score = baseline_output[:, target_channel].sum().item()

        # Initialize sensitivity map
        sensitivity = torch.zeros(B, 1, H, W, device=self.device)
        count = torch.zeros(B, 1, H, W, device=self.device)

        # Slide occlusion window
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + window_size, H)
                x_end = min(x + window_size, W)

                # Create occluded input
                occluded = inputs.clone()
                occluded[:, :, y:y_end, x:x_end] = occlusion_value

                with torch.no_grad():
                    occluded_output = self.model(occluded)
                    occluded_score = occluded_output[:, target_channel].sum().item()

                # Sensitivity = change in output
                delta = abs(baseline_score - occluded_score)

                sensitivity[:, :, y:y_end, x:x_end] += delta
                count[:, :, y:y_end, x:x_end] += 1

        # Average overlapping regions
        sensitivity = sensitivity / (count + 1e-8)

        # Normalize to [0, 1]
        s_min = sensitivity.min()
        s_max = sensitivity.max()
        if s_max - s_min > 1e-8:
            sensitivity = (sensitivity - s_min) / (s_max - s_min)

        return sensitivity

    def explain_sample(
        self,
        inputs,
        window_size=16,
        stride=8,
        occlusion_value=0.0,
        target_channel=0,
    ):
        """
        Explain a single sample.

        Returns:
            sensitivity: (H, W) numpy array
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        sensitivity = self.explain(
            inputs, window_size, stride, occlusion_value, target_channel
        )
        return sensitivity[0, 0].cpu().numpy()
