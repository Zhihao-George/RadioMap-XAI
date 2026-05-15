"""
Stability metric for explanation evaluation.

Core idea: A good explanation should be stable under small perturbations
of the input. If the input changes slightly, the explanation should not
change dramatically.

Metric: L1 or L2 distance between explanations of original and perturbed inputs.
"""

import torch
import numpy as np


class Stability:
    def __init__(self, explainer, model, device="cuda"):
        self.explainer = explainer
        self.model = model
        self.device = device

    def compute(
        self,
        inputs,
        n_perturbations=10,
        noise_std=0.01,
        tx_jitter=2.0,
    ):
        """
        Compute explanation stability under input perturbations.

        Args:
            inputs: (C, H, W) input tensor
            n_perturbations: number of perturbed versions
            noise_std: Gaussian noise standard deviation
            tx_jitter: Tx position jitter in pixels

        Returns:
            stability_score: float in [0, 1], higher = more stable
            details: dict with perturbation details
        """
        self.model.eval()

        # Get original explanation
        orig_input = inputs.unsqueeze(0).to(self.device) if inputs.dim() == 3 else inputs.to(self.device)
        orig_expl = self.explainer.explain_sample(orig_input)

        distances = []
        for i in range(n_perturbations):
            # Add Gaussian noise
            noise = torch.randn_like(orig_input) * noise_std
            perturbed = torch.clamp(orig_input + noise, 0, 1)

            perturbed_expl = self.explainer.explain_sample(perturbed)

            # L2 distance between explanation maps
            dist = np.sqrt(np.mean((orig_expl - perturbed_expl) ** 2))
            distances.append(dist)

        distances = np.array(distances)

        # Stability = 1 - normalized_distance
        max_possible_dist = np.sqrt(2)  # Max L2 between normalized maps
        mean_dist = distances.mean()
        stability_score = float(1.0 - mean_dist / max_possible_dist)
        stability_score = max(0.0, stability_score)

        details = {
            "mean_l2_distance": float(mean_dist),
            "std_l2_distance": float(distances.std()),
            "max_l2_distance": float(distances.max()),
            "min_l2_distance": float(distances.min()),
            "n_perturbations": n_perturbations,
            "noise_std": noise_std,
        }

        return stability_score, details

    def compute_ssim_stability(self, inputs, n_perturbations=10, noise_std=0.01):
        """
        Compute stability using SSIM between original and perturbed explanations.
        """
        self.model.eval()

        orig_input = inputs.unsqueeze(0).to(self.device) if inputs.dim() == 3 else inputs.to(self.device)
        orig_expl = self.explainer.explain_sample(orig_input)

        ssim_scores = []
        for _ in range(n_perturbations):
            noise = torch.randn_like(orig_input) * noise_std
            perturbed = torch.clamp(orig_input + noise, 0, 1)
            perturbed_expl = self.explainer.explain_sample(perturbed)

            ssim = self._compute_ssim(orig_expl, perturbed_expl)
            ssim_scores.append(ssim)

        return float(np.mean(ssim_scores)), {"mean_ssim": float(np.mean(ssim_scores))}

    @staticmethod
    def _compute_ssim(img1, img2, window_size=7):
        """Simple SSIM computation."""
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        mu1 = _uniform_filter(img1, window_size)
        mu2 = _uniform_filter(img2, window_size)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _uniform_filter(img1 ** 2, window_size) - mu1_sq
        sigma2_sq = _uniform_filter(img2 ** 2, window_size) - mu2_sq
        sigma12 = _uniform_filter(img1 * img2, window_size) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())


def _uniform_filter(img, size=7):
    """Simple uniform averaging filter."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(img, size=size)
