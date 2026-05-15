"""
Consistency metric for explanation evaluation.

Core idea: For structurally similar samples (similar building layouts),
the explanation patterns should be consistent.

Metric: Cross-sample explanation correlation for similar inputs.
"""

import torch
import numpy as np


class Consistency:
    def __init__(self):
        pass

    def compute_pairwise(self, explanation_maps):
        """
        Compute pairwise consistency between multiple explanation maps.

        Args:
            explanation_maps: list of (H, W) numpy arrays

        Returns:
            consistency_score: float, mean pairwise correlation
            details: dict
        """
        n = len(explanation_maps)
        if n < 2:
            return 1.0, {"n_maps": n}

        # Flatten and normalize each map
        flat_maps = []
        for emap in explanation_maps:
            flat = emap.flatten()
            flat = flat / (flat.max() + 1e-8)
            flat_maps.append(flat)

        # Pairwise correlation
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                corr = np.corrcoef(flat_maps[i], flat_maps[j])[0, 1]
                correlations.append(corr)

        mean_corr = float(np.mean(correlations))

        details = {
            "n_maps": n,
            "n_pairs": len(correlations),
            "mean_correlation": mean_corr,
            "std_correlation": float(np.std(correlations)),
            "min_correlation": float(np.min(correlations)),
            "max_correlation": float(np.max(correlations)),
        }

        return mean_corr, details

    def compute_spatial_consistency(self, explanation_maps, building_maps=None):
        """
        Compute spatial consistency: do explanations for similar regions
        focus on similar spatial patterns?

        Args:
            explanation_maps: list of (H, W) numpy arrays
            building_maps: optional list of (H, W) building maps for normalization

        Returns:
            score: float
        """
        if len(explanation_maps) < 2:
            return 1.0

        # Compute centroid of each explanation
        centroids = []
        for emap in explanation_maps:
            total = emap.sum() + 1e-8
            y_coords, x_coords = np.mgrid[0:emap.shape[0], 0:emap.shape[1]]
            cx = (x_coords * emap).sum() / total
            cy = (y_coords * emap).sum() / total
            centroids.append((cx, cy))

        # Compute pairwise distance between centroids
        n = len(centroids)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(
                    (centroids[i][0] - centroids[j][0]) ** 2 +
                    (centroids[i][1] - centroids[j][1]) ** 2
                )
                distances.append(dist)

        # Normalize by image diagonal
        H, W = explanation_maps[0].shape
        diag = np.sqrt(H ** 2 + W ** 2)
        norm_distances = np.array(distances) / diag

        consistency = float(1.0 - norm_distances.mean())
        consistency = max(0.0, consistency)

        return consistency

    def compute_cross_domain_consistency(
        self,
        id_explanations,
        ood_explanations,
    ):
        """
        Compute cross-domain explanation consistency.

        Lower drift between ID and OOD explanations indicates better generalization.

        Args:
            id_explanations: list of (H, W) numpy arrays (in-domain)
            ood_explanations: list of (H, W) numpy arrays (out-of-domain)

        Returns:
            drift: float, explanation drift (lower = more consistent)
            details: dict
        """
        # Compute average explanation for each domain
        id_mean = np.mean(id_explanations, axis=0)
        ood_mean = np.mean(ood_explanations, axis=0)

        # L2 drift
        l2_drift = np.sqrt(np.mean((id_mean - ood_mean) ** 2))

        # Cosine similarity
        flat_id = id_mean.flatten()
        flat_ood = ood_mean.flatten()
        cos_sim = np.dot(flat_id, flat_ood) / (
            np.linalg.norm(flat_id) * np.linalg.norm(flat_ood) + 1e-8
        )

        # Correlation
        corr = np.corrcoef(flat_id, flat_ood)[0, 1]

        details = {
            "l2_drift": float(l2_drift),
            "cosine_similarity": float(cos_sim),
            "correlation": float(corr),
            "n_id": len(id_explanations),
            "n_ood": len(ood_explanations),
        }

        return float(l2_drift), details
