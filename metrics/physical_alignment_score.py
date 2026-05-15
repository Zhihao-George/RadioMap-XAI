"""
Physical Alignment Score (PAS) metric.

Core idea: A good explanation should align with known physical
propagation regions (LoS path, obstruction areas, etc.).

Supports multiple alignment measures:
  - IoU (original)
  - Soft-IoU
  - Pearson / Spearman correlation
  - Top-k precision / recall
  - Center-of-mass distance
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


class PhysicalAlignmentScore:
    def __init__(self):
        pass

    def compute(self, explanation_map, physical_mask, top_k_percent=20):
        """
        Original PAS: IoU between top-k% explanation regions and physical prior.

        Args:
            explanation_map: (H, W) numpy array, higher = more important
            physical_mask: (H, W) numpy array, 1.0 for physical regions
            top_k_percent: percentage of top explanation pixels

        Returns:
            pas: float, IoU score in [0, 1]
            details: dict with additional statistics
        """
        H, W = explanation_map.shape
        total_pixels = H * W
        k = max(1, int(total_pixels * top_k_percent / 100))

        flat_expl = explanation_map.flatten()
        top_k_indices = np.argpartition(flat_expl, -k)[-k:]
        explanation_binary = np.zeros(total_pixels, dtype=bool)
        explanation_binary[top_k_indices] = True
        explanation_binary = explanation_binary.reshape(H, W)

        physical_binary = (physical_mask > 0.5).astype(bool)
        if physical_binary.ndim == 1:
            physical_binary = physical_binary.reshape(H, W)

        intersection = np.logical_and(explanation_binary, physical_binary).sum()
        union = np.logical_or(explanation_binary, physical_binary).sum()

        pas = float(intersection / (union + 1e-8))

        details = {
            "top_k_percent": top_k_percent,
            "n_explanation_pixels": int(explanation_binary.sum()),
            "n_physical_pixels": int(physical_binary.sum()),
            "intersection": int(intersection),
            "union": int(union),
            "precision": float(intersection / (explanation_binary.sum() + 1e-8)),
            "recall": float(intersection / (physical_binary.sum() + 1e-8)),
        }

        return pas, details

    def compute_extended(self, explanation_map, physical_mask, top_k_percent=20):
        """
        Extended PAS with multiple alignment measures.

        Returns:
            dict with:
                iou, soft_iou, pearson_corr, spearman_corr,
                precision, recall, center_of_mass_distance
        """
        H, W = explanation_map.shape
        total_pixels = H * W
        k = max(1, int(total_pixels * top_k_percent / 100))

        flat_expl = explanation_map.flatten()
        top_k_indices = np.argpartition(flat_expl, -k)[-k:]
        explanation_binary = np.zeros(total_pixels, dtype=bool)
        explanation_binary[top_k_indices] = True
        explanation_binary = explanation_binary.reshape(H, W)

        physical_binary = (physical_mask > 0.5).astype(bool)
        if physical_binary.ndim == 1:
            physical_binary = physical_binary.reshape(H, W)

        # 1. Hard IoU
        intersection = np.logical_and(explanation_binary, physical_binary).sum()
        union = np.logical_or(explanation_binary, physical_binary).sum()
        iou = float(intersection / (union + 1e-8))

        # 2. Soft IoU: use continuous explanation values
        expl_norm = explanation_map / (explanation_map.max() + 1e-8)
        prior_norm = physical_mask / (physical_mask.max() + 1e-8)
        soft_intersection = np.sum(expl_norm * prior_norm)
        soft_union = np.sum(np.maximum(expl_norm, prior_norm))
        soft_iou = float(soft_intersection / (soft_union + 1e-8))

        # 3. Pearson correlation
        flat_expl_norm = expl_norm.flatten()
        flat_prior_norm = prior_norm.flatten()
        try:
            pearson_corr, _ = pearsonr(flat_expl_norm, flat_prior_norm)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        except Exception:
            pearson_corr = 0.0

        # 4. Spearman correlation
        try:
            spearman_corr, _ = spearmanr(flat_expl_norm, flat_prior_norm)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except Exception:
            spearman_corr = 0.0

        # 5. Top-k precision / recall
        precision = float(intersection / (explanation_binary.sum() + 1e-8))
        recall = float(intersection / (physical_binary.sum() + 1e-8))

        # 6. Center-of-mass distance
        expl_com = self._center_of_mass(explanation_map)
        prior_com = self._center_of_mass(physical_mask)
        diag = np.sqrt(H ** 2 + W ** 2)
        com_distance = float(np.linalg.norm(np.array(expl_com) - np.array(prior_com)) / diag)

        return {
            "iou": iou,
            "soft_iou": soft_iou,
            "pearson_corr": float(pearson_corr),
            "spearman_corr": float(spearman_corr),
            "precision": precision,
            "recall": recall,
            "center_of_mass_distance": com_distance,
            "top_k_percent": top_k_percent,
        }

    def _center_of_mass(self, mask):
        """Compute center of mass of a 2D array."""
        total = mask.sum() + 1e-8
        y_coords, x_coords = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        cx = float((x_coords * mask).sum() / total)
        cy = float((y_coords * mask).sum() / total)
        return (cx, cy)

    def compute_multi_prior(self, explanation_map, prior_masks, top_k_percent=20):
        """Compute PAS for multiple physical priors (original IoU only)."""
        scores = {}
        for name, mask in prior_masks.items():
            pas, details = self.compute(explanation_map, mask, top_k_percent)
            scores[name] = {"pas": pas, "details": details}
        return scores

    def compute_multi_prior_extended(self, explanation_map, prior_masks, top_k_percent=20):
        """Compute extended PAS for multiple physical priors."""
        scores = {}
        for name, mask in prior_masks.items():
            scores[name] = self.compute_extended(explanation_map, mask, top_k_percent)
        return scores
