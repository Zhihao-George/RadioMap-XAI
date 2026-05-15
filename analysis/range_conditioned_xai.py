"""
Range-conditioned XAI analysis.

Analyzes explanation quality stratified by distance from transmitter:
  - Near (0-32 px)
  - Mid (32-96 px)
  - Far (96+ px)

For each range, computes:
  - Mean explanation intensity
  - PAS (physical alignment)
  - Faithfulness degradation when deleting that range
"""

import numpy as np


class RangeConditionedXAI:
    def __init__(self):
        pass

    def compute_distance_bins(self, tx_position, img_size=256, bins=None):
        """
        Create distance-based binary masks.

        Args:
            tx_position: (2,) array [x, y]
            img_size: image dimension
            bins: list of (name, inner_r, outer_r) tuples

        Returns:
            dict of {name: (H, W) binary mask}
        """
        if bins is None:
            bins = [
                ("near", 0, 32),
                ("mid", 32, 96),
                ("far", 96, img_size),
            ]

        H = W = img_size
        tx_x, tx_y = tx_position
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        dist = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2)

        masks = {}
        for name, inner, outer in bins:
            mask = ((dist >= inner) & (dist < outer)).astype(np.float32)
            masks[name] = mask

        return masks

    def compute_range_explanation_intensity(self, explanation_map, distance_masks):
        """Mean explanation intensity in each distance range."""
        results = {}
        total_intensity = explanation_map.sum() + 1e-8

        for name, mask in distance_masks.items():
            region_expl = explanation_map * mask
            n_pixels = mask.sum() + 1e-8
            results[name] = {
                "mean_intensity": float(region_expl.sum() / n_pixels),
                "intensity_fraction": float(region_expl.sum() / total_intensity),
                "n_pixels": int(mask.sum()),
            }

        return results

    def compute_range_pas(self, explanation_map, physical_mask, distance_masks, top_k_percent=20):
        """PAS computed within each distance range separately."""
        H, W = explanation_map.shape
        total_pixels = H * W
        k = max(1, int(total_pixels * top_k_percent / 100))
        flat_expl = explanation_map.flatten()
        top_k_indices = set(np.argpartition(flat_expl, -k)[-k:].tolist())

        physical_binary = (physical_mask > 0.5).flatten()

        results = {}
        for name, mask in distance_masks.items():
            region_mask = mask.flatten().astype(bool)
            top_in_region = sum(1 for idx in top_k_indices if region_mask[idx])
            physical_in_region = int(np.logical_and(region_mask, physical_binary).sum())
            intersection = sum(
                1 for idx in top_k_indices
                if region_mask[idx] and physical_binary[idx]
            )
            region_expl_count = int(region_mask.sum()) if isinstance(mask, np.ndarray) else 0

            # Precision: of top-k pixels in this region, how many align with physical prior
            precision = intersection / (top_in_region + 1e-8)
            # Recall: of physical prior pixels in this region, how many are in top-k
            recall = intersection / (physical_in_region + 1e-8)

            results[name] = {
                "top_k_in_region": top_in_region,
                "physical_in_region": physical_in_region,
                "intersection": intersection,
                "precision": float(precision),
                "recall": float(recall),
            }

        return results

    def compute_full_analysis(self, explanation_map, physical_masks, tx_position, img_size=256):
        """
        Run full range-conditioned analysis.

        Args:
            explanation_map: (H, W) array
            physical_masks: dict of {name: (H, W) mask}
            tx_position: (2,) array
            img_size: int

        Returns:
            dict with intensity and PAS results per range
        """
        distance_masks = self.compute_distance_bins(tx_position, img_size)
        intensity = self.compute_range_explanation_intensity(explanation_map, distance_masks)

        pas_per_range = {}
        for prior_name, prior_mask in physical_masks.items():
            pas_per_range[prior_name] = self.compute_range_pas(
                explanation_map, prior_mask, distance_masks
            )

        return {
            "intensity": intensity,
            "pas_per_range": pas_per_range,
            "distance_masks": distance_masks,
        }
