"""
LoS / NLoS explanation split analysis.

Analyzes how much explanation mass falls in LoS vs NLoS regions,
and how deleting those regions affects prediction error.
"""

import numpy as np
import torch


class LoSNLoSExplanationSplit:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def compute_explanation_mass_split(self, explanation_map, los_mask):
        """
        Compute fraction of explanation mass in LoS vs NLoS regions.

        Args:
            explanation_map: (H, W) array
            los_mask: (H, W) array, 1.0 for LoS

        Returns:
            dict with mass fractions
        """
        total = explanation_map.sum() + 1e-8
        los_mass = float((explanation_map * los_mask).sum())
        nlos_mass = float((explanation_map * (1 - los_mask)).sum())

        return {
            "los_mass_fraction": float(los_mass / total),
            "nlos_mass_fraction": float(nlos_mass / total),
            "los_mean_intensity": float(los_mass / (los_mask.sum() + 1e-8)),
            "nlos_mean_intensity": float(nlos_mass / ((1 - los_mask).sum() + 1e-8)),
            "los_pixel_fraction": float(los_mask.mean()),
        }

    def compute_deletion_impact(self, inputs, explanation_map, los_mask, building_map):
        """
        Measure prediction error when deleting LoS vs NLoS regions.

        Args:
            inputs: (C, H, W) or (B, C, H, W) tensor
            explanation_map: (H, W) array
            los_mask: (H, W) array
            building_map: (H, W) array

        Returns:
            dict with deletion impact
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
        else:
            inputs = torch.from_numpy(inputs).unsqueeze(0)

        inputs = inputs.to(self.device)
        self.model.eval()

        with torch.no_grad():
            baseline_pred = self.model(inputs)

        H, W = explanation_map.shape
        nlos_mask = 1.0 - los_mask
        building_mask = (building_map > 0.5).astype(np.float32)
        boundary_mask = self._compute_boundary_mask(building_map, width=3)
        open_mask = (1.0 - los_mask) * (1.0 - building_mask)

        results = {}
        for region_name, region_mask in [
            ("los", los_mask),
            ("nlos", nlos_mask),
            ("boundary", boundary_mask),
            ("building", building_mask),
            ("open_nlos", open_mask),
        ]:
            mask_tensor = (1.0 - region_mask).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0).unsqueeze(0).to(self.device)
            masked_inputs = inputs * mask_tensor

            with torch.no_grad():
                masked_pred = self.model(masked_inputs)
                error = torch.nn.functional.mse_loss(masked_pred, baseline_pred).item()

            results[region_name] = {
                "deletion_error": float(error),
                "pixel_fraction": float(region_mask.mean()),
            }

        return results

    def _compute_boundary_mask(self, building_map, width=3):
        """Compute mask for pixels near building boundaries."""
        from scipy.ndimage import binary_dilation, binary_erosion
        binary = (building_map > 0.5).astype(bool)
        dilated = binary_dilation(binary, iterations=width)
        eroded = binary_erosion(binary, iterations=width)
        boundary = dilated & ~eroded
        return boundary.astype(np.float32)

    def compute_full_analysis(self, inputs, explanation_map, los_mask, building_map):
        """
        Run full LoS/NLoS split analysis.

        Returns:
            dict with mass_split and deletion_impact
        """
        mass_split = self.compute_explanation_mass_split(explanation_map, los_mask)
        deletion_impact = self.compute_deletion_impact(inputs, explanation_map, los_mask, building_map)

        return {
            "mass_split": mass_split,
            "deletion_impact": deletion_impact,
        }
