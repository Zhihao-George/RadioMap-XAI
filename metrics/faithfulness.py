"""
Faithfulness metric for explanation evaluation.

Core idea: If the explanation correctly identifies important regions,
removing those regions should cause the biggest performance drop.

Supports:
  - Deletion curve: remove top-k% pixels, measure error increase
  - Insertion curve: add back top-k% pixels, measure error decrease
  - AUC for both curves
  - Monotonicity statistics
"""

import torch
import numpy as np


class Faithfulness:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def compute(
        self,
        inputs,
        explanation_map,
        top_k_percentages=None,
        target_channel=0,
    ):
        """
        Compute faithfulness by deleting top-k important regions.

        Args:
            inputs: (B, C, H, W) or (C, H, W) input tensor
            explanation_map: (H, W) numpy array, higher = more important
            top_k_percentages: list of percentages to delete, e.g. [5, 10, 20, 30]
            target_channel: output channel

        Returns:
            results: dict mapping k% -> error increase
        """
        if top_k_percentages is None:
            top_k_percentages = [1, 5, 10, 20, 30]

        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        self.model.eval()
        B, C, H, W = inputs.shape

        with torch.no_grad():
            baseline_pred = self.model(inputs)
            baseline_error = torch.nn.functional.mse_loss(
                baseline_pred, torch.zeros_like(baseline_pred)
            ).item()

        flat_expl = explanation_map.flatten()
        total_pixels = flat_expl.shape[0]

        results = {}
        for k_pct in top_k_percentages:
            k = max(1, int(total_pixels * k_pct / 100))

            top_k_indices = np.argpartition(flat_expl, -k)[-k:]

            mask = np.ones((H, W), dtype=np.float32)
            for idx in top_k_indices:
                y, x = divmod(int(idx), W)
                mask[y, x] = 0.0

            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            masked_inputs = inputs * mask_tensor

            with torch.no_grad():
                masked_pred = self.model(masked_inputs)
                masked_error = torch.nn.functional.mse_loss(
                    masked_pred, baseline_pred
                ).item()

            results[k_pct] = masked_error

        return results

    def compute_deletion_curve(self, inputs, explanation_map, n_steps=20, target_channel=0):
        """
        Compute full deletion curve: gradually delete pixels from most to least important.

        Returns:
            dict with:
                auc: area under deletion curve
                curve: list of (fraction_deleted, error)
                monotonicity: fraction of consecutive steps where error increased
                monotonicity_corr: Spearman correlation between fraction and error
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        self.model.eval()
        B, C, H, W = inputs.shape

        with torch.no_grad():
            baseline_pred = self.model(inputs)

        flat_expl = explanation_map.flatten()
        total_pixels = flat_expl.shape[0]
        sorted_indices = np.argsort(flat_expl)[::-1]

        curve = []
        for step in range(n_steps + 1):
            fraction = step / n_steps
            n_delete = int(fraction * total_pixels)

            mask = np.ones((H, W), dtype=np.float32)
            if n_delete > 0:
                delete_indices = sorted_indices[:n_delete]
                for idx in delete_indices:
                    y, x = divmod(int(idx), W)
                    mask[y, x] = 0.0

            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            masked_inputs = inputs * mask_tensor

            with torch.no_grad():
                masked_pred = self.model(masked_inputs)
                error = torch.nn.functional.mse_loss(masked_pred, baseline_pred).item()

            curve.append((fraction, error))

        errors = [e for _, e in curve]
        auc = float(np.trapz(errors, dx=1.0 / n_steps))

        # Monotonicity: fraction of consecutive steps with error increase
        increases = sum(1 for i in range(1, len(errors)) if errors[i] >= errors[i - 1])
        monotonicity = increases / (len(errors) - 1) if len(errors) > 1 else 1.0

        # Spearman correlation between deletion fraction and error
        from scipy.stats import spearmanr
        fractions = [f for f, _ in curve]
        mono_corr, _ = spearmanr(fractions, errors)

        return {
            "auc": float(auc),
            "curve": curve,
            "monotonicity": float(monotonicity),
            "monotonicity_corr": float(mono_corr),
        }

    def compute_insertion_curve(self, inputs, explanation_map, n_steps=20, target_channel=0):
        """
        Compute insertion curve: start from zero, gradually add most important pixels.

        Returns:
            dict with auc, curve, monotonicity, monotonicity_corr
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        self.model.eval()
        B, C, H, W = inputs.shape

        # Full-input prediction as reference
        with torch.no_grad():
            full_pred = self.model(inputs)
            full_error = torch.nn.functional.mse_loss(full_pred, torch.zeros_like(full_pred)).item()

        flat_expl = explanation_map.flatten()
        total_pixels = flat_expl.shape[0]
        sorted_indices = np.argsort(flat_expl)[::-1]

        curve = []
        for step in range(n_steps + 1):
            fraction = step / n_steps
            n_insert = int(fraction * total_pixels)

            # Start from zero, insert most important pixels
            mask = np.zeros((H, W), dtype=np.float32)
            if n_insert > 0:
                insert_indices = sorted_indices[:n_insert]
                for idx in insert_indices:
                    y, x = divmod(int(idx), W)
                    mask[y, x] = 1.0

            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            inserted_inputs = inputs * mask_tensor

            with torch.no_grad():
                inserted_pred = self.model(inserted_inputs)
                error = torch.nn.functional.mse_loss(inserted_pred, full_pred).item()

            curve.append((fraction, error))

        errors = [e for _, e in curve]
        auc = float(np.trapz(errors, dx=1.0 / n_steps))

        # For insertion, error should decrease as more pixels are inserted
        decreases = sum(1 for i in range(1, len(errors)) if errors[i] <= errors[i - 1])
        monotonicity = decreases / (len(errors) - 1) if len(errors) > 1 else 1.0

        from scipy.stats import spearmanr
        fractions = [f for f, _ in curve]
        mono_corr, _ = spearmanr(fractions, errors)

        return {
            "auc": float(auc),
            "curve": curve,
            "monotonicity": float(monotonicity),
            "monotonicity_corr": float(mono_corr),
        }

    def compute_auc(self, inputs, explanation_map, n_steps=20):
        """Backward compatible: deletion AUC only."""
        result = self.compute_deletion_curve(inputs, explanation_map, n_steps)
        return result["auc"], result["curve"]

    def compute_full_protocol(self, inputs, explanation_map, n_steps=20, target_channel=0):
        """
        Run full faithfulness protocol: deletion + insertion curves.

        Returns:
            dict with deletion and insertion results
        """
        deletion = self.compute_deletion_curve(inputs, explanation_map, n_steps, target_channel)
        insertion = self.compute_insertion_curve(inputs, explanation_map, n_steps, target_channel)

        return {
            "deletion": {
                "auc": deletion["auc"],
                "monotonicity": deletion["monotonicity"],
                "monotonicity_corr": deletion["monotonicity_corr"],
            },
            "insertion": {
                "auc": insertion["auc"],
                "monotonicity": insertion["monotonicity"],
                "monotonicity_corr": insertion["monotonicity_corr"],
            },
            "deletion_curve": deletion["curve"],
            "insertion_curve": insertion["curve"],
        }
