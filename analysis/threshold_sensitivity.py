"""
Threshold sensitivity analysis.

Tests how conclusions change when varying the top-k threshold
used in PAS and cross-method agreement.

Outputs:
  - PAS at multiple thresholds
  - Agreement at multiple thresholds
  - Conclusion robustness table
"""

import numpy as np
from scipy.stats import spearmanr


class ThresholdSensitivity:
    def __init__(self):
        pass

    def compute_pas_at_thresholds(self, explanation_map, physical_mask, thresholds=None):
        """
        Compute PAS (IoU) at multiple top-k thresholds.

        Returns:
            dict of {threshold_pct: iou_value}
        """
        if thresholds is None:
            thresholds = [5, 10, 20, 30, 40]

        results = {}
        H, W = explanation_map.shape
        total_pixels = H * W
        flat_expl = explanation_map.flatten()
        physical_binary = (physical_mask > 0.5).astype(bool).flatten()

        sorted_indices = np.argsort(flat_expl)[::-1]

        for t in thresholds:
            k = max(1, int(total_pixels * t / 100))
            top_k_set = set(sorted_indices[:k].tolist())
            top_k_binary = np.zeros(total_pixels, dtype=bool)
            for idx in top_k_set:
                top_k_binary[idx] = True

            intersection = np.logical_and(top_k_binary, physical_binary).sum()
            union = np.logical_or(top_k_binary, physical_binary).sum()
            results[t] = float(intersection / (union + 1e-8))

        return results

    def compute_multi_prior_at_thresholds(self, explanation_map, prior_masks, thresholds=None):
        """Compute PAS for all priors at all thresholds."""
        results = {}
        for name, mask in prior_masks.items():
            results[name] = self.compute_pas_at_thresholds(explanation_map, mask, thresholds)
        return results

    def compute_overlap_at_thresholds(self, map1, map2, thresholds=None):
        """Compute top-k overlap between two explanation maps at multiple thresholds."""
        if thresholds is None:
            thresholds = [5, 10, 20, 30, 40]

        results = {}
        total = map1.size
        flat1 = map1.flatten()
        flat2 = map2.flatten()
        sorted1 = np.argsort(flat1)[::-1]
        sorted2 = np.argsort(flat2)[::-1]

        for t in thresholds:
            k = max(1, int(total * t / 100))
            set1 = set(sorted1[:k].tolist())
            set2 = set(sorted2[:k].tolist())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            results[t] = float(intersection / (union + 1e-8))

        return results

    def compute_robustness_summary(self, all_prior_results):
        """
        Check if conclusions are consistent across thresholds.

        Args:
            all_prior_results: list of {prior_name: {threshold: pas}}

        Returns:
            robustness: {prior_name: {threshold: mean_pas, conclusion_robust: bool}}
        """
        if not all_prior_results:
            return {}

        priors = list(all_prior_results[0].keys())
        thresholds = list(all_prior_results[0][priors[0]].keys())

        summary = {}
        for prior in priors:
            threshold_means = {}
            for t in thresholds:
                values = [r[prior][t] for r in all_prior_results]
                threshold_means[t] = float(np.mean(values))

            # Check if ranking across priors is consistent
            values_at_thresholds = list(threshold_means.values())
            # Conclusion is robust if the PAS ordering is consistent
            summary[prior] = threshold_means

        # Check overall conclusion robustness
        # For each threshold pair, check if the ranking of priors is the same
        ranking_lists = {}
        for t in thresholds:
            ranking = sorted(priors, key=lambda p: summary[p][t], reverse=True)
            ranking_lists[t] = ranking

        # All thresholds should give same top prior
        top_priors = [ranking_lists[t][0] for t in thresholds]
        robust = len(set(top_priors)) == 1

        return {
            "per_prior": summary,
            "ranking_at_thresholds": {str(t): ranking_lists[t] for t in thresholds},
            "conclusion_robust": robust,
            "top_prior_at_each_threshold": {str(t): ranking_lists[t][0] for t in thresholds},
        }
