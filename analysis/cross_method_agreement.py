"""
Cross-method agreement analysis.

Compares IG / Grad-CAM / Occlusion explanations on the same samples
to assess whether different explanation methods agree on important regions.

Outputs:
  - Top-k overlap (IoU)
  - Spearman rank correlation
  - Cosine similarity
  - Grouped by low-error vs high-error samples
"""

import numpy as np
from scipy.stats import spearmanr


class CrossMethodAgreement:
    def __init__(self):
        pass

    def pairwise_topk_overlap(self, map1, map2, top_k_percent=20):
        """IoU between top-k regions of two explanation maps."""
        total = map1.size
        k = max(1, int(total * top_k_percent / 100))

        flat1 = map1.flatten()
        flat2 = map2.flatten()
        top1 = set(np.argpartition(flat1, -k)[-k:])
        top2 = set(np.argpartition(flat2, -k)[-k:])

        intersection = len(top1 & top2)
        union = len(top1 | top2)
        return intersection / (union + 1e-8)

    def pairwise_spearman(self, map1, map2):
        """Spearman rank correlation between two explanation maps."""
        corr, _ = spearmanr(map1.flatten(), map2.flatten())
        return float(corr) if not np.isnan(corr) else 0.0

    def pairwise_cosine(self, map1, map2):
        """Cosine similarity between two flattened explanation maps."""
        f1 = map1.flatten().astype(np.float64)
        f2 = map2.flatten().astype(np.float64)
        dot = np.dot(f1, f2)
        norm = np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8
        return float(dot / norm)

    def compute_all_pairs(self, explanations, top_k_percent=20):
        """
        Compute all pairwise metrics between explanation methods.

        Args:
            explanations: dict of {method_name: (H, W) array}

        Returns:
            results: dict of {(method1, method2): {overlap, spearman, cosine}}
        """
        methods = list(explanations.keys())
        results = {}

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                key = f"{m1}_vs_{m2}"
                results[key] = {
                    "topk_overlap": self.pairwise_topk_overlap(
                        explanations[m1], explanations[m2], top_k_percent
                    ),
                    "spearman": self.pairwise_spearman(explanations[m1], explanations[m2]),
                    "cosine": self.pairwise_cosine(explanations[m1], explanations[m2]),
                }

        return results

    def compute_summary(self, all_sample_results):
        """
        Aggregate pairwise metrics across samples.

        Args:
            all_sample_results: list of dicts from compute_all_pairs

        Returns:
            summary: dict of {pair: {metric: {mean, std}}}
        """
        if not all_sample_results:
            return {}

        pairs = list(all_sample_results[0].keys())
        metrics = list(all_sample_results[0][pairs[0]].keys())

        summary = {}
        for pair in pairs:
            summary[pair] = {}
            for metric in metrics:
                values = [r[pair][metric] for r in all_sample_results]
                summary[pair][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        return summary

    def compute_grouped_summary(self, all_sample_results, rmses, error_threshold=None):
        """
        Group results by low-error vs high-error samples.

        Args:
            all_sample_results: list of pairwise results
            rmses: list of RMSE values (same order)
            error_threshold: if None, use median

        Returns:
            grouped: {"low_error": summary, "high_error": summary}
        """
        if error_threshold is None:
            error_threshold = float(np.median(rmses))

        low_indices = [i for i, r in enumerate(rmses) if r <= error_threshold]
        high_indices = [i for i, r in enumerate(rmses) if r > error_threshold]

        low_results = [all_sample_results[i] for i in low_indices]
        high_results = [all_sample_results[i] for i in high_indices]

        return {
            "threshold": error_threshold,
            "n_low_error": len(low_indices),
            "n_high_error": len(high_indices),
            "low_error": self.compute_summary(low_results),
            "high_error": self.compute_summary(high_results),
        }
