"""
Generate baseline vs improved comparison figures for enhanced XAI evaluation.

Reads both summary JSONs and produces side-by-side comparison plots.

Usage:
    conda run -n pytorch python3 scripts/generate_enhanced_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(tag):
    path = f"outputs/enhanced_xai/enhanced_xai_summary_{tag}.json"
    with open(path) as f:
        return json.load(f)


def plot_rmse_comparison(bl, imp, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["Baseline\n(L1, 50ep)", "Improved\n(Physics-L1, 20ep)"]
    rmses = [bl["rmse_mean"], imp["rmse_mean"]]
    stds = [bl["rmse_std"], imp["rmse_std"]]
    rmses_db = [r * 139 for r in rmses]
    stds_db = [s * 139 for s in stds]

    bars = ax.bar(models, rmses_db, yerr=stds_db, capsize=8, color=["steelblue", "coral"], width=0.5)
    for bar, val in zip(bars, rmses_db):
        ax.annotate(f"{val:.2f} dB", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Test RMSE Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_compare_rmse.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_faithfulness_comparison(bl, imp, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics = ["deletion_auc", "insertion_auc", "deletion_monotonicity", "insertion_monotonicity"]
    labels = ["Del. AUC", "Ins. AUC", "Del. Mono.", "Ins. Mono."]

    bl_vals = [bl["faithfulness"][m]["mean"] for m in metrics]
    bl_stds = [bl["faithfulness"][m]["std"] for m in metrics]
    imp_vals = [imp["faithfulness"][m]["mean"] for m in metrics]
    imp_stds = [imp["faithfulness"][m]["std"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    axes[0].bar(x - width / 2, bl_vals, width, yerr=bl_stds, label="Baseline", color="steelblue", capsize=4)
    axes[0].bar(x + width / 2, imp_vals, width, yerr=imp_stds, label="Improved", color="coral", capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Faithfulness Metrics")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Extended PAS comparison
    priors = list(bl["extended_pas"].keys())
    pas_metrics = ["iou", "soft_iou", "pearson_corr"]
    pas_labels = ["IoU", "Soft-IoU", "Pearson"]

    x2 = np.arange(len(priors))
    width2 = 0.12
    for i, (metric, label) in enumerate(zip(pas_metrics, pas_labels)):
        bl_v = [bl["extended_pas"][p][metric]["mean"] for p in priors]
        imp_v = [imp["extended_pas"][p][metric]["mean"] for p in priors]
        axes[1].bar(x2 + i * width2 - width2, bl_v, width2, label=f"BL {label}", color=f"C{i}", alpha=0.6)
        axes[1].bar(x2 + i * width2, imp_v, width2, label=f"Imp {label}", color=f"C{i}", alpha=1.0)

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(priors)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Extended PAS Comparison")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Baseline vs Improved: Faithfulness & Physical Alignment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_compare_faithfulness_pas.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_cross_method_comparison(bl, imp, save_dir):
    pairs_bl = bl.get("cross_method_agreement", {})
    pairs_imp = imp.get("cross_method_agreement", {})
    if not pairs_bl:
        return

    pairs = list(pairs_bl.keys())
    metrics = ["topk_overlap", "spearman", "cosine"]
    metric_labels = ["Top-k Overlap", "Spearman", "Cosine"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        bl_vals = [pairs_bl[p][metric]["mean"] for p in pairs]
        imp_vals = [pairs_imp[p][metric]["mean"] for p in pairs]
        bl_stds = [pairs_bl[p][metric]["std"] for p in pairs]
        imp_stds = [pairs_imp[p][metric]["std"] for p in pairs]

        x = np.arange(len(pairs))
        width = 0.35
        axes[i].bar(x - width / 2, bl_vals, width, yerr=bl_stds, label="Baseline", color="steelblue", capsize=4)
        axes[i].bar(x + width / 2, imp_vals, width, yerr=imp_stds, label="Improved", color="coral", capsize=4)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(pairs, rotation=15, ha="right", fontsize=8)
        axes[i].set_title(label)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Cross-Method Agreement: Baseline vs Improved", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_compare_cross_method.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_los_nlos_comparison(bl, imp, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    regions = ["LoS", "NLoS"]
    bl_mass = [bl["los_nlos"]["los_mass_fraction"]["mean"], bl["los_nlos"]["nlos_mass_fraction"]["mean"]]
    imp_mass = [imp["los_nlos"]["los_mass_fraction"]["mean"], imp["los_nlos"]["nlos_mass_fraction"]["mean"]]
    bl_std = [bl["los_nlos"]["los_mass_fraction"]["std"], bl["los_nlos"]["nlos_mass_fraction"]["std"]]
    imp_std = [imp["los_nlos"]["los_mass_fraction"]["std"], imp["los_nlos"]["nlos_mass_fraction"]["std"]]

    x = np.arange(len(regions))
    width = 0.35
    axes[0].bar(x - width / 2, bl_mass, width, yerr=bl_std, label="Baseline", color="steelblue", capsize=5)
    axes[0].bar(x + width / 2, imp_mass, width, yerr=imp_std, label="Improved", color="coral", capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regions)
    axes[0].set_ylabel("Explanation Mass Fraction")
    axes[0].set_title("LoS vs NLoS Explanation Mass")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Deletion impact
    imp_regions_names = list(bl["los_nlos"]["deletion_impact"].keys())
    bl_del = [bl["los_nlos"]["deletion_impact"][r]["mean"] for r in imp_regions_names]
    imp_del = [imp["los_nlos"]["deletion_impact"][r]["mean"] for r in imp_regions_names]
    bl_del_std = [bl["los_nlos"]["deletion_impact"][r]["std"] for r in imp_regions_names]
    imp_del_std = [imp["los_nlos"]["deletion_impact"][r]["std"] for r in imp_regions_names]

    x2 = np.arange(len(imp_regions_names))
    axes[1].bar(x2 - width / 2, bl_del, width, yerr=bl_del_std, label="Baseline", color="steelblue", capsize=5)
    axes[1].bar(x2 + width / 2, imp_del, width, yerr=imp_del_std, label="Improved", color="coral", capsize=5)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(imp_regions_names)
    axes[1].set_ylabel("MSE Increase")
    axes[1].set_title("Region Deletion Impact")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("LoS/NLoS Analysis: Baseline vs Improved", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_compare_los_nlos.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_range_comparison(bl, imp, save_dir):
    bl_int = bl["range_conditioned"]["intensity"]
    imp_int = imp["range_conditioned"]["intensity"]
    ranges = list(bl_int.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    bl_fracs = [bl_int[r]["intensity_fraction"]["mean"] for r in ranges]
    imp_fracs = [imp_int[r]["intensity_fraction"]["mean"] for r in ranges]
    bl_stds = [bl_int[r]["intensity_fraction"]["std"] for r in ranges]
    imp_stds = [imp_int[r]["intensity_fraction"]["std"] for r in ranges]

    x = np.arange(len(ranges))
    width = 0.35
    ax.bar(x - width / 2, bl_fracs, width, yerr=bl_stds, label="Baseline", color="steelblue", capsize=5)
    ax.bar(x + width / 2, imp_fracs, width, yerr=imp_stds, label="Improved", color="coral", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(ranges)
    ax.set_ylabel("Explanation Mass Fraction")
    ax.set_title("Range-Conditioned Explanation Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_compare_range_conditioned.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_summary_table(bl, imp, save_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    rows = [
        ["RMSE (dB)", f"{bl['rmse_mean']*139:.2f}", f"{imp['rmse_mean']*139:.2f}",
         f"{(imp['rmse_mean']-bl['rmse_mean'])*139:.2f} dB"],
        ["Deletion AUC", f"{bl['faithfulness']['deletion_auc']['mean']:.4f}",
         f"{imp['faithfulness']['deletion_auc']['mean']:.4f}", ""],
        ["Insertion AUC", f"{bl['faithfulness']['insertion_auc']['mean']:.4f}",
         f"{imp['faithfulness']['insertion_auc']['mean']:.4f}", ""],
        ["Deletion Monotonicity", f"{bl['faithfulness']['deletion_monotonicity']['mean']:.3f}",
         f"{imp['faithfulness']['deletion_monotonicity']['mean']:.3f}", ""],
    ]

    for prior in bl["extended_pas"]:
        rows.append([
            f"PAS-{prior} IoU",
            f"{bl['extended_pas'][prior]['iou']['mean']:.4f}",
            f"{imp['extended_pas'][prior]['iou']['mean']:.4f}",
            "",
        ])
        rows.append([
            f"PAS-{prior} Pearson",
            f"{bl['extended_pas'][prior]['pearson_corr']['mean']:.4f}",
            f"{imp['extended_pas'][prior]['pearson_corr']['mean']:.4f}",
            "",
        ])

    rows.append([
        "LoS Mass Fraction",
        f"{bl['los_nlos']['los_mass_fraction']['mean']:.3f}",
        f"{imp['los_nlos']['los_mass_fraction']['mean']:.3f}",
        "",
    ])

    pairs = list(bl.get("cross_method_agreement", {}).keys())
    for pair in pairs:
        rows.append([
            f"Agreement {pair}",
            f"{bl['cross_method_agreement'][pair]['topk_overlap']['mean']:.3f}",
            f"{imp['cross_method_agreement'][pair]['topk_overlap']['mean']:.3f}",
            "",
        ])

    col_labels = ["Metric", "Baseline (L1)", "Improved (Physics-L1)", "Delta"]

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Enhanced XAI Evaluation: Baseline vs Improved", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_compare_summary_table.png"), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    save_dir = "outputs/enhanced_xai"
    bl = load_summary("baseline")
    imp = load_summary("improved")

    print("Generating comparison figures...")
    plot_rmse_comparison(bl, imp, save_dir)
    print("  Saved fig_compare_rmse.png")
    plot_faithfulness_comparison(bl, imp, save_dir)
    print("  Saved fig_compare_faithfulness_pas.png")
    plot_cross_method_comparison(bl, imp, save_dir)
    print("  Saved fig_compare_cross_method.png")
    plot_los_nlos_comparison(bl, imp, save_dir)
    print("  Saved fig_compare_los_nlos.png")
    plot_range_comparison(bl, imp, save_dir)
    print("  Saved fig_compare_range_conditioned.png")
    plot_summary_table(bl, imp, save_dir)
    print("  Saved fig_compare_summary_table.png")
    print("Done!")


if __name__ == "__main__":
    main()
