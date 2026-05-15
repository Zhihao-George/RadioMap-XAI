"""
Visualize dataset samples: building map, Tx position, and GT radio map.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.radiomapseer_dataset import RadioMapSeerDataset


def visualize_samples(
    root_dir="./data",
    gain_method="DPM",
    num_samples=6,
    save_dir="./outputs/figures/dataset_samples",
    seed=42,
):
    os.makedirs(save_dir, exist_ok=True)

    dataset = RadioMapSeerDataset(
        root_dir=root_dir,
        gain_method=gain_method,
        split="test",
        seed=seed,
    )

    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Building Map", "Tx Position (Heatmap)", "GT Radio Map", "Overlay"]

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        building = sample["building"].numpy()
        antenna_hm = sample["input"][1].numpy()
        target = sample["target"][0].numpy()
        tx_pos = sample["tx_position"].numpy()

        # Building map
        axes[row, 0].imshow(building, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(col_titles[0] if row == 0 else "")
        axes[row, 0].axis("off")

        # Show the Tx heatmap in a geometry-style coordinate view:
        # x to the right, y upward.
        axes[row, 1].imshow(antenna_hm, cmap="hot", vmin=0, vmax=1, origin="lower")
        axes[row, 1].set_title(col_titles[1] if row == 0 else "")
        axes[row, 1].axis("off")

        # GT radio map
        axes[row, 2].imshow(target, cmap="jet", vmin=0, vmax=1)
        axes[row, 2].set_title(col_titles[2] if row == 0 else "")
        axes[row, 2].axis("off")

        # Overlay: building + Tx + radio map
        axes[row, 3].imshow(building, cmap="gray", alpha=0.5, vmin=0, vmax=1)
        axes[row, 3].imshow(target, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[row, 3].plot(tx_pos[0], tx_pos[1], "r*", markersize=12)
        axes[row, 3].set_title(col_titles[3] if row == 0 else "")
        axes[row, 3].axis("off")

        info_text = f"Map {sample['map_id']}, Tx {sample['tx_idx']}"
        axes[row, 0].set_ylabel(info_text, fontsize=9, rotation=0, labelpad=80)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "dataset_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {num_samples} sample visualizations to {save_path}")

    # Also save individual samples
    for i, idx in enumerate(indices[:3]):
        sample = dataset[idx]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        building = sample["building"].numpy()
        antenna_hm = sample["input"][1].numpy()
        target = sample["target"][0].numpy()
        tx_pos = sample["tx_position"].numpy()

        axes[0].imshow(building, cmap="gray")
        axes[0].set_title("Building Map")
        axes[0].axis("off")

        axes[1].imshow(antenna_hm, cmap="hot", origin="lower")
        axes[1].set_title("Tx Position Heatmap")
        axes[1].axis("off")

        axes[2].imshow(target, cmap="jet")
        axes[2].set_title("GT Radio Map (Gain)")
        axes[2].axis("off")

        plt.suptitle(
            f"Sample: Map {sample['map_id']}, Tx {sample['tx_idx']}, "
            f"Position ({tx_pos[0]:.0f}, {tx_pos[1]:.0f})",
            fontsize=11,
        )
        plt.tight_layout()
        path = os.path.join(save_dir, f"sample_{i}_{sample['map_id']}_{sample['tx_idx']}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved individual sample to {path}")


if __name__ == "__main__":
    visualize_samples()
