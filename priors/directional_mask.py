"""
Directional mask generation.

Creates masks based on antenna directionality and coverage patterns,
simulating the directional propagation characteristics of radio signals.
"""

import numpy as np


def compute_directional_mask(
    tx_position,
    img_size=256,
    n_sectors=8,
    sigma=20.0,
):
    """
    Compute a directional coverage mask based on Tx position.

    Creates a smooth mask that represents signal strength decreasing
    with distance and angular variation.

    Args:
        tx_position: (2,) array [x, y]
        img_size: image size (H = W = img_size)
        n_sectors: number of angular sectors
        sigma: Gaussian decay parameter for distance

    Returns:
        directional_mask: (H, W) numpy array, values in [0, 1]
    """
    H = W = img_size
    tx_x, tx_y = tx_position

    y_coords, x_coords = np.mgrid[0:H, 0:W]

    # Distance from Tx
    dist = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2)

    # Isotropic distance-based decay (simple free-space pathloss model)
    mask = np.exp(-dist / (sigma * 5))
    mask = mask / mask.max()

    return mask.astype(np.float32)


def compute_inverse_distance_mask(tx_position, img_size=256):
    """
    Compute 1/r distance mask (simplified free-space pathloss).

    Args:
        tx_position: (2,) array [x, y]
        img_size: image dimension

    Returns:
        mask: (H, W) numpy array
    """
    H = W = img_size
    tx_x, tx_y = tx_position

    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dist = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2)
    dist = np.maximum(dist, 1.0)  # Avoid division by zero

    mask = 1.0 / dist
    mask = mask / mask.max()

    return mask.astype(np.float32)


def compute_combined_physical_prior(building_map, tx_position, img_size=256):
    """
    Combine multiple physical priors into a single mask.

    Returns:
        combined_mask: (H, W) numpy array
        individual_masks: dict of individual prior masks
    """
    from .los_mask import compute_los_mask_fast
    from .obstruction_mask import compute_obstruction_mask

    los = compute_los_mask_fast(building_map, tx_position)
    obstruction = compute_obstruction_mask(building_map, tx_position)
    directional = compute_directional_mask(tx_position, img_size)

    # Combined: directional base * (1 - obstruction) + LoS bonus
    combined = directional * (1.0 - 0.5 * obstruction) + 0.3 * los
    combined = np.clip(combined, 0, 1)
    combined = combined / (combined.max() + 1e-8)

    individual_masks = {
        "los": los,
        "obstruction": obstruction,
        "directional": directional,
        "combined": combined,
    }

    return combined, individual_masks
