"""
Obstruction mask generation.

Identifies building regions that obstruct radio signal propagation
between the transmitter and surrounding areas.
"""

import numpy as np


def compute_obstruction_mask(building_map, tx_position, sigma=3.0):
    """
    Compute obstruction mask: identifies buildings that are in the propagation path.

    For each building pixel, compute its "obstruction importance" based on:
    1. Whether it blocks LoS from Tx to areas behind it
    2. Its distance from Tx (closer buildings are more impactful)

    Args:
        building_map: (H, W) numpy array, building pixels > 0
        tx_position: (2,) array [x, y] transmitter position
        sigma: Gaussian decay for distance weighting

    Returns:
        obstruction_mask: (H, W) numpy array, values in [0, 1]
    """
    H, W = building_map.shape
    tx_x, tx_y = tx_position

    # Building binary mask
    building_binary = (building_map > 0.5).astype(np.float32)

    # Distance from Tx
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dist_from_tx = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2)

    # Weight buildings by inverse distance (closer = more important)
    distance_weight = np.exp(-dist_from_tx / (sigma * 30))

    # Angular density: count buildings in each angular sector
    angles = np.arctan2(y_coords - tx_y, x_coords - tx_x)
    n_sectors = 36
    sector_idx = ((angles + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors

    sector_building_density = np.zeros(n_sectors)
    for s in range(n_sectors):
        mask = (sector_idx == s) & (building_binary > 0.5)
        if mask.sum() > 0:
            sector_building_density[s] = 1.0

    # Map sector density back to pixels
    sector_weight = np.zeros((H, W))
    for s in range(n_sectors):
        sector_weight[sector_idx == s] = sector_building_density[s]

    # Combine: building pixels weighted by distance and sector
    obstruction_mask = building_binary * distance_weight * sector_weight

    # Normalize
    if obstruction_mask.max() > 1e-8:
        obstruction_mask = obstruction_mask / obstruction_mask.max()

    return obstruction_mask


def compute_building_proximity_mask(building_map, tx_position, max_distance=50):
    """
    Compute mask of buildings within a certain distance of Tx.

    Args:
        building_map: (H, W) numpy array
        tx_position: (2,) array [x, y]
        max_distance: maximum distance in pixels

    Returns:
        proximity_mask: (H, W) numpy array
    """
    H, W = building_map.shape
    tx_x, tx_y = tx_position

    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dist_from_tx = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2)

    building_binary = (building_map > 0.5).astype(np.float32)
    proximity = (dist_from_tx < max_distance).astype(np.float32)

    return building_binary * proximity
