"""
Line-of-Sight (LoS) mask generation.

Computes the LoS corridor between Tx position and all pixels in the map.
The LoS mask identifies regions that have direct visibility to the transmitter,
which is a key physical prior in radio propagation.
"""

import numpy as np


def compute_los_mask(building_map, tx_position, bandwidth=3):
    """
    Compute LoS mask: for each pixel, check if the line from Tx to that pixel
    is free of building obstructions.

    Args:
        building_map: (H, W) numpy array, building pixels > 0
        tx_position: (2,) array [x, y] transmitter position
        bandwidth: half-width of LoS corridor in pixels

    Returns:
        los_mask: (H, W) numpy array, 1.0 for LoS, 0.0 for NLoS
    """
    H, W = building_map.shape
    tx_x, tx_y = tx_position

    # Create distance map from Tx
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    los_mask = np.zeros((H, W), dtype=np.float32)

    # For each pixel, check LoS using Bresenham-like line check
    # For efficiency, we use vectorized approach with sparse sampling
    step = max(1, min(H, W) // 64)

    for y in range(0, H, step):
        for x in range(0, W, step):
            if _check_los(building_map, tx_x, tx_y, float(x), float(y)):
                los_mask[max(0, y - bandwidth):min(H, y + bandwidth + 1),
                         max(0, x - bandwidth):min(W, x + bandwidth + 1)] = 1.0

    # Fill in skipped pixels by nearest neighbor
    if step > 1:
        from scipy.ndimage import zoom
        small_mask = los_mask[::step, ::step]
        los_mask = zoom(small_mask, step, order=0)[:H, :W]

    return los_mask


def _check_los(building_map, x1, y1, x2, y2):
    """Check if line from (x1,y1) to (x2,y2) is free of buildings."""
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    n_steps = max(int(dist * 2), 2)

    for i in range(n_steps + 1):
        t = i / n_steps
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        ix, iy = int(round(x)), int(round(y))

        if 0 <= iy < building_map.shape[0] and 0 <= ix < building_map.shape[1]:
            if building_map[iy, ix] > 0.5:  # Building pixel
                return False
    return True


def compute_los_mask_fast(building_map, tx_position, n_directions=360, max_radius=200):
    """
    Faster LoS mask using ray casting.

    Args:
        building_map: (H, W) numpy array
        tx_position: (2,) array [x, y]
        n_directions: number of rays to cast
        max_radius: maximum ray length

    Returns:
        los_mask: (H, W) numpy array
    """
    H, W = building_map.shape
    tx_x, tx_y = int(tx_position[0]), int(tx_position[1])

    los_mask = np.zeros((H, W), dtype=np.float32)

    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for r in range(1, max_radius):
            x = int(round(tx_x + r * cos_a))
            y = int(round(tx_y + r * sin_a))

            if x < 0 or x >= W or y < 0 or y >= H:
                break

            if building_map[y, x] > 0.5:  # Hit building
                break

            los_mask[y, x] = 1.0

    # Mark Tx position
    if 0 <= tx_y < H and 0 <= tx_x < W:
        los_mask[max(0, tx_y - 2):min(H, tx_y + 3), max(0, tx_x - 2):min(W, tx_x + 3)] = 1.0

    return los_mask
