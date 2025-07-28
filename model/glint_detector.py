# model/glint_detector.py

import numpy as np
from typing import Optional


def findnotches2(d: np.ndarray, threshold_index: int) -> Optional[np.ndarray]:
    """
    Identify notch frequencies in the echo gap matrix `d`.

    Args:
        d (np.ndarray): 2D matrix of echo times, shape (freq_channels, thresholds)
        threshold_index (int): column to analyze (0-based)

    Returns:
        np.ndarray or None: notch indices (int), or None if no pattern found
    """
    if threshold_index >= d.shape[1]:
        return None

    W = d[:, threshold_index]
    non_nan_indices = np.where(~np.isnan(W))[0]

    if len(non_nan_indices) < 2:
        return None

    gaps = np.diff(non_nan_indices)
    gap_breaks = np.where(gaps > 1)[0]

    if len(gap_breaks) == 0:
        return None

    ip_s = non_nan_indices[gap_breaks]
    ip_e = non_nan_indices[gap_breaks + 1]
    ipL = np.round((ip_s + ip_e) / 2).astype(int)

    # Optional: remove phantom notches (very close together)
    if len(ipL) >= 2:
        diffs = np.diff(ipL)
        mode_spacing = np.median(diffs)
        ipL = ipL[diffs >= 0.5 * mode_spacing]

    return ipL if len(ipL) > 0 else None

