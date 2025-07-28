import numpy as np
from model.signal_generator import generate_multiglints
from model.utils import polar_to_cartesian, euclidean_distance
from model.scat_model import run_biscat_main  # stub assumed
from typing import Optional


def estimate_glint_spacing(
    bat,
    target,
    config,
    wave_params,
    num_thresholds: int = 10
) -> Optional[float]:
    """
    Estimate glint spacing in microseconds by analyzing spectral notches.
    
    Args:
        bat (Bat): the current bat object
        target (Target): target object
        config: SCAT config object
        wave_params (WaveParams): wave parameter dataclass
        num_thresholds (int): number of threshold levels (default 10)
    
    Returns:
        float or None: estimated spacing in microseconds
    """
    # 1. Distance from bat to target
    target_pos = polar_to_cartesian(target.r, target.theta)
    dist_to_target = euclidean_distance(bat.position, target_pos)

    # 2. Generate signal with multiple glints
    ts = generate_multiglints(dist_to_target, target.tin)  # tin in µs

    # 3. Run SCAT model
    sim_struct = run_biscat_main(config, ts)

    # 4. Run echo analyzer for each threshold
    Fc = sim_struct['coch']['Fc']
    n_freqs = len(Fc)
    gap_matrix = np.zeros((n_freqs, num_thresholds))

    for t in range(num_thresholds):
        wave_params.simStruct = sim_struct
        wave_params.NoT = t + 1
        _, first_gap = linear_separate_window_10thresholds(wave_params)
        gap_matrix[:, t] = first_gap

    # 5. Use first column with valid data (or default to column 0)
    valid_cols = np.where(~np.isnan(gap_matrix).all(axis=0))[0]
    if len(valid_cols) == 0:
        return None

    from model.glint_detector import findnotches2  # placeholder location
    notches = findnotches2(gap_matrix, valid_cols[0])  # returns indices of notches

    if notches is None or len(notches) < 2:
        return None

    # 6. Compute frequency spacing
    notch_freqs = Fc[notches]  # in Hz
    deltas = np.diff(np.sort(notch_freqs))  # in Hz
    if len(deltas) == 0:
        return None

    # 7. Histogram to find dominant spacing
    hist, edges = np.histogram(deltas, bins=30)
    peak_bin = np.argmax(hist)
    spacing_Hz = (edges[peak_bin] + edges[peak_bin + 1]) / 2

    # 8. Glint spacing = 1 / frequency spacing (Hz → µs)
    glint_spacing_us = 1 / spacing_Hz * 1e6
    return glint_spacing_us

