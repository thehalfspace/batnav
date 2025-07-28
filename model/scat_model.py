# model/scat_model.py

import numpy as np
from scipy.signal import firwin, lfilter
from typing import Dict


def run_biscat_main(config, ts: Dict) -> Dict:
    """
    Simulate cochlear filterbank output from SCAT pipeline.

    Args:
        config: loaded config object
        ts (dict): {'data': np.ndarray, 'fs': int}

    Returns:
        dict: { 'coch': { 'bmm': np.ndarray, 'Fc': np.ndarray } }
    """
    x = ts["data"]
    fs = ts["fs"]

    # If stereo, take mono
    if x.ndim == 2 and x.shape[1] == 2:
        x = np.mean(x, axis=1)

    # --- Filterbank config (override here or use config.binaural.*)
    fmin = 20000
    fmax = 100000
    n_filters = 81
    filter_order = 256

    # Center frequencies (log spaced for auditory realism)
    Fc = np.logspace(np.log10(fmin), np.log10(fmax), n_filters)

    # Allocate output: rows = time, cols = filters
    bmm = np.zeros((len(x), n_filters))

    for i, fc in enumerate(Fc):
        # Bandwidth heuristic
        bw = fc / 5
        low = max(10.0, fc - bw / 2)
        high = min(fs / 2 - 1, fc + bw / 2)

        # Normalize to Nyquist
        taps = firwin(filter_order + 1, [low, high], pass_zero=False, fs=fs)
        filtered = lfilter(taps, 1.0, x)

        bmm[:, i] = filtered

    return {
        "coch": {
            "bmm": bmm,
            "Fc": Fc
        }
    }

