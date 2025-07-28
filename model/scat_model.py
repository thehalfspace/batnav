# model/scat_model.py

import numpy as np
from brian2 import Hz
from brian2hears import Sound, Gammatone
from typing import Dict


def run_biscat_main(config, ts: Dict) -> Dict:
    """
    Cochlear filterbank simulation using brian2hears Gammatone filters.

    Args:
        config: parsed config object
        ts (dict): {'data': np.ndarray, 'fs': int}

    Returns:
        dict with:
            'bmm': basilar membrane motion [time x channel]
            'Fc': center frequencies (Hz)
    """
    x = ts["data"]
    fs = ts["fs"]

    # Ensure mono input
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Wrap in Sound object
    sound = Sound(x, samplerate=fs * Hz)

    # Center frequencies (log-spaced or linear based on config)
    fmin = config.binaural.coch_fmin
    fmax = config.binaural.coch_fmax
    n_filters = config.binaural.coch_steps
    spacing_mode = config.binaural.coch_fcenter

    if spacing_mode == 1:  # linear
        Fc = np.linspace(fmin, fmax, n_filters)
    else:  # log spacing (default)
        Fc = np.logspace(np.log10(fmin), np.log10(fmax), n_filters)

    # Create gammatone filterbank
    gfb = Gammatone(sound, cf=Fc * Hz)  # cf = center_frequencies

    # Run processing
    bmm = gfb.process()
    bmm_np = np.asarray(bmm)  # shape: [time, channel]

    return {
        "coch": {
            "bmm": bmm_np,
            "Fc": Fc,
            "gfb": gfb,
        }
    }

