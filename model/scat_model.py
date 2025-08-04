# model/scat_model.py

import numpy as np
from brian2 import Hz
from brian2hears import Sound, Gammatone, Filterbank, erbspace
from typing import Dict
import matplotlib.pyplot as plt

def run_biscat_main(config, ts: Dict) -> Dict:
    """
    Run SCAT model using gammatone filterbank
    Args:
        config: parsed config object
        ts (dict): {'data': np.ndarray, 'fs': int}

    Returns:
        dict with:
            'bmm': basilar membrane motion [time x channel]
            'Fc': center frequencies (Hz)
    """
    x = ts['data']
    fs = ts['fs']

    # Ensure mono input
    if x.ndim == 2:
        x = x.mean(axis=1)
    
    fmin = config.binaural.coch_fmin
    fmax = config.binaural.coch_fmax
    n_channels = config.binaural.coch_steps
    spacing_mode = config.binaural.coch_fcenter

    # Run the gammatone filterbank
    # Create a Sound object from the input data.
    sound = Sound(x, samplerate=fs * Hz)

    # Center frequencies
    if config.binaural.coch_fcenter == 1:
        desired_bandwidth = config.binaural.coch_bw
        Fc = np.linspace(fmin, fmax, n_channels)
        erb_at_cf = 24.7 + 0.108 * Fc
        b_factor = desired_bandwidth / erb_at_cf
        # Create the Gammatone filterbank.
        gammatone_filterbank = Gammatone(sound, cf=Fc * Hz, b = b_factor)
    else:
        Fc = erbspace(fmin * Hz, fmax * Hz, n_channels)
        # Create the Gammatone filterbank.
        gammatone_filterbank = Gammatone(sound, cf=Fc)
    
   
    # Apply the filterbank to the sound data.
    cochlear_bmm = gammatone_filterbank.process()

    #plot_filtbank(cochlear_bmm.T)
    #breakpoint()
    
    return {
        "coch": {
            "bmm": np.asarray(cochlear_bmm),
            "Fc": Fc,
        }
    }


def plot_filtbank(bmm):
    """
    Plot the gammatone filterbank output
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(bmm, aspect='auto', origin='lower') #, 
    #           extent=[0, duration, 0, num_channels])
    plt.xlabel('Time (s)')
    plt.ylabel('Cochlear Channel')
    plt.title('Simulated Basilar Membrane Movement (Gammatone)')
    plt.colorbar(label='Amplitude')
    plt.show()

