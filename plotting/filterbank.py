# plotting/filterbank.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import sosfreqz


def plot_gammatone_filterbank(gfb, fs: int, title: str = "Gammatone Filterbank (Brian2Hears)"):
    """
    Plot frequency response of Brian2Hears Gammatone filters using filt_b and filt_a.

    Args:
        gfb: Gammatone object from brian2hears
        fs (int): sampling rate in Hz
        title (str): plot title
    """
    cf_kHz = np.array(gfb.cf) / 1e3  # center freqs in kHz
    n_filters = len(cf_kHz)
    colors = plt.get_cmap("jet")(np.linspace(0, 1, n_filters))

    plt.figure(figsize=(10, 6))

    for i in range(n_filters):
        w, h = freqz(gfb.filt_b[i], gfb.filt_a[i], worN=2048, fs=fs)
        plt.plot(w / 1000, 20 * np.log10(np.abs(h) + 1e-12), color=colors[i])

    plt.title(title)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from brian2hears import Sound

def plot_brian2hears_impulse_response(gfb, impulse, fs: int, duration_ms: float = 30.0, title="Gammatone Filterbank Response"):
    """
    Plot magnitude response of Brian2Hears filters by passing an impulse.

    Args:
        gfb: Gammatone filterbank object (Brian2Hears)
        fs (int): sampling rate in Hz
        duration_ms (float): impulse response duration in milliseconds
        title (str): plot title
    """
    duration_samples = int(duration_ms / 1000 * fs)
    sound = Sound(impulse, samplerate=fs)

    # Apply filterbank to the impulse
    gfb_impulse = gfb.__class__(sound, cf=gfb.cf)  # re-init with impulse
    response = gfb_impulse.process()
    response = np.asarray(response)

    # FFT and frequency axis
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    mag_db = 20 * np.log10(np.abs(np.fft.rfft(response, n=n_fft, axis=0)) + 1e-12)

    # Plot
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("jet")
    colors = cmap(np.linspace(0, 1, response.shape[1]))

    for i in range(response.shape[1]):
        plt.plot(freqs / 1000, mag_db[:, i], color=colors[i])

    plt.title(title)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

