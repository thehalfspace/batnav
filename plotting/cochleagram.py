# plotting/cochleagram.py

import matplotlib.pyplot as plt
import numpy as np


def plot_bmm(bmm: np.ndarray, fc: np.ndarray, fs: int, title: str = "Cochleagram"):
    """
    Plot basilar membrane motion (cochleagram).
    Args:
        bmm (np.ndarray): shape [T, F]
        fc (np.ndarray): center frequencies [Hz]
        fs (int): sampling frequency [Hz]
        title (str): optional title
    """
    time_ms = np.arange(bmm.shape[0]) / fs * 1e3
    freqs_kHz = fc / 1000

    plt.figure(figsize=(8, 5))
    plt.imshow(
        bmm.T,
        aspect='auto',
        origin='lower',
        extent=[time_ms[0], time_ms[-1], freqs_kHz[0], freqs_kHz[-1]],
        cmap='viridis',
        vmin=0,vmax=bmm.max()
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (kHz)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

