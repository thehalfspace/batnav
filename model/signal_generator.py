# model/signal_generator.py

from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.io import loadmat
import math

# Constants (can later be loaded from config)
SOUND_SPEED = 340.0       # meters per second
FS = 500_000              # sampling frequency (Hz)
CHIRP_PATH = Path("config/1H_100k_20k_fs_500k_3ms_WelchWin.mat")


def load_chirp() -> np.ndarray:
    """
    Load the chirp waveform used for broadcast and echo.
    Returns:
        waveform (np.ndarray): 1D chirp waveform scaled to match MATLAB
    """
    Y = loadmat(CHIRP_PATH)['Y']
    return Y.flatten() * 2


def apply_echo_delay(delay_m: float, fs: int = FS) -> int:
    """
    Convert a delay in meters to a sample offset (round-trip).
    Args:
        delay_m (float): one-way distance in meters
        fs (int): sampling frequency in Hz
    Returns:
        int: number of samples of delay
    """
    round_trip_time = delay_m / SOUND_SPEED
    return math.floor(round_trip_time * fs)
    #return int(round(round_trip_time * fs))


def generate_sigs_with_delay(delay_m_list: List[float]) -> dict:
    """
    Simulate a binaural signal with different delays to each ear.
    Args:
        delay_m_list (List[float]): list of two distances (left, right) in meters
    Returns:
        dict with keys: 'data' (np.ndarray), 'fs' (int), 'delay' (List[float]), 'time' (np.ndarray)
    """
    assert len(delay_m_list) == 2, "Expecting 2 delays: left and right ear"

    # chirp is equivalent of brc in matlab
    chirp = load_chirp()
    echo = chirp / 2

    # NOTE: const value here 
    noise = np.zeros(1500)

    # Double delay_m_list as done in matlab
    delay_m_list = [2*d for d in delay_m_list]
    delays_samples = [apply_echo_delay(d) for d in delay_m_list]

    # NOTE: 11 here is the space after echo
    max_delay_ms = round(max(delay_m_list) / SOUND_SPEED * 1000 + 11)
    total_len = int(max_delay_ms * 1e-3 * FS)
    time = np.linspace(1 / FS, total_len / FS, total_len)

    data = np.zeros((total_len, 2))

    for i, delay_samples in enumerate(delays_samples):
        sig = np.concatenate([
            noise,
            chirp,
            np.zeros(max(0, delay_samples - len(chirp))),
            echo
        ])
        if len(sig) < total_len:
            sig = np.pad(sig, (0, total_len - len(sig)))
        data[:, i] = sig

    return {
        "data": data,
        "fs": FS,
        "delay": [d / 2 for d in delay_m_list],
        "time": time
    }


def generate_multiglints(delay_m: float, glint_spacing_us: float) -> dict:
    """
    Simulate a monaural echo signal with two glints spaced in time.
    Args:
        delay_m (float): one-way distance to target (in meters)
        glint_spacing_us (float): spacing between glints (in microseconds)
    Returns:
        dict: { 'data': np.ndarray, 'fs': int, 'delay': float, 'time': np.ndarray }
    """
    chirp = load_chirp()
    noise = np.zeros(1500)

    # Convert inputs
    delay_m = delay_m * 2  # round-trip
    Ng = int(glint_spacing_us * 1e-6 * FS)
    delay_samples = apply_echo_delay(delay_m)

    # Create glint echo: echo1 + echo2 (shifted and combined)
    echo = chirp / 3 + np.concatenate([np.zeros(Ng), chirp[:-Ng] / 3])
    echo = echo / np.max(np.abs(echo)) * np.max(np.abs(chirp))  # normalize to chirp

    # Final signal: noise + chirp + delay_padding + glint_echo + tail
    sig = np.concatenate([
        noise,
        chirp,
        np.zeros(max(0, delay_samples - len(chirp))),
        echo,
        chirp[-Ng:] / 3  # trailing glint echo tail
    ])

    # Total length: match MATLAB
    max_delay_ms = round(delay_m / SOUND_SPEED * 1e3 + 11)
    total_len = int(max_delay_ms / 1e3 * FS)
    if len(sig) < total_len:
        sig = np.pad(sig, (0, total_len - len(sig)))

    time = np.linspace(1 / FS, total_len / FS, total_len)

    return {
        "data": sig,
        "fs": FS,
        "delay": delay_m / 2,  # report one-way
        "time": time
    }

