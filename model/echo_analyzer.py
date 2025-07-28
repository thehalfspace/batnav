import numpy as np
from scipy.signal import firwin, lfilter
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class EchoAnalysisResult:
    sd: np.ndarray  # shape (NoT, num_freq)
    sgl: np.ndarray  # shape (NoT, num_freq)
    traces_echo: np.ndarray  # binary detection traces, shape (num_samples, num_freq)

def lowpass_filter(signal: np.ndarray, fs: float, cutoff: float = 10000.0, order: int = 30) -> np.ndarray:
    """
    Apply FIR low-pass filter to a multi-column signal (samples x freq bins)
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b = firwin(order + 1, norm_cutoff)
    return lfilter(b, 1.0, signal, axis=0)

def amp_lat_trading(trace: np.ndarray, threshold: float, twi: int) -> int:
    """
    Locate the most plausible echo position based on amplitude-latency trading.
    Returns the sample index within the window `twi`.
    """
    segment = trace[:twi]
    above_thresh = segment >= threshold
    if not np.any(above_thresh):
        return -1
    return np.argmax(above_thresh)

def linear_separate_window_10thresholds(wav_param) -> EchoAnalysisResult:
    sm_wf = wav_param.simStruct['sm_wf']
    NoT = wav_param.NoT
    Fs = wav_param.Fs
    twi = wav_param.callLenForMostFreq
    threshold_vector = wav_param.threshold_vector

    num_samples, num_freq = sm_wf.shape
    traces_echo = np.zeros_like(sm_wf)

    # Step 1: Low-pass filter
    sm_wf_filtered = lowpass_filter(sm_wf, Fs)

    # Step 2: Normalize and threshold
    sd = np.full((NoT, num_freq), -1, dtype=float)  # delay indices
    sgl = np.full((NoT, num_freq), -1, dtype=float)  # echo start indices

    for j in range(num_freq):
        trace = sm_wf_filtered[:, j]
        norm_trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace) + 1e-12)

        for k in range(NoT):
            threshold = threshold_vector[k]
            idx = amp_lat_trading(norm_trace, threshold, twi)
            if idx >= 0:
                sgl[k, j] = idx
                traces_echo[idx:, j] = 1

                # Find the next rising edge past `idx` with similar amplitude
                echo_window = norm_trace[idx:]
                echo_candidates = np.where(echo_window >= threshold)[0]
                if len(echo_candidates) > 0:
                    sd[k, j] = idx + echo_candidates[0]

    return EchoAnalysisResult(sd=sd, sgl=sgl, traces_echo=traces_echo)

