import numpy as np
from model.signal_generator import generate_multiglints
from model.utils import polar_to_cartesian, euclidean_distance
from model.scat_model import run_biscat_main
from typing import Optional
from typing import List, Tuple
from scipy.signal import firwin, lfilter, hilbert, find_peaks
from model.wave_params import WaveParams



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



def design_lowpass_filter(fs: int, fp: int = 10_000, order: int = 30) -> np.ndarray:
    """Design a lowpass FIR filter using scipy."""
    return firwin(order + 1, cutoff=fp, fs=fs)


def apply_lowpass(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return lfilter(kernel, [1.0], signal)


def amp_latency_trading(waveform: np.ndarray, ref_amp: float, alt_coef: float, fs: int) -> float:
    """
    Compute amplitude-latency trading shift for a single channel.
    """
    envelope = np.abs(hilbert(waveform))
    idx_candidates = np.where(np.abs(envelope - ref_amp) < 0.01)[0]
    if len(idx_candidates) == 0:
        return 0.0

    start_idx = idx_candidates[0]
    segment = envelope[start_idx:]
    thresh = np.max(segment) * ref_amp

    peaks, props = find_peaks(segment, height=thresh)
    if len(peaks) < 2:
        return 0.0

    amp1 = props["peak_heights"][0]
    amp2 = props["peak_heights"][1]

    peak1_range = segment[peaks[0]:peaks[0] + 500]
    peak2_range = segment[max(peaks[1] - 500, 0):min(peaks[1] + 500, len(segment))]

    amp1 = max(amp1, np.max(peak1_range))
    amp2 = max(amp2, np.max(peak2_range))

    if amp2 == 0:
        return 0.0

    delta_db = 20 * np.log10(amp1 / amp2)
    time_us = delta_db * alt_coef
    return time_us * 1e-6 * fs


def linear_separate_window_10thresholds(wave_params: WaveParams) -> Tuple[List[np.ndarray], np.ndarray]:
    sim = wave_params.simStruct
    Fs = wave_params.Fs
    NT = wave_params.NT
    NoT = wave_params.NoT
    ALT_coef = wave_params.ALT
    sep_samples = wave_params.SepbwBRand1stEchoinSmpls
    twi = wave_params.callLenForMostFreq
    twh = wave_params.callLenForHighFreq
    twu = wave_params.callLenSpecial
    th_type = wave_params.th_type
    th_val = wave_params.startingThPercent

    bmm = sim["coch"]["bmm"]
    Fc = sim["coch"]["Fc"]
    n_channels = bmm.shape[1]
    echo_trace = np.zeros(n_channels)
    all_echo_diffs = []

    lp_kernel = design_lowpass_filter(Fs)

    for ch in range(n_channels):
        signal = bmm[:, ch].copy()
        signal[signal < 0] = 0
        smoothed = apply_lowpass(signal, lp_kernel)

        max_val = np.max(smoothed[:sep_samples])
        #print(f"Channel {ch}: max_val at index {np.argmax(smoothed[:sep_samples])}")

        min_val = np.min(smoothed)
        if max_val - min_val == 0:
            all_echo_diffs.append(np.array([]))
            echo_trace[ch] = np.nan
            continue
        smoothed = (smoothed - min_val) * 100 / (max_val - min_val)
        smoothed[smoothed < 0] = 0

        th_start = th_val if th_type == "const" else 10 / (ch + th_val) * np.max(smoothed)
        thresholds = np.linspace(th_start, 98, NT)
        threshold = thresholds[NoT - 1]
        
        # In the process of matching matlab first_gap_L, 
        # The first one might be too wide for window size
        # So we use the bottom one
        #WL = int((twh if ch > 60 else twi) * Fs)
        WL = int(twi * Fs * 0.5)  
        
        pulse_indices = []
        i = 50
        while i < sep_samples:
            if smoothed[i] >= threshold:
                pulse_indices.append(i)
                i += WL
            else:
                i += 1

        echo_indices = []
        i = sep_samples
        while i < len(smoothed):
            if smoothed[i] >= threshold:
                echo_indices.append(i)
                i += WL
            else:
                i += 1

        echo_indices = [i for i in echo_indices if abs(i - sep_samples) > 50]

        shift_samples = amp_latency_trading(bmm[:, ch], ref_amp=0.1, alt_coef=ALT_coef, fs=Fs)
        shift_samples = max(0, int(abs(shift_samples)))
        echo_indices = [i + shift_samples for i in echo_indices]

        if echo_indices and pulse_indices:
            diffs = np.array(echo_indices) - pulse_indices[0]
            all_echo_diffs.append(diffs)
            echo_trace[ch] = echo_indices[0]
        else:
            all_echo_diffs.append(np.array([]))
            echo_trace[ch] = np.nan

    return all_echo_diffs, echo_trace

