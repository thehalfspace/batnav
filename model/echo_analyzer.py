# model/echo_analyzer.py

import numpy as np
from model.signal_generator import generate_multiglints
from model.glint_detector import findnotches2  # placeholder location
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
    """
    # 1. Distance from bat to target
    target_pos = polar_to_cartesian(target.r, target.theta)
    dist_to_target = euclidean_distance(bat.position, target_pos)
    print(f"🦇 Bat–Target distance: {dist_to_target:.3f} m")

    # 2. Generate signal with multiple glints
    ts = generate_multiglints(dist_to_target, target.tin)
    print(f"📡 Generated multiglints: data shape = {ts['data'].shape}, fs = {ts['fs']}")

    # 3. Run SCAT model
    sim_struct = run_biscat_main(config, ts)
    bmm = sim_struct['coch']['bmm']
    Fc = sim_struct['coch']['Fc']
    print(f"🧠 SCAT output: bmm shape = {bmm.shape}, Fc range = {Fc[0]:.1f}-{Fc[-1]:.1f} Hz")

    # 4. Run echo analyzer for each threshold
    n_freqs = len(Fc)
    gap_matrix = np.full((n_freqs, num_thresholds), np.nan)

    for t in range(num_thresholds):
        wave_params.simStruct = sim_struct
        wave_params.NoT = t + 1
        _, first_gap = linear_separate_window_10thresholds(wave_params)
        gap_matrix[:, t] = first_gap
        n_valid = np.sum(~np.isnan(first_gap))
        print(f"🔍 Threshold {t+1}: valid gaps = {n_valid}")

    # 5. Check if any threshold has usable data
    valid_cols = np.where(~np.isnan(gap_matrix).all(axis=0))[0]
    print(f"📊 Valid columns in gap matrix: {valid_cols}")
    if len(valid_cols) == 0:
        print("❌ All thresholds are NaN — returning None")
        return None

    # 6. Find notches from best threshold column
    selected_col = valid_cols[0]
    notches = findnotches2(gap_matrix, selected_col)
    print(f"🎯 Notches from threshold {selected_col+1}: {notches}")
    
    if notches is None or len(notches) < 2:
        print("❌ Not enough notches found — returning None")
        return None


    # 7. Compute glint spacing from notch frequency intervals
    notch_freqs = Fc[notches]
    deltas = np.diff(np.sort(notch_freqs))
    print(f"📐 Notch frequencies (Hz): {notch_freqs}")
    print(f"🔬 Frequency deltas: {deltas}")

    if len(deltas) == 0:
        print("❌ No frequency deltas — returning None")
        return None

    # 8. Estimate dominant spacing via histogram
    hist, edges = np.histogram(deltas, bins=30)
    peak_bin = np.argmax(hist)
    spacing_Hz = (edges[peak_bin] + edges[peak_bin + 1]) / 2
    glint_spacing_us = 1 / spacing_Hz * 1e6

    print(f"✅ Estimated glint spacing: {glint_spacing_us:.2f} µs")
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
    echo_trace = np.full(n_channels, np.nan)
    all_echo_diffs = []

    lp_kernel = design_lowpass_filter(Fs)

    # For the first threshold, overwrite the input interval by setting
    # higher/wider length of threhsold
    if NoT == 1:
        twi = 1e-3
        twh = 1e-3

    for ch in range(n_channels):
        signal = bmm[:, ch].copy()
        signal[signal < 0] = 0
        smoothed = apply_lowpass(signal, lp_kernel)

        max_val = np.max(smoothed[:sep_samples])
        min_val = np.min(smoothed)
        if max_val - min_val == 0:
            all_echo_diffs.append(np.array([]))
            continue

        norm = (smoothed - min_val) * 100 / (max_val - min_val)
        norm[norm < 0] = 0

        th_start = th_val if th_type == "const" else 10 / (ch + th_val) * np.max(norm)
        thresholds = np.linspace(th_start, 98, NT)
        threshold = thresholds[NoT - 1]
        
        pulse_indices = []
        echo_indices = []
        
        # In the process of matching matlab first_gap_L, 
        # The first one might be too wide for window size
        # So we use the top one
        #WL = int(twi * Fs * 0.5)

        if ch > 20:
            WL = int((twh if ch > 60 else twi) * Fs)
			
            # Find pulse
            i = 49
            while i < sep_samples:
                if smoothed[i] >= threshold:
                    pulse_indices.append(i)
                    i += WL
                else:
                    i += 1
            
            # Find echo
            gg = np.where(smoothed[sep_samples:] >= threshold)[0]
            if len(gg) > 0:
                a = sep_samples + gg[0]
                echo_indices.append(a)
                while True:
                    candidates = np.where(smoothed[a + WL : ] >= threshold)[0]
                    if len(candidates) == 0:
                        break
                    a = a + WL + candidates[0]
                    echo_indices.append(a)
        else:
            WL2 = int(twu * Fs)
            kk = np.where(smoothed[49:] >= threshold)[0]
            if len(kk) > 0:
                a = 49 + kk[0]
                pulse_indices.append(a)
                while True:
                    candidates = np.where(smoothed[a + WL2 :] >= threshold)[0]
                    if len(candidates) == 0:
                        break
                    a = a + WL2 + candidates[0]
                    echo_indices.append(a)
        
        # Remove echo close to boundary
        echo_indices = [i for i in echo_indices if abs(i - sep_samples) > 50]
        pulse_indices = [i for i in pulse_indices if i >= 50]

        # Apply amplitude-latency trading shift
        shift_samples = amp_latency_trading(bmm[:, ch], ref_amp=0.1, alt_coef=ALT_coef, fs=Fs)
        shift_samples = int(abs(shift_samples)) if shift_samples < 0 else 0
        echo_indices = [i + shift_samples for i in echo_indices]
        
        # Final gap calculation
        if echo_indices and pulse_indices:
            diffs = np.array(echo_indices) - pulse_indices[0]
            print(f"→ Gaps: {diffs}")
            all_echo_diffs.append(diffs)
            echo_trace[ch] = echo_indices[0]

            if 23 < ch < 28 and len(pulse_indices) == 2 and len(echo_indices) == 1:
                # Estimate surrounding echo traces 
                val_l = np.nanmean(echo_trace[:5])
                val_h = np.nanmean(echo_trace[18:23])
                if abs(echo_indices[0] - val_l) > abs(echo_indices[0] - val_h):
                    diff = echo_indices[0] - pulse_indices[0]
                else:
                    diff = echo_indices[0] - pulse_indices[1]
                diffs = np.array([diff])
            else:
                diffs = np.array(echo_indices) - pulse_indices[0]

        else:
            all_echo_diffs.append(np.array([]))
            echo_trace[ch] = np.nan

        breakpoint()

    return all_echo_diffs, echo_trace

