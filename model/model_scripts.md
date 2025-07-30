

## `./__init__.py`

```python
# model/__init__.py
# Makes the model directory a package

```


## `./bat.py`

```python
# model/bat.py

from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple


@dataclass
class Bat:
    # Head position (x, y)
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # Heading unit vector (pointing forward)
    heading: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    # Ear separation (meters)
    ear_distance: float = 0.014  # default 14 mm

    # Flyspeed (m/s)
    speed: float = 0.4

    # Head rotation angle (degrees, 0 = facing up)
    angle_deg: float = 0.0

    # History logs
    path_history: List[np.ndarray] = field(default_factory=list)
    angle_history: List[float] = field(default_factory=list)
    speed_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.log_state()

    def get_ear_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (left, right) ear coordinates based on current heading and position."""
        perp = np.array([-self.heading[1], self.heading[0]])  # perpendicular to heading
        offset = (self.ear_distance / 2) * perp
        left_ear = self.position - offset
        right_ear = self.position + offset
        return left_ear, right_ear

    def rotate_head(self, delta_angle: float):
        """Rotate head left (+) or right (-) by delta_angle (degrees)."""
        self.angle_deg += delta_angle
        theta_rad = np.radians(self.angle_deg)
        self.heading = np.array([np.sin(theta_rad), np.cos(theta_rad)])
        self.log_state()

    def move_forward(self):
        """Move in the direction of current heading by flyspeed."""
        self.position += self.heading * self.speed
        self.log_state()

    def log_state(self):
        """Store current state to history."""
        self.path_history.append(self.position.copy())
        self.angle_history.append(self.angle_deg)
        self.speed_history.append(self.speed)

    def reset(self):
        """Reset bat to origin."""
        self.position = np.array([0.0, 0.0])
        self.heading = np.array([0.0, 1.0])
        self.angle_deg = 0.0
        self.path_history.clear()
        self.angle_history.clear()
        self.speed_history.clear()
        self.log_state()

```


## `./echo_analyzer.py`

```python
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

```


## `./filterbank.py`

```python
# model/filterbank.py

def run_filterbank(ts, config, method: str = "gammatone") -> Dict:
    if method == "gammatone":
        return run_gammatone(ts, config)
    elif method == "scipy":
        return run_firbank(ts, config)
    else:
        raise ValueError(f"Unknown filterbank method: {method}")

```


## `./glint_detector.py`

```python
# model/glint_detector.py

import numpy as np
from typing import Optional
from scipy.stats import mode


def findnotches2(d: np.ndarray, threshold_index: int) -> Optional[np.ndarray]:
    """
    Identify notch frequencies in the echo gap matrix `d`.

    Args:
        d (np.ndarray): 2D matrix of echo times, shape (freq_channels, thresholds)
        threshold_index (int): column to analyze (0-based)

    Returns:
        np.ndarray or None: notch indices (int), or None if no pattern found
    """
    if threshold_index >= d.shape[1]:
        return None

    W = d[:, threshold_index]
    non_nan_indices = np.where(~np.isnan(W))[0]

    if len(non_nan_indices) < 2:
        return None

    gaps = np.diff(non_nan_indices)
    gap_breaks = np.where(gaps > 1)[0]

    if len(gap_breaks) == 0:
        return None

    ip_s = non_nan_indices[gap_breaks]
    ip_e = non_nan_indices[gap_breaks + 1]
    ipL = np.round((ip_s + ip_e) / 2).astype(int)

    # Optional: remove phantom notches (very close together)
    if len(ipL) >= 2:
        diffs = np.diff(ipL)
        mode_spacing = mode(diffs, keepdims=False).mode
        ipL = ipL[diffs >= 0.5 * mode_spacing]

    return ipL if len(ipL) > 0 else None

```


## `./scat_model.py`

```python
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

```


## `./signal_generator.py`

```python
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

```


## `./target.py`

```python
# model/target.py

from pydantic import BaseModel

class Target(BaseModel):
    index: int
    r: float           # radial distance from origin (m)
    theta: float       # angle in degrees (relative to vertical)
    tin: float         # glint spacing in µs
    NoG: int           # number of glints

```


## `./utils.py`

```python
# model/utils.py

import numpy as np
from typing import List
from model.target import Target
from model.bat import Bat


def polar_to_cartesian(r: float, theta_deg: float) -> np.ndarray:
    """
    Convert polar coordinates to 2D Cartesian coordinates.
    Args:
        r (float): radial distance in meters
        theta_deg (float): angle in degrees (0 = up, increasing counterclockwise)
    Returns:
        np.ndarray: Cartesian coordinates [x, y]
    """
    theta_rad = np.radians(theta_deg + 90)  # +90 so 0° points upward
    return np.array([r * np.cos(theta_rad), r * np.sin(theta_rad)])


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 2D points.
    """
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def batch_euclidean_distances(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Vectorized distance from each row in `points` to `ref`.
    """
    return np.linalg.norm(points - ref, axis=1)


def angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """
    Signed angle (degrees) from v → u.
    Positive = u is to the left of v, Negative = right.
    """
    u3 = np.append(u, 0)
    v3 = np.append(v, 0)
    angle = np.arctan2(np.linalg.norm(np.cross(v3, u3)), np.dot(v3, u3))
    angle_deg = np.degrees(angle)
    return angle_deg if np.cross(v3, u3)[2] >= 0 else -angle_deg


def compute_delays_to_ears(bat: Bat, target_pos: np.ndarray) -> List[float]:
    """
    Compute physical delay (distance in meters) from target to each ear.
    Returns:
        List[float]: [delay_to_left, delay_to_right]
    """
    left_ear, right_ear = bat.get_ear_positions()
    return [
        euclidean_distance(target_pos, left_ear),
        euclidean_distance(target_pos, right_ear)
    ]


def apply_amplitude_latency_trading(bat: Bat, target_pos: np.ndarray, delays: List[float]) -> List[float]:
    """
    Adjust delay values by amplitude-latency trading effect.
    Returns:
        List[float]: [adjusted_delay_left, adjusted_delay_right] (in meters)
    """
    aim_vector = target_pos - bat.position
    offset_angle = angle_between_vectors(aim_vector, bat.heading)  # degrees

    # Empirical formula from Chen Ming's MATLAB code
    trading_offset = 17 / 60 * offset_angle * 11e-6 * 340  # meters
    return [delays[0] - trading_offset, delays[1] + trading_offset]


def estimate_itd_from_histograms(gap_L: List[int], gap_R: List[int], bins: int = 30) -> float:
    """
    Estimate interaural time delay (ITD) from histogram peaks in sample space.
    Returns:
        float: absolute delay in samples
    """
    hL, bin_edges_L = np.histogram(gap_L, bins=bins)
    hR, bin_edges_R = np.histogram(gap_R, bins=bins)
    smpl_L = bin_edges_L[np.argmax(hL)]
    smpl_R = bin_edges_R[np.argmax(hR)]
    return abs(smpl_R - smpl_L)


def choose_speed_from_delay(delay_right: float, sample_rate: float) -> float:
    """
    Heuristic for adjusting bat flyspeed based on delay to right ear.
    Args:
        delay_right (float): in meters
        sample_rate (float): samples per second
    Returns:
        float: flyspeed in m/s
    """
    delay_samples = delay_right / 340 * sample_rate * 2
    if delay_samples < round(1.5 * 2 / 340 * sample_rate):
        return 0.2
    elif delay_samples > round(3.0 * 2 / 340 * sample_rate):
        return 0.5
    else:
        return 0.4

def choose_rotation_step(itd_samples: float) -> int:
    """
    Choose bat head rotation step size based on interaural time difference (ITD).
    
    Args:
        itd_samples (float): estimated ITD in samples
    
    Returns:
        int: rotation step in degrees
    """
    if itd_samples > 150:
        return 20  # target is behind the bat
    elif itd_samples < 80:
        return 10  # target is in front
    else:
        return 15  # somewhere intermediate


def find_nearest_target(targets: List[Target], bat_position: np.ndarray, available_indices: List[int]) -> int:
    """
    Find index (from available_indices) of target nearest to bat.
    """
    coords = np.array([
        polar_to_cartesian(targets[i].r, targets[i].theta)
        for i in available_indices
    ])
    dists = batch_euclidean_distances(coords, bat_position)
    return available_indices[np.argmin(dists)]

```


## `./wave_params.py`

```python
# model/wave_params.py

from dataclasses import dataclass
from typing import Literal

@dataclass
class WaveParams:
    Fs: int = 500_000                          # sampling frequency
    callLenForMostFreq: float = 0.5e-3         # .5 ms
    callLenForHighFreq: float = 1e-3           # 1 ms
    callLenSpecial: float = 1.8e-3             # 1.8 ms
    sepFlag: int = 1                           # 1 = fixed separation
    whenBrStart: float = 0.5e-3                # 0.5 ms
    startingThPercent: int = 3                 # threshold start percent
    th_type: Literal["const", "dynamic"] = "const"
    SepbwBRand1stEchoinSmpls: int = 5000       # ~10ms at 500kHz
    color: str = "b"
    ALT: int = -25                             # amplitude latency trading
    NT: int = 10                               # number of thresholds
    NoT: int = 1                                # current threshold (to be iterated)
    simStruct: dict = None                      # to be set per signal

```
