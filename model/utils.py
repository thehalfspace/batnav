# model/utils.py

import numpy as np
from typing import List, Union
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


def estimate_itd_from_histograms(gap_L: List[int], gap_R: List[int], bins: Union[int, str] = 'auto') -> float:
    """
    Estimate interaural time delay (ITD) from histogram peaks in sample space.
    Returns:
        float: absolute delay in samples
    """
    if len(gap_L) == 0 or len(gap_R) == 0:
        print('Gaps in L and R ears are too small, bins can not be estimated')
        return float('nan')

    # Filter out NaN and infinite values
    gap_L_clean = np.array(gap_L)
    gap_R_clean = np.array(gap_R)
    gap_L_clean = gap_L_clean[np.isfinite(gap_L_clean)]
    gap_R_clean = gap_R_clean[np.isfinite(gap_R_clean)]
    
    if len(gap_L_clean) == 0 or len(gap_R_clean) == 0:
        print('No valid (finite) gap values found after filtering NaN/inf')
        return float('nan')

    print("GapL: ", len(gap_L_clean))
    print("GapR: ", len(gap_R_clean))

    hL, bin_edges_L = np.histogram(gap_L_clean, bins=bins)
    hR, bin_edges_R = np.histogram(gap_R_clean, bins=bins)
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

