# model/utils.py

import numpy as np
from typing import List
from model.target import Target


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return the Euclidean distance between two 2D points.
    """
    return np.linalg.norm(np.array(a) - np.array(b))


def distances_to_targets(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of distances from multiple 2D points to a reference point.
    
    Args:
        points (np.ndarray): shape (N, 2)
        reference (np.ndarray): shape (2,)
    
    Returns:
        np.ndarray: distances of shape (N,)
    """
    diff = points - reference
    return np.linalg.norm(diff, axis=1)


def angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute signed angle (in degrees) from vector v to vector u.
    Positive = left turn, Negative = right turn.

    Args:
        u (np.ndarray): target vector (e.g., target direction)
        v (np.ndarray): reference vector (e.g., bat heading)

    Returns:
        float: signed angle in degrees
    """
    u = np.array([*u, 0])
    v = np.array([*v, 0])

    angle_rad = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    angle_deg = np.degrees(angle_rad)

    # Determine sign: positive = left, negative = right
    rot_z = np.cross(v, u)[2]
    return angle_deg if rot_z >= 0 else -angle_deg


def find_nearest_target(
    targets: List[Target],
    bat_position: np.ndarray,
    available_indices: List[int]
) -> int:
    """
    Return the index (within available_indices) of the nearest target.

    Args:
        targets (List[Target]): list of all target objects
        bat_position (np.ndarray): bat's head (x, y)
        available_indices (List[int]): indices in 'targets' to consider

    Returns:
        int: index in available_indices corresponding to nearest target
    """
    coords = np.array([
        [
            targets[i].r * np.cos(np.radians(targets[i].theta + 90)),
            targets[i].r * np.sin(np.radians(targets[i].theta + 90))
        ]
        for i in available_indices
    ])
    distances = distances_to_targets(coords, bat_position)
    nearest_local_index = np.argmin(distances)
    return available_indices[nearest_local_index]

