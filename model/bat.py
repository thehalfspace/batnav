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
    ear_history: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    heading_history: List[np.ndarray] = field(default_factory=list)

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
        self.ear_history.append(self.get_ear_positions())
        self.heading_history.append(self.heading.copy())

    def reset(self):
        """Reset bat to origin."""
        self.position = np.array([0.0, 0.0])
        self.heading = np.array([0.0, 1.0])
        self.angle_deg = 0.0
        self.path_history.clear()
        self.angle_history.clear()
        self.speed_history.clear()
        self.ear_history.clear()
        self.heading_history.clear()
        self.log_state()

