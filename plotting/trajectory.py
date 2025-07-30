# plotting/trajectory.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from model.utils import polar_to_cartesian

def plot_static_trajectory(trajectory_data: dict, targets: list = None):
    """
    Plot static trajectory of the bat and overlay target positions and glint spacing.

    Parameters:
        trajectory_data (dict): Dict with keys: position, heading, ears, visited, glint_spacing.
        targets (list of Target): Optional list of Target objects with .r, .theta, .index
    """
    pos = np.array(trajectory_data["position"])
    headings = np.array(trajectory_data["heading"])
    ears = np.array(trajectory_data["ears"])
    visited = trajectory_data.get("visited", [])
    glint_spacing = trajectory_data.get("glint_spacing", [])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Bat Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # --- Plot path (thin dotted line)
    ax.plot(pos[:, 0], pos[:, 1], 'k:', lw=1, label="Bat Path")

    # --- Plot ears as lines
    for e in ears:
        ax.plot([e[0], e[2]], [e[1], e[3]], color='cyan', alpha=0.5)

    # --- Plot headings and labels
    for i in range(min(len(pos), len(headings))):
        p = pos[i]
        v = headings[i]

        ax.arrow(p[0], p[1], v[0]*0.2, v[1]*0.2,
                 head_width=0.05, head_length=0.05, fc='red', ec='red')

        if i < len(glint_spacing):
            ax.text(p[0]+0.05, p[1]+0.05, f"{glint_spacing[i]:.0f} µs", color='green', fontsize=7)

        if i < len(visited):
            ax.text(p[0], p[1]-0.05, f"{visited[i]}", color='blue', fontsize=7, ha='center')

    # --- Plot targets using polar_to_cartesian
    if targets:
        for t in targets:
            x, y = polar_to_cartesian(t.r, t.theta)
            ax.scatter(x, y, s=20, color='blue', marker='o', edgecolor='black')
            ax.text(x, y + 0.3, f"T{t.index}", ha='center', color='navy', fontsize=4)

    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_bat_trajectory(trajectory_data: dict):
    """
    Animate bat's movement through the scene.

    trajectory_data should contain:
        - position: List[np.ndarray] of bat positions (x, y)
        - heading: List[np.ndarray] of bat forward unit vectors
        - ears: List[np.ndarray] of [Lx, Ly, Rx, Ry]
        - visited: List[int] of target indices
        - glint_spacing: List[float] of glint spacing estimates
    """
    positions = np.array(trajectory_data["position"])
    headings = np.array(trajectory_data["heading"])
    ears = np.array(trajectory_data["ears"])
    visited = trajectory_data["visited"]
    glint_spacing = trajectory_data["glint_spacing"]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(np.min(positions[:,0]) - 1, np.max(positions[:,0]) + 1)
    ax.set_ylim(np.min(positions[:,1]) - 1, np.max(positions[:,1]) + 1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("🦇 Bat Trajectory Animation")

    bat_dot, = ax.plot([], [], 'ro', label='Bat position')
    ear_line, = ax.plot([], [], 'b-', lw=1, label='Ears')
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        bat_dot.set_data([], [])
        ear_line.set_data([], [])
        step_text.set_text("")
        return bat_dot, ear_line, step_text

    def update(i):
        pos = positions[i]
        vec = headings[i]
        ear = ears[i]  # [Lx, Ly, Rx, Ry]

        bat_dot.set_data([pos[0]], [pos[1]])
        ear_line.set_data([ear[0], ear[2]], [ear[1], ear[3]])

        # Remove previous arrows
        ax.patches.clear()
        ax.arrow(pos[0], pos[1], vec[0]*0.2, vec[1]*0.2, 
                 head_width=0.03, head_length=0.05, fc='k', ec='k')

        return bat_dot, ear_line, step_text

    ani = animation.FuncAnimation(fig, update, frames=len(positions),
                                  init_func=init, blit=False, interval=500, repeat=False)

    ax.legend()
    plt.show()

