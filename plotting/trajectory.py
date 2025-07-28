# plotting/trajectory.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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

