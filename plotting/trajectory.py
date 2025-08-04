# plotting/trajectory.py

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from model.utils import polar_to_cartesian
import os

def plot_bat_steps(trajectory_data: dict, output_dir: str = "data"):
    """
    Plot the number of steps it took to rotate bat head and move
    """
    angles = np.array(trajectory_data["angles"])
    step_log = trajectory_data["step_log"]

    indices = [entry["index"] for entry in step_log]
    iterations = [entry["iteration"] for entry in step_log]

    # --- Plot Bat Rotation Angle over Time ---
    fig1, ax1 = plt.subplots()
    ax1.plot(indices, angles[indices], marker='o', linestyle='-', color='teal')
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Rotation Angle (deg)")
    ax1.set_title("Bat Rotation Angle History")
    ax1.grid(True)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "rotation_angle_history.png"), dpi=300)
    plt.close(fig1)

    # --- Plot Steps Taken per Target Attempt ---
    df = pl.DataFrame(step_log)
    step_counts = df.group_by("iteration").len()
    total_steps = len(angles)

    fig2, ax2 = plt.subplots()
    ax2.bar(step_counts["iteration"], step_counts["len"], color='teal')
    ax2.set_xlabel("Target index")
    ax2.set_ylabel("Number of Steps to Align")
    ax2.set_title(f"Steps per Target Attempt. Total Steps: {total_steps}")
    ax2.grid(axis='y')
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "steps_per_target.png"), dpi=300)
    plt.close(fig2)

    print(f"Trajectory plots saved in {output_dir}")



def plot_static_trajectory(trajectory_data: dict, targets: list = None, output_dir: str = 'data'):
    """
    Plot static trajectory of the bat and overlay target positions.

    Parameters:
        trajectory_data (dict): Dict with keys: position, heading, ears, visited, glint_spacing.
        targets (list of Target): Optional list of Target objects with .r, .theta, .index
    """
    pos = np.array(trajectory_data["position"])
    headings = np.array(trajectory_data["heading"])
    visited = trajectory_data.get("visited", [])
    milestones = trajectory_data.get("milestones", [])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Bat Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # --- Plot path (dotted line with larger points)
    ax.plot(pos[:, 0], pos[:, 1], 'k:', lw=1, label="Bat Path")
    ax.scatter(pos[:, 0], pos[:, 1], s=15, color='black', alpha=0.7)

    # --- Plot heading arrows
    for i in range(min(len(pos), len(headings))):
        p = pos[i]
        v = headings[i]
        ax.arrow(p[0], p[1], v[0]*0.2, v[1]*0.2,
                 head_width=0.05, head_length=0.05, fc='red', ec='red')

        # If this step hit a target (i.e., head aligned), draw dotted line to that target
        #if i < len(visited) and targets:
        #    target = next((t for t in targets if t.index == visited[i]), None)
        #    if target:
        #        tx, ty = polar_to_cartesian(target.r, target.theta)
        #        ax.plot([p[0], tx], [p[1], ty], 'g--', lw=0.8, alpha=0.7)

        # When bat estimates glint spacing
        for m in milestones:
            idx = m["index"]
            target_idx = m["target_idx"]
            success = m["success"]
            
            if idx < len(pos):
                p = pos[idx]
                target = next((t for t in targets if t.index == target_idx), None)
                if target:
                    tx, ty = polar_to_cartesian(target.r, target.theta)
                    color = 'limegreen' if success else 'orange'
                    ax.plot([p[0], tx], [p[1], ty], '--', lw=1.0, alpha=0.8, color=color)


    # --- Plot targets as hollow circles with labels
    if targets:
        for t in targets:
            x, y = polar_to_cartesian(t.r, t.theta)
            ax.scatter(x, y, s=60, facecolors='none', edgecolors='navy', linewidths=1.2)
            ax.text(x, y + 0.3, f"T{t.index + 1}", ha='center', color='navy', fontsize=6)

    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bat_trajectory.png'), dpi=300)
    plt.show()

