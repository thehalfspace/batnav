# plotting/glint_spacing.py

import matplotlib.pyplot as plt
import os
def plot_glint_spacing_estimates(trajectory_data: dict, tars: list = None, 
                                 output_dir: str = "data"):
    """
    Plot true glint spacing vs bat estimated glint spacing over steps.
    Parameters:
        tars (list of Target): Each with .index and .tin (true glint spacing)
        milestones (list of dict): Each with 'index', 'target_idx', 'success'
        glint_estimates (list of float): Bat's estimated glint spacings
    """
    milestones = trajectory_data.get("milestones", [])
    glint_estimates = trajectory_data.get("glint_spacing", [])

    fig, ax = plt.subplots()
    ax.set_title("Glint Spacing Estimates per Step")
    ax.set_xlabel("Steps ")
    ax.set_ylabel("Glint Spacing (µs)")

    # Plot target true spacings as horizontal dashed lines
    for t in tars:
        ax.axhline(y=t.tin, linestyle='--', color='gray', alpha=0.5)
        ax.text(x=0.5, y=t.tin + 3, s=f"T{t.index + 1}: {t.tin} µs", fontsize=7, color='tab:blue')

    # Plot bat's glint estimates
    for m, est in zip(milestones, glint_estimates):
        idx = m['index']
        target_idx = m['target_idx']
        success = m['success']
        color = 'tab:green' if success else 'tab:orange'
        ax.scatter(idx, est, color=color, s=40, label=f"T{target_idx}" if success else None)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "glint_spacings.png"), dpi=300)
    plt.show()

