"""Generate various test scenarios for bat navigation simulation.

This script creates different target configurations to test the binaural tracking
system under various spatial distributions and conditions.
"""

import numpy as np
from model.scenario_generator import ScenarioGenerator, ScenarioConfig


def main():
    """Generate multiple scenario configurations."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize generator with default config
    generator = ScenarioGenerator()
    
    print("Generating scenarios...")
    
    # Generate uniform distribution scenarios
    generator.generate_uniform_scenario(n_targets=10, name="uniform_10")
    generator.generate_uniform_scenario(n_targets=20, name="uniform_20")
    generator.generate_uniform_scenario(n_targets=50, name="uniform_50")
    print("✓ Uniform scenarios generated")
    
    # Generate Gaussian distribution scenarios
    generator.generate_gaussian_scenario(n_targets=15, name="gaussian_center")
    generator.generate_gaussian_scenario(
        n_targets=20, name="gaussian_forward", 
        center=(0.0, 10.0), std=(2.0, 2.0)
    )
    generator.generate_gaussian_scenario(
        n_targets=20, name="gaussian_wide", 
        center=(0.0, 7.5), std=(4.0, 3.0)
    )
    print("✓ Gaussian scenarios generated")
    
    # Generate clustered scenarios
    generator.generate_clustered_scenario(n_targets=21, name="clustered_3", n_clusters=3)
    generator.generate_clustered_scenario(n_targets=25, name="clustered_5", n_clusters=5)
    print("✓ Clustered scenarios generated")
    
    # Generate ring scenarios
    generator.generate_ring_scenario(n_targets=30, name="ring_medium")
    generator.generate_ring_scenario(
        n_targets=40, name="ring_wide", 
        inner_radius=5.0, outer_radius=10.0
    )
    print("✓ Ring scenarios generated")
    
    # Generate grid scenarios
    generator.generate_grid_scenario(name="grid_2m", grid_spacing=2.0)
    generator.generate_grid_scenario(name="grid_1m", grid_spacing=1.0)
    print("✓ Grid scenarios generated")
    
    # Generate sparse scenario (few targets, large space)
    generator.generate_uniform_scenario(n_targets=5, name="sparse_5")
    print("✓ Sparse scenario generated")
    
    # Generate dense scenario (many targets, concentrated)
    dense_config = ScenarioConfig(
        x_bounds=(-5.0, 5.0),
        y_bounds=(5.0, 12.0)
    )
    dense_generator = ScenarioGenerator(dense_config)
    dense_generator.generate_uniform_scenario(n_targets=50, name="dense_30")
    print("✓ Dense scenario generated")
    
    print(f"\nAll scenarios saved to config/ directory")
    print("Available scenarios:")
    scenarios = [
        "uniform_10", "uniform_20", "uniform_50",
        "gaussian_center", "gaussian_forward", "gaussian_wide", 
        "clustered_3", "clustered_5",
        "ring_medium", "ring_wide",
        "grid_2m", "grid_1m",
        "sparse_5", "dense_30"
    ]
    
    for scenario in scenarios:
        print(f"  - {scenario}.csv")


if __name__ == "__main__":
    main()
