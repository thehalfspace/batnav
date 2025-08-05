"""Scenario generator for binaural bat tracking simulations.

This module generates various target configurations for testing the bat navigation
system with different spatial distributions and glint spacing patterns.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""
    x_bounds: Tuple[float, float] = (-9.0, 9.0)  # meters
    y_bounds: Tuple[float, float] = (0.0, 15.0)  # meters
    glint_range: Tuple[float, float] = (0.0, 1000.0)  # microseconds
    target_glint: float = 100.0  # microseconds
    num_glints: int = 2  # NoG field
    

class ScenarioGenerator:
    """Generates target scenarios for bat navigation simulation."""
    
    def __init__(self, config: ScenarioConfig = None):
        """Initialize scenario generator.
        
        Args:
            config: Configuration for scenario bounds and parameters
        """
        self.config = config or ScenarioConfig()
        self.output_dir = Path("config")
        self.output_dir.mkdir(exist_ok=True)
    
    def _cartesian_to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """Convert cartesian coordinates to polar (r, theta in degrees).
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            
        Returns:
            Tuple of (radius, angle_degrees)
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.degrees(np.arctan2(y, x))
        # Convert to 0-360 range
        if theta < 0:
            theta += 360
        return r, theta
    
    def _ensure_target_glint(self, glint_spacings: np.ndarray) -> np.ndarray:
        """Ensure at least one target has the desired glint spacing.
        
        Args:
            glint_spacings: Array of glint spacing values
            
        Returns:
            Modified array with at least one target glint spacing
        """
        if len(glint_spacings) > 0:
            # Replace first target with desired glint spacing
            glint_spacings[0] = self.config.target_glint
        return glint_spacings
    
    def generate_uniform_scenario(self, n_targets: int, name: str) -> pl.DataFrame:
        """Generate targets with uniform spatial distribution.
        
        Args:
            n_targets: Number of targets to generate
            name: Scenario name for output file
            
        Returns:
            DataFrame with generated scenario
        """
        # Generate uniform coordinates
        x_coords = np.random.uniform(
            self.config.x_bounds[0], self.config.x_bounds[1], n_targets
        )
        y_coords = np.random.uniform(
            self.config.y_bounds[0], self.config.y_bounds[1], n_targets
        )
        
        # Convert to polar coordinates
        r_values = []
        theta_values = []
        for x, y in zip(x_coords, y_coords):
            r, theta = self._cartesian_to_polar(x, y)
            r_values.append(r)
            theta_values.append(theta)
        
        # Generate glint spacings
        glint_spacings = np.random.uniform(
            self.config.glint_range[0], self.config.glint_range[1], n_targets
        )
        glint_spacings = self._ensure_target_glint(glint_spacings)
        
        # Create DataFrame
        scenario_df = pl.DataFrame({
            'index': range(n_targets),
            'r': np.round(r_values, 2),
            'theta': np.round(theta_values, 2),
            'NoG': [self.config.num_glints] * n_targets,
            'tin': np.round(glint_spacings).astype(int)
        })
        
        # Save to file
        output_path = self.output_dir / f"{name}.csv"
        scenario_df.write_csv(output_path)
        
        return scenario_df
    
    def generate_gaussian_scenario(self, n_targets: int, name: str, 
                                 center: Tuple[float, float] = (0.0, 7.5),
                                 std: Tuple[float, float] = (3.0, 2.5)) -> pl.DataFrame:
        """Generate targets with Gaussian spatial distribution.
        
        Args:
            n_targets: Number of targets to generate
            name: Scenario name for output file
            center: (x, y) center of Gaussian distribution
            std: (x, y) standard deviations
            
        Returns:
            DataFrame with generated scenario
        """
        # Generate Gaussian coordinates with clipping to bounds
        x_coords = np.clip(
            np.random.normal(center[0], std[0], n_targets),
            self.config.x_bounds[0], self.config.x_bounds[1]
        )
        y_coords = np.clip(
            np.random.normal(center[1], std[1], n_targets),
            self.config.y_bounds[0], self.config.y_bounds[1]
        )
        
        # Convert to polar coordinates
        r_values = []
        theta_values = []
        for x, y in zip(x_coords, y_coords):
            r, theta = self._cartesian_to_polar(x, y)
            r_values.append(r)
            theta_values.append(theta)
        
        # Generate glint spacings
        glint_spacings = np.random.uniform(
            self.config.glint_range[0], self.config.glint_range[1], n_targets
        )
        glint_spacings = self._ensure_target_glint(glint_spacings)
        
        # Create DataFrame
        scenario_df = pl.DataFrame({
            'index': range(n_targets),
            'r': np.round(r_values, 2),
            'theta': np.round(theta_values, 2),
            'NoG': [self.config.num_glints] * n_targets,
            'tin': np.round(glint_spacings).astype(int)
        })
        
        # Save to file
        output_path = self.output_dir / f"{name}.csv"
        scenario_df.write_csv(output_path)
        
        return scenario_df
    
    def generate_clustered_scenario(self, n_targets: int, name: str,
                                  n_clusters: int = 3) -> pl.DataFrame:
        """Generate targets in spatial clusters.
        
        Args:
            n_targets: Number of targets to generate
            name: Scenario name for output file
            n_clusters: Number of spatial clusters
            
        Returns:
            DataFrame with generated scenario
        """
        # Generate cluster centers
        cluster_centers_x = np.random.uniform(
            self.config.x_bounds[0], self.config.x_bounds[1], n_clusters
        )
        cluster_centers_y = np.random.uniform(
            self.config.y_bounds[0], self.config.y_bounds[1], n_clusters
        )
        
        # Assign targets to clusters
        targets_per_cluster = n_targets // n_clusters
        remaining_targets = n_targets % n_clusters
        
        x_coords = []
        y_coords = []
        
        for i in range(n_clusters):
            n_in_cluster = targets_per_cluster + (1 if i < remaining_targets else 0)
            
            # Generate targets around cluster center
            cluster_x = np.random.normal(
                cluster_centers_x[i], 1.0, n_in_cluster
            )
            cluster_y = np.random.normal(
                cluster_centers_y[i], 1.0, n_in_cluster
            )
            
            # Clip to bounds
            cluster_x = np.clip(cluster_x, self.config.x_bounds[0], self.config.x_bounds[1])
            cluster_y = np.clip(cluster_y, self.config.y_bounds[0], self.config.y_bounds[1])
            
            x_coords.extend(cluster_x)
            y_coords.extend(cluster_y)
        
        # Convert to polar coordinates
        r_values = []
        theta_values = []
        for x, y in zip(x_coords, y_coords):
            r, theta = self._cartesian_to_polar(x, y)
            r_values.append(r)
            theta_values.append(theta)
        
        # Generate glint spacings
        glint_spacings = np.random.uniform(
            self.config.glint_range[0], self.config.glint_range[1], n_targets
        )
        glint_spacings = self._ensure_target_glint(glint_spacings)
        
        # Create DataFrame
        scenario_df = pl.DataFrame({
            'index': range(n_targets),
            'r': np.round(r_values, 2),
            'theta': np.round(theta_values, 2),
            'NoG': [self.config.num_glints] * n_targets,
            'tin': np.round(glint_spacings).astype(int)
        })
        
        # Save to file
        output_path = self.output_dir / f"{name}.csv"
        scenario_df.write_csv(output_path)
        
        return scenario_df
    
    def generate_ring_scenario(self, n_targets: int, name: str,
                             inner_radius: float = 3.0,
                             outer_radius: float = 8.0) -> pl.DataFrame:
        """Generate targets in a ring pattern.
        
        Args:
            n_targets: Number of targets to generate
            name: Scenario name for output file
            inner_radius: Inner radius of ring
            outer_radius: Outer radius of ring
            
        Returns:
            DataFrame with generated scenario
        """
        # Generate radii uniformly in ring area
        u = np.random.uniform(0, 1, n_targets)
        r_values = np.sqrt(u * (outer_radius**2 - inner_radius**2) + inner_radius**2)
        
        # Generate angles uniformly
        theta_values = np.random.uniform(0, 360, n_targets)
        
        # Convert to cartesian to check bounds
        x_coords = r_values * np.cos(np.radians(theta_values))
        y_coords = r_values * np.sin(np.radians(theta_values))
        
        # Filter targets within bounds
        valid_mask = (
            (x_coords >= self.config.x_bounds[0]) & 
            (x_coords <= self.config.x_bounds[1]) &
            (y_coords >= self.config.y_bounds[0]) & 
            (y_coords <= self.config.y_bounds[1])
        )
        
        r_values = r_values[valid_mask]
        theta_values = theta_values[valid_mask]
        n_valid = len(r_values)
        
        # Generate glint spacings
        glint_spacings = np.random.uniform(
            self.config.glint_range[0], self.config.glint_range[1], n_valid
        )
        glint_spacings = self._ensure_target_glint(glint_spacings)
        
        # Create DataFrame
        scenario_df = pl.DataFrame({
            'index': range(n_valid),
            'r': np.round(r_values, 2),
            'theta': np.round(theta_values, 2),
            'NoG': [self.config.num_glints] * n_valid,
            'tin': np.round(glint_spacings).astype(int)
        })
        
        # Save to file
        output_path = self.output_dir / f"{name}.csv"
        scenario_df.write_csv(output_path)
        
        return scenario_df
    
    def generate_grid_scenario(self, name: str, grid_spacing: float = 2.0) -> pl.DataFrame:
        """Generate targets in a regular grid pattern.
        
        Args:
            name: Scenario name for output file
            grid_spacing: Spacing between grid points
            
        Returns:
            DataFrame with generated scenario
        """
        # Generate grid coordinates
        x_points = np.arange(
            self.config.x_bounds[0], self.config.x_bounds[1] + grid_spacing, grid_spacing
        )
        y_points = np.arange(
            self.config.y_bounds[0], self.config.y_bounds[1] + grid_spacing, grid_spacing
        )
        
        x_grid, y_grid = np.meshgrid(x_points, y_points)
        x_coords = x_grid.flatten()
        y_coords = y_grid.flatten()
        
        # Remove origin point if it exists
        origin_mask = (x_coords != 0) | (y_coords != 0)
        x_coords = x_coords[origin_mask]
        y_coords = y_coords[origin_mask]
        
        n_targets = len(x_coords)
        
        # Convert to polar coordinates
        r_values = []
        theta_values = []
        for x, y in zip(x_coords, y_coords):
            r, theta = self._cartesian_to_polar(x, y)
            r_values.append(r)
            theta_values.append(theta)
        
        # Generate glint spacings
        glint_spacings = np.random.uniform(
            self.config.glint_range[0], self.config.glint_range[1], n_targets
        )
        glint_spacings = self._ensure_target_glint(glint_spacings)
        
        # Create DataFrame
        scenario_df = pl.DataFrame({
            'index': range(n_targets),
            'r': np.round(r_values, 2),
            'theta': np.round(theta_values, 2),
            'NoG': [self.config.num_glints] * n_targets,
            'tin': np.round(glint_spacings).astype(int)
        })
        
        # Save to file
        output_path = self.output_dir / f"{name}.csv"
        scenario_df.write_csv(output_path)
        
        return scenario_df