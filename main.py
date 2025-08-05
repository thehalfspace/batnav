# main.py
from config_loader import load_config, load_scenarios, SCENARIO_PATH
from model.bat import Bat
from model.target import Target
from model.signal_generator import generate_sigs_with_delay, generate_multiglints
from model.wave_params import WaveParams
from model.utils import *
from model.echo_analyzer import linear_separate_window_10thresholds, estimate_glint_spacing
from model.scat_model import run_biscat_main
from plotting.trajectory import *
from plotting.glint_spacing import *
from pathlib import Path
from model.simdata_io import *

import numpy as np
import matplotlib.pyplot as plt
import sys
import logging

def create_output_dirs():
    folder_name = SCENARIO_PATH.stem
    output_path = Path.cwd() / Path('data/') / folder_name

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created Output directory {output_path}")
    return output_path

def setup_logging(output_path):
    """Setup logging to redirect print statements to output.log file."""
    log_file = output_path / "output.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )
    
    # Redirect print statements to logging
    class LogPrint:
        def write(self, text):
            if text.strip():  # Only log non-empty strings
                logging.info(text.strip())
        def flush(self):
            pass
    
    sys.stdout = LogPrint()
    print(f"Logging setup complete. Output will be saved to {log_file}")

def run_binaural_tracking():
    # Create output directory and setup logging
    output_path = create_output_dirs()
    setup_logging(output_path)
    
    # Load config and scenario
    config = load_config()
    targets = load_scenarios()
    wave_params = WaveParams()
    sample_rate = wave_params.Fs

    # Set up bat and simulation state
    bat = Bat()
    desired_spacing_us = 100
    tolerance_us = 10
    ITD_THRESHOLD = 20 # Samples
    glint_spacing = desired_spacing_us + 100 # initial value, must be wrong number
    max_steps = 50 # config.binaural.max_iterations

    visited_targets = []
    excluded = set()
    glint_estimates = []
    milestones = [] # Track the iterations: index, target_idx, success (glint estimation)
    step_log = []   # Log bat steps: index, target_idx, iteration

    #for step in range(max_steps):
    step = 0
    max_iteration = 0
    max_iteration_limit = 1000
    
    while abs(glint_spacing - desired_spacing_us) > tolerance_us and max_iteration < max_iteration_limit:
        print(f"\n🚩 Step {step+1}")
        step += 1

        available = [t.index for t in targets if t.index not in excluded]
        if not available:
            print("No targets left. Exiting.")
            break

        print("Available targets: ", available)

        # --- Find nearest target
        nearest_idx = find_nearest_target(targets, bat.position, available)
        target = next(t for t in targets if t.index == nearest_idx)
        visited_targets.append(target.index)

        target_pos = polar_to_cartesian(target.r, target.theta)

        # --- Rotate head until ITD (sense delay) is aligned ---
        while True and max_iteration < max_iteration_limit: # abs(itd_samples)> ITD_THRESHOLD:
            # Calculate delays
            raw_delays = compute_delays_to_ears(bat, target_pos)
            alt_delays = apply_amplitude_latency_trading(bat, target_pos, raw_delays)

            # --- Generate echo signal
            ts = generate_sigs_with_delay(alt_delays)
            tsL = {"fs": sample_rate, "data": ts["data"][:, 0]}
            tsR = {"fs": sample_rate, "data": ts["data"][:, 1]}

            # --- Run SCAT filterbank
            simL = run_biscat_main(config, tsL)
            simR = run_biscat_main(config, tsR)

            # --- Linear 10-threshold first gap detection
            wave_params.simStruct = simL
            wave_params.NoT = 1
            _, first_gap_L = linear_separate_window_10thresholds(wave_params)

            wave_params.simStruct = simR
            wave_params.NoT = 1
            _, first_gap_R = linear_separate_window_10thresholds(wave_params)

            # ITD estimate
            itd_samples = estimate_itd_from_histograms(first_gap_L, first_gap_R)
            print(f"🎧 Estimated ITD: {itd_samples:.1f} samples")

            # If aligned, break out of inner loop
            if abs(itd_samples) < ITD_THRESHOLD:
                print("Bat head is aligned: ")
                break

            # Else rotate and move

            # --- Movement logic
            rot_step = choose_rotation_step(itd_samples) # Choose how much to rotate
            direction = np.sign(alt_delays[0] - alt_delays[1]) 
            bat.rotate_head(direction * rot_step)
            bat.speed = choose_speed_from_delay(alt_delays[1], sample_rate)
            bat.move_forward()
            max_iteration += 1
            #max_iteration = len(bat.path_history)
            print("Bat Position: ", bat.position)

            # Log steps
            step_log.append({
                'index': len(bat.path_history) - 1, # Index in path history after movement
                'target_idx': target.index,         # Current target index
                'iteration': step                   # Current outer loop step
                })

        # --- Get dechirped response ---
        # Calculate delays
        raw_delays = compute_delays_to_ears(bat, target_pos)
        alt_delays = apply_amplitude_latency_trading(bat, target_pos, raw_delays)

        ts = generate_sigs_with_delay(alt_delays)
        tsL = {"fs": sample_rate, "data": ts["data"][:, 0]}
        tsR = {"fs": sample_rate, "data": ts["data"][:, 1]}

        simL = run_biscat_main(config, tsL)
        simR = run_biscat_main(config, tsR)

        wave_params.simStruct = simL
        wave_params.NoT = 1
        _, first_gap_L = linear_separate_window_10thresholds(wave_params)

        wave_params.simStruct = simR
        wave_params.NoT = 1
        _, first_gap_R = linear_separate_window_10thresholds(wave_params)

        # ITD initial estimate
        itd_samples = estimate_itd_from_histograms(first_gap_L, first_gap_R)

        # --- Glint spacing estimation (after alignment) ---
        print("Target: ", target.tin)
        glint_spacing = estimate_glint_spacing(bat, target, config, wave_params)

        if glint_spacing is None:
            print("❌ Glint spacing estimate failed.")
            glint_spacing = desired_spacing_us + 100 # Assign wrong value 
            #excluded.add(target.index)
            #continue
        
        # Track milestone
        milestones.append({
            'index': len(bat.path_history) - 1, # Current step index
            'target_idx': target.index,
            'success': abs(glint_spacing - desired_spacing_us) <= tolerance_us
            })

        glint_estimates.append(glint_spacing)
        print(f"📏 Glint spacing = {glint_spacing:.1f} µs")

        if abs(glint_spacing - desired_spacing_us) <= tolerance_us:
            print(f"✅ Target {target.index} matched spacing goal.")
            break
        else:
            print(f"❌ Target {target.index} spacing off by {abs(glint_spacing - desired_spacing_us):.1f} µs.")
            excluded.add(target.index)

    # --- Bundle data for animation ---
    trajectory_data = {
            "position": bat.path_history,
            "heading": bat.heading_history,
            "ears": bat.ear_history,
            "angles": bat.angle_history,
            "visited": visited_targets,
            "glint_spacing": glint_estimates,
            "milestones": milestones,
            "step_log": step_log,
            }

    print("\n📍 Final position:", bat.position)
    print("🧭 Headings:", bat.heading_history[-1])
    print("🦇 Ears:", bat.get_ear_positions())
    print("📊 Glint estimates (µs):", glint_estimates)
    print("🎯 Path: ", visited_targets)

    return trajectory_data, targets, output_path


def main():
    return 0


if __name__ == "__main__":
    td, tar, output_path = run_binaural_tracking()
    save_simulation_data(td, tar, output_path)
    plot_static_trajectory(td, tar, output_path)
    animate_bat_trajectory(td, tar, output_path)
    plot_bat_steps(td, output_path)
    plot_glint_spacing_estimates(td, tar, output_path)

