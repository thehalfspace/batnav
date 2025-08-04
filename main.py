# main.py
from config_loader import load_config, load_scenarios
from model.bat import Bat
from model.target import Target
from model.signal_generator import generate_sigs_with_delay, generate_multiglints
from model.wave_params import WaveParams
from model.utils import *
from model.echo_analyzer import linear_separate_window_10thresholds, estimate_glint_spacing
from model.scat_model import run_biscat_main
from plotting.trajectory import plot_static_trajectory

import numpy as np
import matplotlib.pyplot as plt


def run_binaural_tracking():
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

    #for step in range(max_steps):
    step = 0
    
    while abs(glint_spacing - desired_spacing_us) > tolerance_us:
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
        while True: # abs(itd_samples)> ITD_THRESHOLD:
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
            print("Bat Position: ", bat.position)

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
        # breakpoint()
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
            }

    print("\n📍 Final position:", bat.position)
    print("🧭 Headings:", bat.heading_history[-1])
    print("🦇 Ears:", bat.get_ear_positions())
    print("📊 Glint estimates (µs):", glint_estimates)
    print("🎯 Path: ", visited_targets)

    return trajectory_data, targets


def main():
    return 0


if __name__ == "__main__":
    td, tar = run_binaural_tracking()
    plot_static_trajectory(td, tar)

