# main.py
from config_loader import load_config, load_scenarios
from model.bat import Bat
from model.target import Target
from model.signal_generator import generate_sigs_with_delay, generate_multiglints
from model.wave_params import WaveParams
from model.utils import *
from model.echo_analyzer import linear_separate_window_10thresholds, estimate_glint_spacing
from model.scat_model import run_biscat_main
# from model.scat_model import run_biscat_main

import numpy as np
import matplotlib.pyplot as plt

# Reload modules for debugging: Can drop later
import importlib
#importlib.reload(generate_sigs_with_delay)
#importlib.reload(generate_multiglints)
# main.py


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
    max_steps = config.binaural.max_iterations

    visited_targets = []
    excluded = set()
    glint_estimates = []
    vec_all = []
    coordear_all = []

    for step in range(max_steps):
        print(f"\n🚩 Step {step+1}")

        available = [t.index for t in targets if t.index not in excluded]
        if not available:
            print("No targets left. Exiting.")
            break

        # --- Find nearest target
        nearest_idx = find_nearest_target(targets, bat.position, available)
        target = next(t for t in targets if t.index == nearest_idx)
        visited_targets.append(target.index)

        target_pos = polar_to_cartesian(target.r, target.theta)
        raw_delays = compute_delays_to_ears(bat, target_pos)
        alt_delays = apply_amplitude_latency_trading(bat, target_pos, raw_delays)

        # --- Generate echo signal
        ts = generate_sigs_with_delay(alt_delays)
        tsL = {"fs": sample_rate, "data": ts["data"][:, 0]}
        tsR = {"fs": sample_rate, "data": ts["data"][:, 1]}

        # --- Run SCAT filterbank
        simL = run_biscat_main(config, tsL)
        simR = run_biscat_main(config, tsR)

        print(f"🧪 BMM shape L: {simL['coch']['bmm'].shape}")
        print(f"🧪 Fc range: {simL['coch']['Fc'][0]:.1f} - {simL['coch']['Fc'][-1]:.1f} Hz")
        breakpoint()

        # --- Linear 10-threshold detection
        wave_params.simStruct = simL
        wave_params.NoT = 1
        _, first_gap_L = linear_separate_window_10thresholds(wave_params)

        wave_params.simStruct = simR
        wave_params.NoT = 1
        _, first_gap_R = linear_separate_window_10thresholds(wave_params)

        itd_samples = estimate_itd_from_histograms(first_gap_L, first_gap_R)
        print(f"🎧 Estimated ITD: {itd_samples:.1f} samples")

        # --- Movement logic
        bat.rotate_head(np.sign(alt_delays[0] - alt_delays[1]) * 15)
        bat.speed = choose_speed_from_delay(alt_delays[1], sample_rate)
        bat.move_forward()

        vec_all.append(bat.heading.copy())
        earL, earR = bat.get_ear_positions()
        coordear_all.append(np.concatenate([earL, earR]))

        # --- Glint spacing estimation
        glint_spacing = estimate_glint_spacing(bat, target, config, wave_params)
        if glint_spacing is None:
            print("❌ Glint spacing estimate failed.")
            excluded.add(target.index)
            continue

        glint_estimates.append(glint_spacing)
        print(f"📏 Glint spacing = {glint_spacing:.1f} µs")

        if abs(glint_spacing - desired_spacing_us) <= tolerance_us:
            print(f"✅ Target {target.index} matched spacing goal.")
            break
        else:
            print(f"❌ Target {target.index} spacing off by {abs(glint_spacing - desired_spacing_us):.1f} µs.")
            excluded.add(target.index)

    print("\n📍 Final position:", bat.position)
    print("🧭 Headings:", vec_all[-1])
    print("🦇 Ears:", coordear_all[-1])
    print("📊 Glint estimates (µs):", glint_estimates)
    print("🎯 Path: ", visited_targets)


def main():
    return 0


if __name__ == "__main__":
    debug_test_components()
    #main()

def debug_test_components():
    # Load configuration
    config = load_config()
    print("🔧 Config loaded.")
    print(f"  Sample rate: {config.wave.sample_rate} Hz")
    print(f"  Ear separation: {config.binaural.ear_separation} m")

    # Load target scenario
    targets = load_scenarios()
    print(f"🎯 Loaded {len(targets)} targets.")
    for t in targets[:3]:  # Preview first 3
        print(f"  - Target {t.index}: r={t.r} m, θ={t.theta}°, glint={t.tin} µs")
    
    # Initialize bat
    bat = Bat()
    print("Initial ears:", bat.get_ear_positions())
    print("Checking bat position:", bat.position)
    bat.rotate_head(15)
    bat.move_forward()
    print("Moved to:", bat.position)

    # Test signal_generator
    # Binaural signal with delay (L,R) = (2.5 m, 2.0 m)
    sig = generate_sigs_with_delay([2.5, 2.0])
    print("Stereo output:", sig["data"])

    # Multi-glint echo
    multi = generate_multiglints(2.0, 100)
    print("Multiglints: ", multi["data"])

    # Load wave parameters
    wavParams = WaveParams()
    print("Fs: ", wavParams.Fs)

    # Helper functions
    print("Testing helper functions:")
    bat_pos = np.array([0.0,0.0])
    available = [i for i in range(len(targets))]

    nearest = find_nearest_target(targets, bat_pos, available)
    print("Closest target:", targets[nearest])

    # Echo Analyzer
    breakpoint()



    # TODO: Initialize Bat and begin tracking loop
    print("\n🚧 TODO: Begin bat tracking loop here...")

