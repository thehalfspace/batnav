# main.py
from config_loader import load_config, load_scenarios
from model.bat import Bat
from model.target import Target
from model.signal_generator import generate_sigs_with_delay, generate_multiglints
from model.wave_params import WaveParams
from model.utils import *
from model.echo_analyzer import linear_separate_window_10thresholds, estimate_glint_spacing
from model.scat_model import run_biscat_main
from plotting.cochleagram import plot_bmm
from plotting.filterbank import plot_gammatone_filterbank #, plot_brian2hears_sos_filterbank
from plotting.trajectory import animate_bat_trajectory, plot_static_trajectory

import numpy as np
import matplotlib.pyplot as plt

# Reload modules for debugging: Can drop later
import importlib
#importlib.reload(generate_sigs_with_delay)
#importlib.reload(generate_multiglints)

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
    tarvec = []
    vec_all = []
    coordear_all = []

    for step in range(15): #max_steps):
        print(f"\n🚩 Step {step+1}")

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
        raw_delays = compute_delays_to_ears(bat, target_pos)
        alt_delays = apply_amplitude_latency_trading(bat, target_pos, raw_delays)

        # --- Generate echo signal
        ts = generate_sigs_with_delay(alt_delays)
        tsL = {"fs": sample_rate, "data": ts["data"][:, 0]}
        tsR = {"fs": sample_rate, "data": ts["data"][:, 1]}

        # --- Run SCAT filterbank
        simL = run_biscat_main(config, tsL)
        simR = run_biscat_main(config, tsR)

        # plot_bmm(simL["coch"]["bmm"], simL["coch"]["Fc"], sample_rate, title=f"BMM (Basilar Membrane Motion) for Target {target.index} (Left Ear)")
        # plot_brian2hears_sos_filterbank(simL["coch"]["gfb"], fs=sample_rate)



        print(f"🧪 BMM shape L: {simL['coch']['bmm'].shape}")
        print(f"🧪 Fc range: {simL['coch']['Fc'][0]:.1f} - {simL['coch']['Fc'][-1]:.1f} Hz")

        # --- Linear 10-threshold detection
        wave_params.simStruct = simL
        wave_params.NoT = 1
        _, first_gap_L = linear_separate_window_10thresholds(wave_params)

        wave_params.simStruct = simR
        wave_params.NoT = 1
        _, first_gap_R = linear_separate_window_10thresholds(wave_params)

        print("First_Gap_L size:", first_gap_L.shape)
        print("First_Gap_R size:", first_gap_R.shape)

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
            # excluded.add(target.index)
            continue

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
            "heading": vec_all,
            "ears": coordear_all,
            "angles": bat.angle_history,
            "visited": visited_targets,
            "glint_spacing": glint_estimates,
            }

    print("\n📍 Final position:", bat.position)
    print("🧭 Headings:", vec_all[-1])
    print("🦇 Ears:", coordear_all[-1])
    print("📊 Glint estimates (µs):", glint_estimates)
    print("🎯 Path: ", visited_targets)

    return trajectory_data, targets

    # Plot trajectory
    animate_bat_trajectory(trajectory_data)


def main():
    return 0


if __name__ == "__main__":
    td, tar = run_binaural_tracking()
    plot_static_trajectory(td, tar)
    #main()

