# main.py
from config_loader import load_config, load_scenarios
from model.bat import Bat
from model.signal_generator import generate_sigs_with_delay, generate_multiglints
#import model.signal_generator as bsig


# Reload modules for debugging: Can drop later
import importlib
#importlib.reload(generate_sigs_with_delay)
#importlib.reload(generate_multiglints)

def main():
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
    breakpoint()

    # 

    # TODO: Initialize Bat and begin tracking loop
    print("\n🚧 TODO: Begin bat tracking loop here...")

if __name__ == "__main__":
    main()

