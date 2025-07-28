# main.py
from config_loader import load_config, load_scenarios
from model.bat import Bat

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

    # TODO: Initialize Bat and begin tracking loop
    print("\n🚧 TODO: Begin bat tracking loop here...")

if __name__ == "__main__":
    main()

