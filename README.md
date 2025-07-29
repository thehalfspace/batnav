Here’s a minimal `README.md` tailored for your BatNav project:

---

````markdown
# 🦇 BatNav: Bat Navigation via Binaural Tracking

**BatNav** is a Python reimplementation of a biologically inspired bat navigation model based on SCAT (Spectrogram Correlation and Transformation). It simulates a bat using binaural hearing to track and locate glint-reflecting targets in a 2D space.

> 🧬 *Adapted from the original MATLAB SCAT implementation:*  
> https://github.com/gomingchen/SCAT

---

## 📦 Installation (Python ≥ 3.10)

We use [uv](https://github.com/astral-sh/uv) for fast, lockfile-based installs:

```bash
# Clone the repository
git clone https://github.com/yourusername/batnav
cd batnav

# Install dependencies via pyproject.toml using uv
uv venv
uv pip install -e .  # Editable mode
````

To use `uv` as your Python:

```bash
source .venv/bin/activate
```

---

## 🚀 Running the Simulation

```bash
python main.py
```

Outputs include:

* Console logs of tracking steps and glint spacing estimates
* Static trajectory plot with visited targets and ear positions
* Animation of bat movement (WIP)

---

## 🧠 Method Overview

* **Signal Generation**: Computes left/right ear delays to target with amplitude-latency trading (ALT).
* **Filterbank Processing**: Uses `brian2hears` gammatone filters to simulate cochlear response (`bmm`).
* **Thresholding**: Detects echo onsets using a 10-threshold linear separation method with smoothing.
* **ITD Estimation**: Estimates interaural time difference (ITD) from histograms of glint gaps.
* **Movement Logic**: Bat turns head based on ITD and moves toward target until glint spacing matches a goal.
* **Tracking Loop**: Bats navigate a series of polar-distributed targets using binaural cues.

---

## 📁 Project Structure

```
batnav/
├── main.py                  # Core tracking loop
├── model/                   # Signal processing and tracking logic
├── plotting/                # Static and animated trajectory visualizations
├── config/                  # YAML scenario configs 
├── pyproject.toml           # Dependency & package info
└── README.md
```

---

## 🧪 Status

🔧 WIP
🔧 Debugging echo/glint mismatches
🎞️ Animation in progress

```

---

```

