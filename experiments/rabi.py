"""
rabi.py — Rabi Fine-Resolution Experiment
==========================================
Runs a Rabi sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

<<<<<<< HEAD
from copy import copy
=======
>>>>>>> 97d45ef1414acb8f1afa722f31298bbf46a5c75f
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import yaml

import qickdawg as qd
from qickdawg import RabiFineRes

from config import load_config, build_nv_config, connect, ns_to_samples

# =============================================================================
# EXPERIMENT PARAMETERS — edit these before each run
# =============================================================================

# Sweep bounds in nanoseconds — converted to samples automatically below.
# Use ns here so the values are human-readable and comparable across setups.
MW_DURATION_START_NS = 50     # ns
MW_DURATION_STOP_NS  = 500    # ns
MW_DURATION_DELTA_NS = 5      # ns  (step size)

REPS          = 1

# Transition — set to "lower_dip", "upper_dip", or None to use config.yaml default.
<<<<<<< HEAD
# Swapping here also pulls in the correct freq_fMHz, mw_gain for that transition.
TRANSITION    = None   # None = use calibration.default_transition
=======
TRANSITION    = None   # None = use calibration.default_transition

# Optional per-run overrides. If left None, values come from transition calibration.
OVERRIDE_FREQ_MHZ = None   # e.g. 1845.7
OVERRIDE_MW_GAIN  = None   # e.g. 1800

>>>>>>> 97d45ef1414acb8f1afa722f31298bbf46a5c75f
PULSE_SEQ_DELAY_TUS = 0.2     # us — overrides config.yaml default for Rabi
GET_REFERENCE = True          # acquire reference readout with MW gain = 0

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "rabi"

# =============================================================================
# Setup
# =============================================================================

cfg    = load_config()
connect(cfg)

# soccfg is needed for the sample conversion — adjust if qickdawg exposes it
# differently (e.g. qd.soccfg, or returned from start_client)
soccfg = qd.soccfg

config = build_nv_config(cfg)

# Convert sweep bounds from ns → samples (_tdds)
mw_ch = cfg["hardware"]["mw_channel"]
start_tdds = ns_to_samples(MW_DURATION_START_NS, soccfg, mw_ch)
stop_tdds  = ns_to_samples(MW_DURATION_STOP_NS,  soccfg, mw_ch)
delta_tdds = ns_to_samples(MW_DURATION_DELTA_NS, soccfg, mw_ch)

print(f"[rabi] Sweep: {start_tdds} → {stop_tdds} tdds  (Δ {delta_tdds})")

<<<<<<< HEAD
# Apply experiment-specific overrides
if TRANSITION is not None:
    t = cfg["calibration"][TRANSITION]
    config.freq_fMHz = t["freq_fMHz"]
    config.mw_gain   = t["mw_gain"]
=======
# Choose transition context: explicit TRANSITION, otherwise config default.
active_transition = TRANSITION or cfg["calibration"]["default_transition"]
t = cfg["calibration"][active_transition]

# Resolve run parameters with precedence:
# explicit file override -> selected/default transition values.
config.freq_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["freq_fMHz"]
config.mw_gain   = OVERRIDE_MW_GAIN  if OVERRIDE_MW_GAIN  is not None else t["mw_gain"]

print(f"[rabi] Active transition: {active_transition} | freq={config.freq_fMHz} MHz | gain={config.mw_gain}")

>>>>>>> 97d45ef1414acb8f1afa722f31298bbf46a5c75f
config.pulse_seq_delay_tus = PULSE_SEQ_DELAY_TUS
config.reps                = REPS
config.get_reference       = GET_REFERENCE

config.add_unitless_linear_sweep(
    "mw_duration_tdds",
    start_tdds,
    stop_tdds,
    delta=delta_tdds,
)

# =============================================================================
# Acquire
# =============================================================================

prog = RabiFineRes(config)
data = prog.acquire(progress=True)

# =============================================================================
# Save to HDF5
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = OUTPUT_DIR / f"rabi_{timestamp}.h5"

with h5py.File(out_path, "w") as f:
    f.create_dataset("data", data=data)

    # Full config.yaml embedded as a string — file is self-documenting
    f.attrs["config_yaml"] = yaml.dump(cfg)

    # Experiment-specific parameters as explicit attributes
    exp = f.create_group("experiment")
    exp.attrs["mw_duration_start_ns"]  = MW_DURATION_START_NS
    exp.attrs["mw_duration_stop_ns"]   = MW_DURATION_STOP_NS
    exp.attrs["mw_duration_delta_ns"]  = MW_DURATION_DELTA_NS
    exp.attrs["mw_duration_start_tdds"] = start_tdds
    exp.attrs["mw_duration_stop_tdds"]  = stop_tdds
    exp.attrs["mw_duration_delta_tdds"] = delta_tdds
    exp.attrs["reps"]                  = REPS
    exp.attrs["freq_fMHz"]             = config.freq_fMHz
<<<<<<< HEAD
    exp.attrs["transition"]            = TRANSITION or cfg["calibration"]["default_transition"]
=======
    exp.attrs["transition"]            = active_transition
>>>>>>> 97d45ef1414acb8f1afa722f31298bbf46a5c75f
    exp.attrs["pulse_seq_delay_tus"]   = PULSE_SEQ_DELAY_TUS
    exp.attrs["get_reference"]         = GET_REFERENCE
    exp.attrs["timestamp"]             = timestamp

print(f"[rabi] Saved → {out_path}")

# =============================================================================
# Plot
# =============================================================================

x_ns = np.arange(
    MW_DURATION_START_NS,
    MW_DURATION_STOP_NS + MW_DURATION_DELTA_NS,
    MW_DURATION_DELTA_NS,
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_ns, data, marker="o", markersize=3, linewidth=1)
ax.set_xlabel("MW Pulse Duration (ns)")
ax.set_ylabel("Signal (ADC counts)")
ax.set_title(
    f"Rabi  |  {timestamp}  |  "
    f"{config.freq_fMHz} MHz  |  gain={config.mw_gain}  |  "
    f"laser={cfg['optics']['excitation_laser_power_mW']} mW"
)
fig.tight_layout()

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[rabi] Plot saved → {plot_path}")