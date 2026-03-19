"""
counting_duration_fine_res.py — Counting Duration Fine Resolution Experiment
==============================================================================
Sweeps laser readout offset timing with fine (200ps) resolution control of MW
pulse duration. Saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import yaml

import qickdawg as qd
from qickdawg import CountingDurationFineRes

from config import load_config, build_nv_config, connect

# =============================================================================
# EXPERIMENT PARAMETERS — edit these before each run
# =============================================================================

# MW pulse duration in nanoseconds
# Set to a pi pulse from calibration, or override here.
MW_DURATION_NS = None  # None = use config.yaml default; else e.g. 650

# Laser readout offset sweep bounds (in nanoseconds)
READOUT_OFFSET_START_NS = 0.0      # ns
READOUT_OFFSET_STOP_NS  = 2000.0   # ns
READOUT_OFFSET_DELTA_NS = 100.0    # ns (step size)

REPS = 100000

# Transition — set to "lower_dip", "upper_dip", or None to use config.yaml default.
TRANSITION = None  # None = use calibration.default_transition

# Optional per-run overrides
OVERRIDE_FREQ_MHZ = None   # e.g. 1845.7
OVERRIDE_MW_GAIN  = None   # e.g. 1800

GET_REFERENCE = True  # acquire reference readout with MW gain = 0

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "counting_duration_fine_res"

# =============================================================================
# Setup
# =============================================================================

cfg    = load_config()
connect(cfg)

# soccfg is needed for sample conversions
soccfg = qd.soccfg

config = build_nv_config(cfg)

mw_ch = cfg["hardware"]["mw_channel"]

# Choose transition context
active_transition = TRANSITION or cfg["calibration"]["default_transition"]
t = cfg["calibration"][active_transition]

# Resolve run parameters
config.freq_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["freq_fMHz"]
config.mw_gain   = OVERRIDE_MW_GAIN  if OVERRIDE_MW_GAIN  is not None else t["mw_gain"]

# Determine MW pulse duration (ns → tdds samples)
mw_duration_ns = MW_DURATION_NS
if mw_duration_ns is None:
    mw_duration_ns = t["pi_pulse_tns"]
if mw_duration_ns is None:
    raise ValueError(
        "No pi pulse duration found. Set MW_DURATION_NS, "
        "or provide calibration.<transition>.pi_pulse_tns in config.yaml."
    )

# Convert ns to samples (tdds)
samps_per_clk = soccfg['gens'][mw_ch]['samps_per_clk']
ns_per_sample = (soccfg.cycles2us(1) * 1000) / samps_per_clk
config.mw_duration_tdds = int(round(mw_duration_ns / ns_per_sample))

print(f"[counting_duration_fine_res] MW duration: {mw_duration_ns} ns → {config.mw_duration_tdds} tdds samples")
print(f"[counting_duration_fine_res] Active transition: {active_transition} | freq={config.freq_fMHz} MHz | gain={config.mw_gain}")

# Apply experiment-specific settings
config.reps = REPS
config.get_reference = GET_REFERENCE

# =============================================================================
# Acquire (outer loop over laser readout offset)
# =============================================================================

readout_offsets_ns = np.arange(
    READOUT_OFFSET_START_NS,
    READOUT_OFFSET_STOP_NS + READOUT_OFFSET_DELTA_NS,
    READOUT_OFFSET_DELTA_NS,
)
nsweep_points = len(readout_offsets_ns)

# Accumulate results across multiple acquisitions
accumulated_data = []

for i, offset_ns in enumerate(readout_offsets_ns):
    # Convert offset from ns to registers
    offset_tus = offset_ns / 1000.0
    config.laser_readout_offset_tus = offset_tus
    
    # Acquire at this offset
    prog = CountingDurationFineRes(config)
    data = prog.acquire(progress=(i == 0))  # Show progress bar on first acquisition only
    
    accumulated_data.append(data)
    print(f"[counting_duration_fine_res] Acquired offset {offset_ns:.1f} ns ({i+1}/{nsweep_points})")

# Stack results: shape will depend on CountingDurationFineRes output
# Typically (nsweep_points,) for signal, or (2, nsweep_points) if reference is included
stacked_data = np.array(accumulated_data)

# =============================================================================
# Save to HDF5
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = OUTPUT_DIR / f"counting_duration_fine_res_{timestamp}.h5"

with h5py.File(out_path, "w") as f:
    f.create_dataset("data", data=stacked_data)

    # Full config.yaml embedded as a string
    f.attrs["config_yaml"] = yaml.dump(cfg)

    # Experiment-specific parameters
    exp = f.create_group("experiment")
    exp.attrs["readout_offset_start_ns"]  = READOUT_OFFSET_START_NS
    exp.attrs["readout_offset_stop_ns"]   = READOUT_OFFSET_STOP_NS
    exp.attrs["readout_offset_delta_ns"]  = READOUT_OFFSET_DELTA_NS
    exp.attrs["readout_offsets_ns"]       = readout_offsets_ns
    exp.attrs["nsweep_points"]            = nsweep_points
    exp.attrs["mw_duration_ns"]           = mw_duration_ns
    exp.attrs["mw_duration_tdds"]         = config.mw_duration_tdds
    exp.attrs["reps"]                     = REPS
    exp.attrs["freq_mhz"]                 = config.freq_fMHz
    exp.attrs["mw_gain"]                  = config.mw_gain
    exp.attrs["transition"]               = active_transition
    exp.attrs["get_reference"]            = GET_REFERENCE
    exp.attrs["timestamp"]                = timestamp

print(f"[counting_duration_fine_res] Saved → {out_path}")

# =============================================================================
# Plot
# =============================================================================

data_squeezed = np.squeeze(np.asarray(stacked_data))

data_squeezed = np.squeeze(np.asarray(stacked_data))

# Handle acquisition output shapes:
# - (N,) or (N,1): single signal (possibly after averaging)
# - (2,N): signal and reference (already averaged)
# - (nsweep_outer, N): raw data across sweep — use as-is or average if needed
# - (nsweep_outer, 2, N): raw signal/reference across sweep
reference = None

if data_squeezed.ndim == 1:
    signal = data_squeezed

elif data_squeezed.ndim == 2 and data_squeezed.shape[0] == 2:
    signal = data_squeezed[0]
    reference = data_squeezed[1]

elif data_squeezed.ndim == 2:
    # Data shape (nsweep_points, signal_dim) — already stacked from outer loop, use first dim
    signal = data_squeezed

elif data_squeezed.ndim == 3 and data_squeezed.shape[1] == 2:
    # Shape (nsweep_points, 2, signal_dim) — signal and reference stacked
    signal = data_squeezed[:, 0, :]
    reference = data_squeezed[:, 1, :]

else:
    raise ValueError(f"Unexpected data shape after squeeze: {data_squeezed.shape}")

fig, ax = plt.subplots(figsize=(10, 5))

if reference is not None:
    contrast = signal / reference if np.all(reference > 0) else signal - reference
    
    ax.plot(readout_offsets_ns, signal, marker="o", markersize=4, linewidth=1.5, label="Signal", color="tab:blue")
    ax.plot(readout_offsets_ns, reference, marker="s", markersize=4, linewidth=1.5, label="Reference", color="tab:orange")
    ax_contrast = ax.twinx()
    ax_contrast.plot(readout_offsets_ns, contrast, marker="^", markersize=4, linewidth=1, label="Contrast", color="tab:green", alpha=0.6)
    ax_contrast.set_ylabel("Contrast (ratio or diff)", color="tab:green")
    ax.legend(loc="upper left")
else:
    ax.plot(readout_offsets_ns, signal, marker="o", markersize=4, linewidth=1.5, label="Signal", color="tab:blue")
    ax.legend()

ax.set_xlabel("Laser Readout Offset (ns)")
ax.set_ylabel("Counts / s")
ax.set_title(
    f"Counting Duration Fine Resolution  |  {timestamp}  |  "
    f"MW={config.mw_gain} reg  |  "
    f"laser={cfg['optics']['excitation_laser_power_mW']} mW"
)
ax.grid(alpha=0.3)
fig.tight_layout()

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[counting_duration_fine_res] Plot saved → {plot_path}")
