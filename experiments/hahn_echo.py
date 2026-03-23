"""
hahn_echo.py — Hahn Echo Fine-Resolution Experiment
====================================================
Runs a Hahn echo tau sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qickdawg.nvtestsuite.hahn_echo_fine_res import HahnEchoFineRes

from config import (
    load_config,
    build_nv_config,
    connect,
    save_experiment_hdf5,
)
from plotting_utils import (
    extract_standard_traces,
    plot_debug_traces,
    plot_contrast_twin,
)

# =============================================================================
# EXPERIMENT PARAMETERS — edit these before each run
# =============================================================================

# Sweep bounds in fine-time nanoseconds (ftns).
TAU_START_FTNS = 200.0
TAU_END_FTNS = 1_000_000.0//60
TAU_DELTA_FTNS = 0.0 # Ignored when scaling_mode is 'exponential'
SCALING_FACTOR = "3/2"

REPS = 50000

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

GET_REFERENCE = True
PLOT_USE_COUNTS_S = True
PLOT_DEBUG_RAW = False
PLOT_METADATA_POSITION = "bottom"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "hahn_echo"

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)

config = build_nv_config(cfg)

active_transition = TRANSITION or cfg["calibration"]["default_transition"]
t = cfg["calibration"][active_transition]

config.mw_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["mw_fMHz"]
config.mw_gain = OVERRIDE_MW_GAIN if OVERRIDE_MW_GAIN is not None else t["mw_gain"]

if OVERRIDE_MW_PI2_FTSAMP is not None:
    if OVERRIDE_MW_PI2_FTNS is not None:
        raise ValueError("Set only one of OVERRIDE_MW_PI2_FTSAMP or OVERRIDE_MW_PI2_FTNS.")
    config.mw_pi2_ftsamp = int(OVERRIDE_MW_PI2_FTSAMP)
elif OVERRIDE_MW_PI2_FTNS is not None:
    config.mw_pi2_ftns = float(OVERRIDE_MW_PI2_FTNS)
elif t.get("mw_pi2_ftsamp") is not None:
    config.mw_pi2_ftsamp = int(t["mw_pi2_ftsamp"])
else:
    raise ValueError(
        "No calibration pi/2 pulse found for this transition. "
        "Set OVERRIDE_MW_PI2_FTSAMP or provide calibration.<transition>.mw_pi2_ftsamp."
    )

config.reps = int(REPS)
config.get_reference = bool(GET_REFERENCE)

# Keep tau delta populated for downstream code paths even with exponential sweep.
config.tau_delta_ftns = float(TAU_DELTA_FTNS)

config.add_exponential_sweep(
    "tau",
    "ftns",
    start=TAU_START_FTNS,
    stop=TAU_END_FTNS,
    scaling_factor=SCALING_FACTOR,
)

print(
    f"[hahn_echo] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
    f"(exponential, scaling_factor={SCALING_FACTOR}, configured delta={config.tau_delta_ftns:.3f})"
)
print(
    f"[hahn_echo] Active transition: {active_transition} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}"
)
print(f"[hahn_echo] mw_pi2_ftsamp={config.mw_pi2_ftsamp}, reps={config.reps}")

# =============================================================================
# Acquire
# =============================================================================

prog = HahnEchoFineRes(config)
data = prog.acquire(progress=True)

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    HahnEchoFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="hahn_echo_fine_res",
)

print(f"[hahn_echo] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

if not hasattr(data, "tau_ftus"):
    raise ValueError("Hahn echo output missing expected sweep axis tau_ftus.")
x_axis = np.asarray(data.tau_ftus, dtype=float)

traces = extract_standard_traces(data, x_axis=x_axis, use_counts_s=PLOT_USE_COUNTS_S)
metadata = {
    "mw_MHz": f"{config.mw_fMHz:.3f}",
    "gain": config.mw_gain,
    "pi2_ftsamp": config.mw_pi2_ftsamp,
    "pi2_ftns": f"{config.mw_pi2_ftns:.2f}",
    "reps": config.reps,
    "laser_mW": cfg["optics"]["excitation_laser_power_mW"],
    "units": "cts/s" if PLOT_USE_COUNTS_S else "raw",
}

if PLOT_DEBUG_RAW:
    plot_debug_traces(
        x_axis,
        traces,
        x_label="Tau (ftus)",
        y_label="Counts/s" if PLOT_USE_COUNTS_S else "Counts",
        title=f"Hahn Echo Debug Raw | {timestamp}",
        metadata=metadata,
        metadata_position=PLOT_METADATA_POSITION,
    )

fig, _, _, _ = plot_contrast_twin(
    x_axis,
    traces,
    x_label="Tau (ftus)",
    title=f"Hahn Echo | {timestamp}",
    metadata=metadata,
    metadata_position=PLOT_METADATA_POSITION,
)

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[hahn_echo] Plot saved -> {plot_path}")
