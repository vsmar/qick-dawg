"""
cpmg.py — CPMG-XY Fine-Resolution Experiment
============================================
Runs a CPMG-XY sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import qickdawg as qd
from qickdawg.nvtestsuite.cpmg_xy_subnano_fine_res_test import CPMGXYFineRes

from config import (
    load_config,
    build_nv_config,
    connect,
    ns_to_samples,
    get_ns_per_sample,
    save_experiment_hdf5,
)

# =============================================================================
# EXPERIMENT PARAMETERS — edit these before each run
# =============================================================================

# Sweep bounds in tdds (fine-resolution sample units).
TAU_START_TDDS = 500
TAU_STOP_TDDS = 10100
TAU_DELTA_TDDS = 25

# Optional sweep bounds in ns. If any of these is set, all three must be set,
# and they override the TAU_*_TDDS values above.
TAU_START_NS = None
TAU_STOP_NS = None
TAU_DELTA_NS = None

N_CPMG = 32
REPS = 100000

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_TDDS = None
OVERRIDE_MW_PI2_NS = None

GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cpmg"

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)

soccfg = qd.soccfg
config = build_nv_config(cfg)

mw_ch = cfg["hardware"]["mw_channel"]
ns_per_sample = get_ns_per_sample(soccfg, mw_ch)

active_transition = TRANSITION or cfg["calibration"]["default_transition"]
t = cfg["calibration"][active_transition]

config.mw_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["mw_fMHz"]
config.mw_gain = OVERRIDE_MW_GAIN if OVERRIDE_MW_GAIN is not None else t["mw_gain"]

if OVERRIDE_MW_PI2_TDDS is not None:
    if OVERRIDE_MW_PI2_NS is not None:
        raise ValueError("Set only one of OVERRIDE_MW_PI2_TDDS or OVERRIDE_MW_PI2_NS.")
    config.mw_pi2_tdds = int(OVERRIDE_MW_PI2_TDDS)
elif OVERRIDE_MW_PI2_NS is not None:
    config.mw_pi2_tdds = ns_to_samples(float(OVERRIDE_MW_PI2_NS), soccfg, mw_ch)
elif t.get("mw_pi2_tdds") is not None:
    config.mw_pi2_tdds = int(t["mw_pi2_tdds"])
else:
    raise ValueError(
        "No calibration pi/2 pulse found for this transition. "
        "Set OVERRIDE_MW_PI2_TDDS or provide calibration.<transition>.mw_pi2_tdds."
    )

config.n_cpmg = int(N_CPMG)
config.reps = int(REPS)
config.get_reference = bool(GET_REFERENCE)

use_tau_ns = any(v is not None for v in (TAU_START_NS, TAU_STOP_NS, TAU_DELTA_NS))
if use_tau_ns:
    if not all(v is not None for v in (TAU_START_NS, TAU_STOP_NS, TAU_DELTA_NS)):
        raise ValueError("If using ns sweep bounds, set TAU_START_NS, TAU_STOP_NS, and TAU_DELTA_NS.")

    tau_start_tdds = int(ns_to_samples(float(TAU_START_NS), soccfg, mw_ch))
    tau_stop_tdds = int(ns_to_samples(float(TAU_STOP_NS), soccfg, mw_ch))
    tau_delta_tdds = int(ns_to_samples(float(TAU_DELTA_NS), soccfg, mw_ch))
    if tau_delta_tdds <= 0:
        raise ValueError("TAU_DELTA_NS is too small after conversion; it must map to at least 1 tdds.")
else:
    tau_start_tdds = int(TAU_START_TDDS)
    tau_stop_tdds = int(TAU_STOP_TDDS)
    tau_delta_tdds = int(TAU_DELTA_TDDS)

tau_start_ns = tau_start_tdds * ns_per_sample
tau_stop_ns = tau_stop_tdds * ns_per_sample
tau_delta_ns = tau_delta_tdds * ns_per_sample

config.add_unitless_linear_sweep(
    "tau_tdds",
    tau_start_tdds,
    tau_stop_tdds,
    delta=tau_delta_tdds,
)

print(
    f"[cpmg] Sweep: {tau_start_tdds} -> {tau_stop_tdds} tdds "
    f"(delta {tau_delta_tdds})"
)
print(
    f"[cpmg] Sweep (ns): {tau_start_ns:.3f} -> {tau_stop_ns:.3f} ns "
    f"(delta {tau_delta_ns:.3f} ns)"
)
print(
    f"[cpmg] Active transition: {active_transition} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}"
)
print(f"[cpmg] n_cpmg={config.n_cpmg}, mw_pi2_tdds={config.mw_pi2_tdds}, reps={config.reps}")

# =============================================================================
# Acquire
# =============================================================================

prog = CPMGXYFineRes(config)
data = prog.acquire(progress=True)

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    CPMGXYFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="cpmg_xy_fine_res",
    ns_per_sample=ns_per_sample,
    custom_attrs={
        "tau_start_tdds": tau_start_tdds,
        "tau_stop_tdds": tau_stop_tdds,
        "tau_delta_tdds": tau_delta_tdds,
        "tau_start_ns": tau_start_ns,
        "tau_stop_ns": tau_stop_ns,
        "tau_delta_ns": tau_delta_ns,
        "n_cpmg": int(N_CPMG),
        "transition": active_transition,
        "get_reference": bool(GET_REFERENCE),
    },
)

print(f"[cpmg] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

if hasattr(data, "tau_tdds"):
    x_tdds = np.asarray(data.tau_tdds, dtype=float)
elif hasattr(data, "tau_samples"):
    x_tdds = np.asarray(data.tau_samples, dtype=float)
elif hasattr(data, "sweep_pts"):
    x_tdds = np.asarray(data.sweep_pts, dtype=float)
else:
    raise ValueError("CPMG output missing sweep axis (tau_tdds/tau_samples/sweep_pts).")

if x_tdds.ndim != 1:
    raise ValueError(f"Expected 1D sweep axis, got shape {x_tdds.shape}.")

x_us = x_tdds * ns_per_sample / 1000.0
pi2_ns = float(config.mw_pi2_tdds) * ns_per_sample

signal_raw = getattr(data, "signal1_cts_s", getattr(data, "signal1", None))
if signal_raw is None:
    raise ValueError("CPMG output missing signal1/signal1_cts_s field.")
signal = np.asarray(signal_raw, dtype=float)
if signal.ndim != 1:
    raise ValueError(f"Expected 1D signal data, got shape {signal.shape}.")

reference_raw = getattr(data, "signal2_cts_s", getattr(data, "signal2", None))
reference = np.asarray(reference_raw, dtype=float) if reference_raw is not None else None
if reference is not None and reference.ndim != 1:
    raise ValueError(f"Expected 1D reference data, got shape {reference.shape}.")

contrast_raw = getattr(data, "contrast", None)
if contrast_raw is not None:
    contrast = np.asarray(contrast_raw, dtype=float)
    if contrast.ndim != 1:
        raise ValueError(f"Expected 1D contrast data, got shape {contrast.shape}.")
elif reference is not None:
    contrast = signal / np.clip(reference, 1e-12, None)
else:
    contrast = None

if len(signal) != len(x_us):
    raise ValueError(f"Length mismatch: sweep has {len(x_us)} points but signal has {len(signal)}.")
if reference is not None and len(reference) != len(x_us):
    raise ValueError(f"Length mismatch: sweep has {len(x_us)} points but reference has {len(reference)}.")
if contrast is not None and len(contrast) != len(x_us):
    raise ValueError(f"Length mismatch: sweep has {len(x_us)} points but contrast has {len(contrast)}.")

fig, ax1 = plt.subplots(figsize=(10.0, 4.5))
ax2 = ax1.twinx() if contrast is not None else None

colors = {
    "signal": "tab:blue",
    "reference": "tab:orange",
    "contrast": "tab:green",
}

lines = []
ax1.plot(x_us, signal, "-", linewidth=1.2, alpha=0.35, color=colors["signal"])
lines += ax1.plot(
    x_us,
    signal,
    "o",
    markersize=6,
    linestyle="None",
    color=colors["signal"],
    markeredgecolor="white",
    markeredgewidth=0.4,
    label="signal",
)

if reference is not None:
    ax1.plot(x_us, reference, "-", linewidth=1.2, alpha=0.35, color=colors["reference"])
    lines += ax1.plot(
        x_us,
        reference,
        "o",
        markersize=6,
        linestyle="None",
        color=colors["reference"],
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="reference",
    )

if ax2 is not None:
    ax2.plot(x_us, contrast, "--", linewidth=1.0, alpha=0.4, color=colors["contrast"])
    lines += ax2.plot(
        x_us,
        contrast,
        "s",
        markersize=5.2,
        linestyle="None",
        color=colors["contrast"],
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="contrast ratio",
    )
    ax2.set_ylabel("Contrast ratio")

ax1.set_xlabel("Tau (us)")
ax1.set_ylabel("Counts/s")
ax1.set_title(
    f"CPMG-XY  |  {timestamp}  |  "
    f"mw={config.mw_fMHz:.3f} MHz  |  gain={config.mw_gain}  |  "
    f"pi/2={config.mw_pi2_tdds} tdds ({pi2_ns:.2f} ns)  |  "
    f"n={config.n_cpmg}  |  reps={config.reps}  |  "
    f"laser={cfg['optics']['excitation_laser_power_mW']} mW"
)
ax1.grid(alpha=0.2, linewidth=0.7)

if lines:
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        framealpha=0.98,
        ncols=1,
        facecolor="white",
    )

fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[cpmg] Plot saved -> {plot_path}")