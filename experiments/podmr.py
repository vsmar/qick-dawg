"""
podmr.py — Pulsed-ODMR Fine-Resolution Experiment
===================================================
Runs a frequency sweep with MW pulse control, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from copy import copy
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import curve_fit

import qickdawg as qd
from qickdawg import PODMRFineRes

from config import load_config, build_nv_config, connect, ns_to_samples, mhz_to_freg

# =============================================================================
# EXPERIMENT PARAMETERS — edit these before each run
# =============================================================================

# Frequency sweep bounds in MHz — converted to freg automatically below.
# Use MHz here so the values are human-readable and comparable across setups.
ODMR_START_MHZ  = 1840.0   # MHz
ODMR_STOP_MHZ   = 1850.0   # MHz
ODMR_DELTA_MHZ  = 0.1      # MHz  (step size)

# MW Pulse duration in nanoseconds — converted to samples automatically.
# Override calibration value if not None.
MW_DURATION_NS  = None     # None = use calibration.pi_pulse_tns from default_transition

REPS            = 1

# Transition — set to "lower_dip", "upper_dip", or None to use config.yaml default.
# Swapping here also pulls in the correct freq_fMHz, mw_gain for that transition.
TRANSITION      = None     # None = use calibration.default_transition
PULSE_SEQ_DELAY_TUS = 0.2  # us — overrides config.yaml default for PODMR
GET_REFERENCE   = True     # acquire reference readout with MW gain = 0

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "podmr"

# =============================================================================
# Setup
# =============================================================================

cfg    = load_config()
connect(cfg)

# soccfg is needed for sample conversions
soccfg = qd.soccfg

config = build_nv_config(cfg)

# Convert sweep bounds from MHz → freg
mw_ch = cfg["hardware"]["mw_channel"]
start_freg = mhz_to_freg(ODMR_START_MHZ,  soccfg, mw_ch)
stop_freg  = mhz_to_freg(ODMR_STOP_MHZ,   soccfg, mw_ch)
delta_freg = mhz_to_freg(ODMR_DELTA_MHZ,  soccfg, mw_ch)

# Calculate number of sweep points from bounds and step
nsweep_points = int(round((ODMR_STOP_MHZ - ODMR_START_MHZ) / ODMR_DELTA_MHZ)) + 1

print(f"[podmr] Frequency sweep: {ODMR_START_MHZ} → {ODMR_STOP_MHZ} MHz "
      f"(Δ {ODMR_DELTA_MHZ} MHz)  |  {nsweep_points} points")
print(f"[podmr] In freg units: {start_freg} → {stop_freg}  (Δ {delta_freg})")

# Determine MW pulse duration (ns → tdds samples)
if MW_DURATION_NS is not None:
    mw_duration_tdds = ns_to_samples(MW_DURATION_NS, soccfg, mw_ch)
    duration_source = f"{MW_DURATION_NS} ns (user override)"
else:
    # Use calibration.pi_pulse_tns from the active transition
    t = cfg["calibration"][cfg["calibration"]["default_transition"]]
    mw_pi_ns = t["pi_pulse_tns"]
    if mw_pi_ns is None:
        raise ValueError(
            "MW_DURATION_NS is None and calibration.pi_pulse_tns is not set. "
            "Either set MW_DURATION_NS above or run a calibration first."
        )
    mw_duration_tdds = ns_to_samples(mw_pi_ns, soccfg, mw_ch)
    duration_source = f"{mw_pi_ns} ns (calibration.pi_pulse_tns)"

print(f"[podmr] MW duration: {duration_source} → {mw_duration_tdds} tdds samples")

# Apply experiment-specific overrides
if TRANSITION is not None:
    t = cfg["calibration"][TRANSITION]
    config.freq_fMHz = t["freq_fMHz"]
    config.mw_gain   = t["mw_gain"]

config.pulse_seq_delay_tus = PULSE_SEQ_DELAY_TUS
config.reps                = REPS
config.get_reference       = GET_REFERENCE

# Configure frequency sweep (start/stop in freg)
config.mw_start_freg = start_freg
config.mw_end_freg   = stop_freg
config.nsweep_points = nsweep_points

# Configure MW pulse
config.mw_duration_tdds = mw_duration_tdds

# =============================================================================
# Acquire
# =============================================================================

prog = PODMRFineRes(config)
data = prog.acquire(progress=True)

# =============================================================================
# Save to HDF5
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = OUTPUT_DIR / f"podmr_{timestamp}.h5"

with h5py.File(out_path, "w") as f:
    f.create_dataset("data", data=data)

    # Full config.yaml embedded as a string — file is self-documenting
    f.attrs["config_yaml"] = yaml.dump(cfg)

    # Experiment-specific parameters as explicit attributes
    exp = f.create_group("experiment")
    exp.attrs["odmr_start_mhz"]     = ODMR_START_MHZ
    exp.attrs["odmr_stop_mhz"]      = ODMR_STOP_MHZ
    exp.attrs["odmr_delta_mhz"]     = ODMR_DELTA_MHZ
    exp.attrs["odmr_start_freg"]    = start_freg
    exp.attrs["odmr_stop_freg"]     = stop_freg
    exp.attrs["odmr_delta_freg"]    = delta_freg
    exp.attrs["nsweep_points"]      = nsweep_points
    exp.attrs["mw_duration_ns"]     = MW_DURATION_NS or cfg["calibration"][cfg["calibration"]["default_transition"]].get("pi_pulse_tns")
    exp.attrs["mw_duration_tdds"]   = mw_duration_tdds
    exp.attrs["reps"]               = REPS
    exp.attrs["freq_mhz"]           = config.freq_fMHz
    exp.attrs["mw_gain"]            = config.mw_gain
    exp.attrs["transition"]         = TRANSITION or cfg["calibration"]["default_transition"]
    exp.attrs["pulse_seq_delay_tus"] = PULSE_SEQ_DELAY_TUS
    exp.attrs["get_reference"]      = GET_REFERENCE
    exp.attrs["timestamp"]          = timestamp

print(f"[podmr] Saved → {out_path}")

# =============================================================================
# Plot
# =============================================================================

def laplace_model(x, A, mu, b, C):
    return A * np.exp(-np.abs(x - mu) / b) + C


def fit_laplacian_dip(freq, signal):
    # Initial guesses from the lowest point and half-depth width estimate.
    A0 = np.min(signal) - np.max(signal)
    mu0 = freq[np.argmin(signal)]
    half_level = np.max(signal) + 0.5 * A0
    width_pts = np.sum(signal < half_level)
    dx = freq[1] - freq[0] if len(freq) > 1 else 1.0
    b0 = max(width_pts * dx / (2 * np.log(2)), dx)
    C0 = np.max(signal)

    popt, _ = curve_fit(
        laplace_model,
        freq,
        signal,
        p0=[A0, mu0, b0, C0],
        maxfev=5000,
    )
    return popt


def plot_dip_fit_and_annotate(axis, freq, signal):
    try:
        A, mu, b, C = fit_laplacian_dip(freq, signal)
        fit_curve = laplace_model(freq, A, mu, b, C)
        axis.plot(freq, fit_curve, color="black", linestyle="--", alpha=0.6, label="Dip fit")

        y_dip = np.min(signal)
        axis.plot(mu, y_dip, "ro", markersize=5)
        axis.annotate(
            f"{mu:.3f} MHz",
            xy=(mu, y_dip),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="red",
        )
        print(f"[podmr] Dip frequency (fit): {mu:.6f} MHz")
        return mu
    except Exception as exc:
        # Keep plotting robust if fitting fails on low-SNR traces.
        mu = freq[np.argmin(signal)]
        y_dip = np.min(signal)
        axis.plot(mu, y_dip, "ro", markersize=5)
        axis.annotate(
            f"{mu:.3f} MHz",
            xy=(mu, y_dip),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="red",
        )
        print(f"[podmr] Dip frequency (argmin): {mu:.6f} MHz (fit failed: {exc})")
        return mu

x_mhz = np.linspace(
    ODMR_START_MHZ,
    ODMR_STOP_MHZ,
    nsweep_points,
)

fig, ax = plt.subplots(figsize=(10, 5))

# Plot signal and reference if available
if len(data.shape) > 1 and data.shape[0] == 2:
    # data has shape (2, nsweep_points) — signal and reference
    signal = data[0]
    reference = data[1]
    contrast = signal / reference if np.all(reference > 0) else signal - reference
    
    ax.plot(x_mhz, signal, marker="o", markersize=4, linewidth=1.5, label="Signal", color="tab:blue")
    ax.plot(x_mhz, reference, marker="s", markersize=4, linewidth=1.5, label="Reference", color="tab:orange")
    ax_contrast = ax.twinx()
    ax_contrast.plot(x_mhz, contrast, marker="^", markersize=4, linewidth=1, label="Contrast", color="tab:green", alpha=0.6)
    ax_contrast.set_ylabel("Contrast (ratio or diff)", color="tab:green")
    plot_dip_fit_and_annotate(ax_contrast, x_mhz, contrast)
else:
    # data has shape (nsweep_points,) — signal only
    ax.plot(x_mhz, data, marker="o", markersize=4, linewidth=1.5)
    ax.legend(["Signal"])
    plot_dip_fit_and_annotate(ax, x_mhz, data)

ax.set_xlabel("MW Frequency (MHz)")
ax.set_ylabel("Signal (ADC counts)")
ax.set_title(
    f"Pulsed-ODMR  |  {timestamp}  |  "
    f"π-pulse={config.mw_gain} reg  |  "
    f"laser={cfg['optics']['excitation_laser_power_mW']} mW"
)
ax.grid(alpha=0.3)
fig.tight_layout()

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[podmr] Plot saved → {plot_path}")
