"""
rabi.py — Rabi Fine-Resolution Experiment
==========================================
Runs a Rabi sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning

import qickdawg as qd
from qickdawg import RabiFineRes

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

# Sweep bounds in nanoseconds — converted to samples automatically below.
# Use ns here so the values are human-readable and comparable across setups.
MW_DURATION_START_NS = 50     # ns
MW_DURATION_STOP_NS  = 300 #2000    # ns
MW_DURATION_DELTA_NS = 5      # ns  (step size)

REPS          = 50000 #150000

# Transition — set to "lower_dip", "upper_dip", or None to use config.yaml default.
TRANSITION    = None   # None = use calibration.default_transition

# Optional per-run overrides. If left None, values come from transition calibration.
OVERRIDE_FREQ_MHZ = None   # e.g. 1845.7
OVERRIDE_MW_GAIN  = None   # e.g. 20000

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

# Choose transition context: explicit TRANSITION, otherwise config default.
active_transition = TRANSITION or cfg["calibration"]["default_transition"]
t = cfg["calibration"][active_transition]

# Resolve run parameters with precedence:
# explicit file override -> selected/default transition values.
config.mw_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["mw_fMHz"]
config.mw_gain   = OVERRIDE_MW_GAIN  if OVERRIDE_MW_GAIN  is not None else t["mw_gain"]

print(f"[rabi] Active transition: {active_transition} | freq={config.mw_fMHz} MHz | gain={config.mw_gain}")
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

ns_per_sample = get_ns_per_sample(soccfg, mw_ch)
out_path, timestamp = save_experiment_hdf5(
    RabiFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="rabi_fine_res",
    ns_per_sample=ns_per_sample,
)

print(f"[rabi] Saved → {out_path}")

# =============================================================================
# Plot
# =============================================================================

# Build plotting arrays directly from analyzed RabiFineRes output.
# If these are not 1D, the sweep/source mapping is likely wrong upstream.
if not hasattr(data, "mw_duration_tdds"):
    raise ValueError("RabiFineRes output missing sweep field 'mw_duration_tdds'.")

x_tdds = np.asarray(data.mw_duration_tdds, dtype=float)
if x_tdds.ndim != 1:
    raise ValueError(f"Expected 1D mw_duration_tdds, got shape {x_tdds.shape}.")

x_ns = x_tdds * ns_per_sample

signal_raw = getattr(data, "signal1_cts_s", getattr(data, "signal1", None))
if signal_raw is None:
    raise ValueError("RabiFineRes output missing signal1/signal1_cts_s field.")
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

if len(signal) != len(x_ns):
    raise ValueError(
        f"Length mismatch: mw_duration_tdds has {len(x_ns)} points but signal has {len(signal)}."
    )
if reference is not None and len(reference) != len(x_ns):
    raise ValueError(
        f"Length mismatch: mw_duration_tdds has {len(x_ns)} points but reference has {len(reference)}."
    )
if contrast is not None and len(contrast) != len(x_ns):
    raise ValueError(
        f"Length mismatch: mw_duration_tdds has {len(x_ns)} points but contrast has {len(contrast)}."
    )


def _guess_rabi_period_ns(x_axis_ns: np.ndarray, y: np.ndarray) -> float:
    """Estimate dominant oscillation period from FFT of mean-centered data."""
    if len(x_axis_ns) < 4:
        return float(np.max(x_axis_ns) - np.min(x_axis_ns))

    dx = np.diff(x_axis_ns)
    if not np.all(dx > 0):
        return float(np.max(x_axis_ns) - np.min(x_axis_ns))

    dt = float(np.median(dx))
    if dt <= 0:
        return float(np.max(x_axis_ns) - np.min(x_axis_ns))

    y_centered = y - np.mean(y)
    freqs = np.fft.rfftfreq(len(y_centered), d=dt)
    spec = np.abs(np.fft.rfft(y_centered))
    if len(freqs) < 2:
        return float(np.max(x_axis_ns) - np.min(x_axis_ns))

    # Ignore DC bin and select strongest non-zero frequency.
    best_idx = int(np.argmax(spec[1:]) + 1)
    f0 = float(freqs[best_idx])
    if f0 <= 0:
        return float(np.max(x_axis_ns) - np.min(x_axis_ns))

    return 1.0 / f0


def _fit_rabi_contrast(x_axis_ns: np.ndarray, y: np.ndarray):
    """Fit contrast with phase fixed to zero; fallback to no-decay cosine."""

    def decaying_cos(x, amplitude, period_ns, offset, tau_ns):
        return offset + amplitude * np.cos(2.0 * np.pi * x / period_ns) * np.exp(-x / tau_ns)

    def no_decay_cos(x, amplitude, period_ns, offset):
        return offset + amplitude * np.cos(2.0 * np.pi * x / period_ns)

    span = float(np.max(x_axis_ns) - np.min(x_axis_ns))
    if span <= 0:
        return None

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = max(y_max - y_min, 1e-9)

    amplitude0 = 0.5 * y_range
    offset0 = float(np.mean(y))
    period0 = _guess_rabi_period_ns(x_axis_ns, y)
    period0 = float(np.clip(period0, max(1e-6, span / 20.0), max(span * 2.0, 1e-6)))
    tau0 = max(span, 1e-6)

    p0_full = [amplitude0, period0, offset0, tau0]
    bounds_full = (
        [0.0, max(1e-6, span / 50.0), y_min - y_range, max(1e-6, span / 50.0)],
        [3.0 * y_range, max(1e-6, span * 5.0), y_max + y_range, max(1e-6, span * 200.0)],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            params_full, _ = curve_fit(
                decaying_cos,
                x_axis_ns,
                y,
                p0=p0_full,
                bounds=bounds_full,
                maxfev=40000,
            )

            tau_fit = float(params_full[3])
            tau_effectively_infinite = tau_fit > 15.0 * span
            if not tau_effectively_infinite:
                fit_x = np.linspace(float(np.min(x_axis_ns)), float(np.max(x_axis_ns)), max(500, len(x_axis_ns) * 20))
                return {
                    "model": "decaying",
                    "params": params_full,
                    "fit_x": fit_x,
                    "fit_y": decaying_cos(fit_x, *params_full),
                }
        except (RuntimeError, ValueError, OptimizeWarning):
            pass

    # Fallback: if decay is not identifiable, fit a simpler no-decay model.
    p0_simple = [amplitude0, period0, offset0]
    bounds_simple = (
        [0.0, max(1e-6, span / 50.0), y_min - y_range],
        [3.0 * y_range, max(1e-6, span * 5.0), y_max + y_range],
    )

    try:
        params_simple, _ = curve_fit(
            no_decay_cos,
            x_axis_ns,
            y,
            p0=p0_simple,
            bounds=bounds_simple,
            maxfev=40000,
        )
        fit_x = np.linspace(float(np.min(x_axis_ns)), float(np.max(x_axis_ns)), max(500, len(x_axis_ns) * 20))
        return {
            "model": "no_decay",
            "params": params_simple,
            "fit_x": fit_x,
            "fit_y": no_decay_cos(fit_x, *params_simple),
        }
    except (RuntimeError, ValueError):
        return None

fig, ax1 = plt.subplots(figsize=(10.0, 4.5))
ax2 = ax1.twinx() if contrast is not None else None

colors = {
    "signal": "tab:blue",
    "reference": "tab:orange",
    "contrast": "tab:green",
    "fit": "black",
}

lines = []
ax1.plot(x_ns, signal, "-", linewidth=1.2, alpha=0.35, color=colors["signal"])
lines += ax1.plot(
    x_ns,
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
    ax1.plot(x_ns, reference, "-", linewidth=1.2, alpha=0.35, color=colors["reference"])
    lines += ax1.plot(
        x_ns,
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
    ax2.plot(x_ns, contrast, "--", linewidth=1.0, alpha=0.4, color=colors["contrast"])
    lines += ax2.plot(
        x_ns,
        contrast,
        "s",
        markersize=5.2,
        linestyle="None",
        color=colors["contrast"],
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="contrast ratio",
    )

    fit_summary_text = None
    fit = _fit_rabi_contrast(x_ns, contrast)
    if fit is not None:
        period_fit_ns = float(fit["params"][1])
        pi2_ns = period_fit_ns / 4.0
        rabi_freq_mhz = 1000.0 / period_fit_ns
        lines += ax2.plot(
            fit["fit_x"],
            fit["fit_y"],
            "-",
            linewidth=2.0,
            color=colors["fit"],
            label="contrast fit",
        )

        if fit["model"] == "decaying":
            tau_fit_ns = float(fit["params"][3])
            print(
                "[rabi] Contrast fit (decaying): "
                f"A={fit['params'][0]:.6g}, T={period_fit_ns:.3f} ns, "
                f"C={fit['params'][2]:.6g}, "
                f"tau={tau_fit_ns:.3f} ns"
            )
            fit_summary_text = (
                f"Fit (decaying cosine): A={fit['params'][0]:.4g}, "
                f"f_Rabi={rabi_freq_mhz:.3f} MHz, pi/2={pi2_ns:.2f} ns, "
                f"C={fit['params'][2]:.4g}, "
                f"tau={tau_fit_ns:.2f} ns"
            )
            ax2.text(
                0.02,
                0.98,
                (
                    f"f_Rabi = {rabi_freq_mhz:.3f} MHz\\n"
                    f"tau = {tau_fit_ns:.2f} ns"
                ),
                transform=ax2.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray"),
            )
        else:
            print(
                "[rabi] Contrast fit (no decay): "
                f"A={fit['params'][0]:.6g}, T={period_fit_ns:.3f} ns, "
                f"C={fit['params'][2]:.6g}"
            )
            fit_summary_text = (
                f"Fit (cosine): A={fit['params'][0]:.4g}, "
                f"f_Rabi={rabi_freq_mhz:.3f} MHz, pi/2={pi2_ns:.2f} ns, "
                f"C={fit['params'][2]:.4g}"
            )
            ax2.text(
                0.02,
                0.98,
                f"f_Rabi = {rabi_freq_mhz:.3f} MHz",
                transform=ax2.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray"),
            )
    else:
        print("[rabi] Contrast fit skipped: fit did not converge.")
        fit_summary_text = "Fit: did not converge."

    ax2.set_ylabel("Contrast ratio")

ax1.set_xlabel("MW Pulse Duration (ns)")
ax1.set_ylabel("Counts/s")
ax1.set_title(
    f"Rabi  |  {timestamp}  |  "
    f"{config.mw_fMHz:.3f} MHz  |  gain={config.mw_gain}  |  "
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

if ax2 is not None and fit_summary_text is not None:
    fig.text(
        0.5,
        0.01,
        fit_summary_text,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
else:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[rabi] Plot saved → {plot_path}")