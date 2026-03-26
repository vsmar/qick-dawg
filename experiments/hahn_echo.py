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
from scipy.optimize import curve_fit

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
TAU_END_FTNS = 800_000.0
TAU_DELTA_FTNS = 0.0 # Ignored when scaling_mode is 'exponential'
SCALING_FACTOR = "9/8" # "9/8"

REPS = 50_000

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
run_id = out_path.stem

print(f"[hahn_echo] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

if not hasattr(data, "tau_ftus"):
    raise ValueError("Hahn echo output missing expected sweep axis tau_ftus.")
x_axis = np.asarray(data.tau_ftus, dtype=float)

traces = extract_standard_traces(data, x_axis=x_axis, use_counts_s=PLOT_USE_COUNTS_S)
metadata = {
    "run_id": run_id,
    "mw_MHz": f"{config.mw_fMHz:.3f}",
    "gain": config.mw_gain,
    "pi2_ftsamp": config.mw_pi2_ftsamp,
    "pi2_ftns": f"{config.mw_pi2_ftns:.2f}",
    "sequence": "pi/2 - tau - pi - tau - pi/2",
    "reps": config.reps,
    "laser_mW": cfg["optics"]["excitation_laser_power_mW"],
    "units": "cts/s" if PLOT_USE_COUNTS_S else "raw",
}

# =============================================================================
# T2 Fitting
# =============================================================================

def _exp_decay_t2(tau_us_arr, amplitude, t2_us, offset):
    """Exponential decay model for T2: y = C + A*exp(-2*tau/T2)"""
    tau_us_arr = np.asarray(tau_us_arr, dtype=float)
    safe_t2 = np.clip(float(t2_us), 1e-9, None)
    return offset + amplitude * np.exp(-2.0 * np.clip(tau_us_arr, 0.0, None) / safe_t2)


def _fit_t2_contrast(x_axis, contrast_data):
    """Fit contrast vs tau to T2 exponential decay."""
    x_us = np.asarray(x_axis, dtype=float).reshape(-1)
    y = np.asarray(contrast_data, dtype=float).reshape(-1)

    finite = np.isfinite(x_us) & np.isfinite(y)
    x_us = x_us[finite]
    y = y[finite]
    if len(x_us) < 5:
        return None

    order = np.argsort(x_us)
    x_us = x_us[order]
    y = y[order]

    span = float(np.max(x_us) - np.min(x_us))
    if span <= 0:
        return None

    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_range = max(y_hi - y_lo, float(np.std(y)) * 2.0, 1e-6)

    p0 = [
        float(y[0] - y[-1]),
        max(span / 2.0, 1e-3),
        float(y[-1]),
    ]
    bounds = (
        [-5.0 * y_range, max(span / 500.0, 1e-3), y_lo - y_range],
        [5.0 * y_range, max(span * 20.0, 10.0), y_hi + y_range],
    )

    try:
        params, cov = curve_fit(
            _exp_decay_t2,
            x_us,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=60000,
        )
        fit_x = np.linspace(float(np.min(x_us)), float(np.max(x_us)), max(500, len(x_us) * 20))
        fit_sigma = np.sqrt(np.diag(cov)) if cov is not None else np.full(len(params), np.nan)
        return {
            "params": params,
            "sigma": fit_sigma,
            "fit_x": fit_x,
            "fit_y": _exp_decay_t2(fit_x, *params),
            "t2_us": float(params[1]),
            "t2_sigma_us": float(fit_sigma[1]) if len(fit_sigma) > 1 else np.nan,
        }
    except (RuntimeError, ValueError):
        return None


if PLOT_DEBUG_RAW:
    plot_debug_traces(
        x_axis,
        traces,
        x_label=r"$\tau$ (us)",
        y_label="Counts/s" if PLOT_USE_COUNTS_S else "Counts",
        title=f"Hahn Echo Debug Raw | $\\pi/2 - \\tau - \\pi - \\tau - \\pi/2$ | {timestamp}",
        metadata=metadata,
        metadata_position=PLOT_METADATA_POSITION,
    )

# Fit T2 from contrast trace
contrast = traces.get("contrast")
t2_fit = None
if contrast is not None and len(contrast) >= 5:
    t2_fit = _fit_t2_contrast(x_axis, contrast)
    if t2_fit is not None:
        print(f"[hahn_echo] T2 fit: A={t2_fit['params'][0]:.6g}, T2={t2_fit['t2_us']:.3f} us, C={t2_fit['params'][2]:.6g}")

# Plot with or without fit
fig, ax_contrast, ax_signals, _ = plot_contrast_twin(
    x_axis,
    traces,
    x_label=r"$\tau$ (us)",
    title=f"Hahn Echo | $\\pi/2 - \\tau - \\pi - \\tau - \\pi/2$ | {timestamp}",
    metadata=metadata,
    metadata_position=PLOT_METADATA_POSITION,
)

# Add T2 fit curve and annotation to contrast panel if fit succeeded
if t2_fit is not None and ax_contrast is not None:
    ax_contrast.plot(t2_fit["fit_x"], t2_fit["fit_y"], "-", linewidth=2.0, color="black", alpha=0.8, label="T2 fit")
    
    if np.isfinite(t2_fit["t2_sigma_us"]):
        t2_text = f"T2 = {t2_fit['t2_us']:.3f} +/- {t2_fit['t2_sigma_us']:.3f} us"
    else:
        t2_text = f"T2 = {t2_fit['t2_us']:.3f} us"
    fit_text = f"Fit: y = C + A*exp(-2*τ/T2), A={t2_fit['params'][0]:.4g}, C={t2_fit['params'][2]:.4g}"
    
    ax_contrast.text(
        0.02,
        0.98,
        f"{t2_text}\n{fit_text}",
        transform=ax_contrast.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    ax_contrast.legend(loc="upper right", framealpha=0.95)

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[hahn_echo] Plot saved -> {plot_path}")
