"""
podmr.py — Pulsed-ODMR Fine-Resolution Experiment
==================================================
Runs a PODMR frequency sweep, saves data + full config to HDF5, and plots
signal/reference/contrast with a dip fit on contrast ratio.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import qickdawg as qd
from qickdawg import PODMRFineRes

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

# Sweep bounds in MHz.
ODMR_START_MHZ = 1843.0
ODMR_STOP_MHZ = 1849.0
ODMR_DELTA_MHZ = 0.2

REPS = 100000

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_MW_GAIN = 2000

# Set either tdds directly, or ns (which will be converted to tdds).
OVERRIDE_MW_PI_TDDS = None
OVERRIDE_MW_PI_NS = 339.5*2

GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "podmr"

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)

soccfg = qd.soccfg
config = build_nv_config(cfg)

mw_ch = cfg["hardware"]["mw_channel"]
active_transition = TRANSITION or cfg["calibration"]["default_transition"]
t = cfg["calibration"][active_transition]

config.mw_gain = OVERRIDE_MW_GAIN if OVERRIDE_MW_GAIN is not None else t["mw_gain"]

if OVERRIDE_MW_PI_TDDS is not None and OVERRIDE_MW_PI_NS is not None:
    raise ValueError("Set only one of OVERRIDE_MW_PI_TDDS or OVERRIDE_MW_PI_NS.")

if OVERRIDE_MW_PI_TDDS is not None:
    config.mw_pi_tdds = int(OVERRIDE_MW_PI_TDDS)
    pi_source = f"OVERRIDE_MW_PI_TDDS={config.mw_pi_tdds}"
elif OVERRIDE_MW_PI_NS is not None:
    config.mw_pi_tdds = ns_to_samples(float(OVERRIDE_MW_PI_NS), soccfg, mw_ch)
    pi_source = f"OVERRIDE_MW_PI_NS={OVERRIDE_MW_PI_NS} ns"
else:
    if t.get("mw_pi_tdds") is None:
        raise ValueError(
            "No calibration pi pulse found for this transition. "
            "Set OVERRIDE_MW_PI_TDDS (or OVERRIDE_MW_PI_NS)."
        )
    config.mw_pi_tdds = int(t["mw_pi_tdds"])
    pi_source = f"calibration.{active_transition}.mw_pi_tdds={config.mw_pi_tdds}"

config.reps = REPS
config.get_reference = GET_REFERENCE

config.add_linear_sweep("mw", "fMHz", start=ODMR_START_MHZ, stop=ODMR_STOP_MHZ, delta=ODMR_DELTA_MHZ)

print(
    f"[podmr] Sweep: {ODMR_START_MHZ:.3f} -> {ODMR_STOP_MHZ:.3f} MHz "
    f"(delta {ODMR_DELTA_MHZ:.3f} MHz)"
)
print(f"[podmr] Active transition: {active_transition} | mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}")
print(f"[podmr] pi pulse: {pi_source}")

# =============================================================================
# Acquire
# =============================================================================

prog = PODMRFineRes(config)
data = prog.acquire(progress=True)

# =============================================================================
# Save to HDF5
# =============================================================================

ns_per_sample = get_ns_per_sample(soccfg, mw_ch)
out_path, timestamp = save_experiment_hdf5(
    PODMRFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="podmr_fine_res",
    ns_per_sample=ns_per_sample,
    custom_attrs={
        "odmr_start_mhz": ODMR_START_MHZ,
        "odmr_stop_mhz": ODMR_STOP_MHZ,
        "odmr_delta_mhz": ODMR_DELTA_MHZ,
        "transition": active_transition,
        "get_reference": GET_REFERENCE,
    },
)

print(f"[podmr] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

if hasattr(data, "mw_fMHz"):
    x_mhz = np.asarray(data.mw_fMHz, dtype=float)
elif hasattr(data, "frequencies"):
    x_mhz = np.asarray(data.frequencies, dtype=float)
else:
    raise ValueError("PODMRFineRes output missing frequency sweep axis (mw_fMHz/frequencies).")

if x_mhz.ndim != 1:
    raise ValueError(f"Expected 1D frequency axis, got shape {x_mhz.shape}.")

signal_raw = getattr(data, "signal1_cts_s", getattr(data, "signal1", None))
if signal_raw is None:
    raise ValueError("PODMRFineRes output missing signal1/signal1_cts_s field.")
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

if len(signal) != len(x_mhz):
    raise ValueError(f"Length mismatch: frequency has {len(x_mhz)} points but signal has {len(signal)}.")
if reference is not None and len(reference) != len(x_mhz):
    raise ValueError(f"Length mismatch: frequency has {len(x_mhz)} points but reference has {len(reference)}.")
if contrast is not None and len(contrast) != len(x_mhz):
    raise ValueError(f"Length mismatch: frequency has {len(x_mhz)} points but contrast has {len(contrast)}.")


def _fit_podmr_dip(freq_axis_mhz: np.ndarray, y: np.ndarray):
    """Fit contrast dip with a Laplace profile; fallback to argmin if fit fails."""

    def laplace_dip(x, depth, center_mhz, width_mhz, baseline):
        return baseline - depth * np.exp(-np.abs(x - center_mhz) / width_mhz)

    span = float(np.max(freq_axis_mhz) - np.min(freq_axis_mhz))
    if span <= 0:
        return None

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = max(y_max - y_min, 1e-9)

    depth0 = max(float(np.median(y)) - y_min, 1e-6)
    center0 = float(freq_axis_mhz[int(np.argmin(y))])
    width0 = max(span / 25.0, 1e-6)
    baseline0 = float(np.median(y))

    p0 = [depth0, center0, width0, baseline0]
    bounds = (
        [1e-7, float(np.min(freq_axis_mhz)), 1e-7, y_min - y_range],
        [max(5.0 * y_range, 1e-6), float(np.max(freq_axis_mhz)), span, y_max + y_range],
    )

    try:
        params, _ = curve_fit(
            laplace_dip,
            freq_axis_mhz,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=30000,
        )
        fit_x = np.linspace(float(np.min(freq_axis_mhz)), float(np.max(freq_axis_mhz)), max(500, len(freq_axis_mhz) * 20))
        return {
            "model": "laplace_dip",
            "params": params,
            "fit_x": fit_x,
            "fit_y": laplace_dip(fit_x, *params),
        }
    except (RuntimeError, ValueError):
        idx = int(np.argmin(y))
        return {
            "model": "argmin",
            "params": np.array([np.nan, float(freq_axis_mhz[idx]), np.nan, np.nan]),
            "fit_x": None,
            "fit_y": None,
        }


fig, ax1 = plt.subplots(figsize=(10.0, 4.5))
ax2 = ax1.twinx() if contrast is not None else None

colors = {
    "signal": "tab:blue",
    "reference": "tab:orange",
    "contrast": "tab:green",
    "fit": "black",
    "dip": "red",
}

lines = []
ax1.plot(x_mhz, signal, "-", linewidth=1.2, alpha=0.35, color=colors["signal"])
lines += ax1.plot(
    x_mhz,
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
    ax1.plot(x_mhz, reference, "-", linewidth=1.2, alpha=0.35, color=colors["reference"])
    lines += ax1.plot(
        x_mhz,
        reference,
        "o",
        markersize=6,
        linestyle="None",
        color=colors["reference"],
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="reference",
    )

fit_summary_text = None
if ax2 is not None:
    ax2.plot(x_mhz, contrast, "--", linewidth=1.0, alpha=0.4, color=colors["contrast"])
    lines += ax2.plot(
        x_mhz,
        contrast,
        "s",
        markersize=5.2,
        linestyle="None",
        color=colors["contrast"],
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="contrast ratio",
    )

    fit = _fit_podmr_dip(x_mhz, contrast)
    if fit is not None:
        dip_center_mhz = float(fit["params"][1])
        dip_y = float(np.interp(dip_center_mhz, x_mhz, contrast))

        if fit["fit_y"] is not None:
            lines += ax2.plot(
                fit["fit_x"],
                fit["fit_y"],
                "-",
                linewidth=2.0,
                color=colors["fit"],
                label="contrast fit",
            )

        lines += ax2.plot(
            [dip_center_mhz],
            [dip_y],
            "o",
            markersize=6,
            color=colors["dip"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            label="dip center",
        )

        ax2.annotate(
            f"{dip_center_mhz:.3f} MHz",
            xy=(dip_center_mhz, dip_y),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=colors["dip"],
        )

        ax2.text(
            0.02,
            0.98,
            f"f_dip = {dip_center_mhz:.3f} MHz",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        if fit["model"] == "laplace_dip":
            depth = float(fit["params"][0])
            width = float(fit["params"][2])
            baseline = float(fit["params"][3])
            fit_summary_text = (
                f"Fit (Laplace dip): center={dip_center_mhz:.3f} MHz, "
                f"depth={depth:.4g}, width={width:.4g} MHz, baseline={baseline:.4g}"
            )
            print(
                "[podmr] Contrast dip fit: "
                f"center={dip_center_mhz:.6f} MHz, depth={depth:.6g}, width={width:.6g} MHz, baseline={baseline:.6g}"
            )
        else:
            fit_summary_text = f"Fit fallback (argmin): center={dip_center_mhz:.3f} MHz"
            print(f"[podmr] Dip center (argmin fallback): {dip_center_mhz:.6f} MHz")

    ax2.set_ylabel("Contrast ratio")

ax1.set_xlabel("MW Frequency (MHz)")
ax1.set_ylabel("Counts/s")
ax1.set_title(
    f"PODMR  |  {timestamp}  |  "
    f"mw={config.mw_fMHz:.3f} MHz  |  gain={config.mw_gain}  |  "
    f"pi={config.mw_pi_tdds} tdds  |  "
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
print(f"[podmr] Plot saved -> {plot_path}")
