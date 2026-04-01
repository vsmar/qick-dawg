"""
ramsey.py - Ramsey Fine-Resolution Experiment
=============================================
Runs a Ramsey fine-resolution sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qickdawg.nvtestsuite.ramsey_fine_res import RamseyFineRes

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
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

# Sweep bounds in fine-time nanoseconds (ftns).
TAU_START_FTNS = 200.0
TAU_END_FTNS = 15_000.0

SCALING_MODE = "linear" # "linear" or "exponential"
TAU_DELTA_FTNS = 100.0   #!!! Ignored when scaling_mode is 'exponential'
SCALING_FACTOR = "5/4"  #!!! Ignored when scaling_mode is 'linear'

REPS = 80_000

# Transition - set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

FREQ_DETUNE_MHZ = None

GET_REFERENCE = True
PLOT_USE_COUNTS_S = True
PLOT_DEBUG_RAW = False
PLOT_METADATA_POSITION = "bottom"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ramsey"

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)

config = build_nv_config(cfg)

active_transition = TRANSITION or cfg["calibration"]["default_transition"]
t = cfg["calibration"][active_transition]

config.mw_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["mw_fMHz"]
config.mw_fMHz += FREQ_DETUNE_MHZ if FREQ_DETUNE_MHZ is not None else 0.0
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
config.scaling_factor = SCALING_FACTOR

if SCALING_MODE == "linear":
    config.add_linear_sweep(
        "tau",
        "ftns",
        start=TAU_START_FTNS,
        stop=TAU_END_FTNS,
        delta=TAU_DELTA_FTNS,
    )
elif SCALING_MODE == "exponential":
    config.add_exponential_sweep(
        "tau",
        "ftns",
        start=TAU_START_FTNS,
        stop=TAU_END_FTNS,
        scaling_factor=SCALING_FACTOR,
    )
else:
    raise ValueError("SCALING_MODE must be 'linear' or 'exponential'.")

if SCALING_MODE == "linear":
    print(
        f"[ramsey] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
        f"({SCALING_MODE}, configured delta={config.tau_delta_ftns:.3f})"
    )
else:
    print(
        f"[ramsey] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
        f"({SCALING_MODE}, scaling_factor={SCALING_FACTOR})"
    )
print(
    f"[ramsey] Active transition: {active_transition} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}"
)
print(f"[ramsey] mw_pi2_ftsamp={config.mw_pi2_ftsamp}, reps={config.reps}")

# =============================================================================
# Acquire
# =============================================================================

prog = RamseyFineRes(config)
data = prog.acquire(progress=True)

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    RamseyFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="ramsey_fine_res",
)
run_id = out_path.stem

print(f"[ramsey] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

if hasattr(data, "tau_ftus"):
    x_axis_us = np.asarray(data.tau_ftus, dtype=float)
elif hasattr(data, "tau_ftns"):
    # Convert fine-time ns sweep axis into us for fit readability.
    x_axis_us = np.asarray(data.tau_ftns, dtype=float) / 1000.0
else:
    raise ValueError("Ramsey output missing expected sweep axis (tau_ftus or tau_ftns).")


def _ramsey_damped_cosine(
    x_us: np.ndarray,
    amplitude: float,
    freq_mhz: float,
    phase_rad: float,
    t2s_us: float,
    offset: float,
) -> np.ndarray:
    x_us = np.asarray(x_us, dtype=float)
    safe_t2 = np.clip(t2s_us, 1e-9, None)
    envelope = np.exp(-np.clip(x_us, 0.0, None) / safe_t2)
    return offset + amplitude * np.cos(2.0 * np.pi * freq_mhz * x_us + phase_rad) * envelope


def _ramsey_exp_decay(
    x_us: np.ndarray,
    amplitude: float,
    t2s_us: float,
    offset: float,
) -> np.ndarray:
    x_us = np.asarray(x_us, dtype=float)
    safe_t2 = np.clip(t2s_us, 1e-9, None)
    return offset + amplitude * np.exp(-np.clip(x_us, 0.0, None) / safe_t2)


def _fit_ramsey_contrast(
    x_us: np.ndarray,
    y: np.ndarray,
    detune_mhz: float | None,
) -> dict[str, object] | None:
    x_us = np.asarray(x_us, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    finite = np.isfinite(x_us) & np.isfinite(y)
    x_us = x_us[finite]
    y = y[finite]
    if len(x_us) < 6:
        return None

    order = np.argsort(x_us)
    x_us = x_us[order]
    y = y[order]

    x_span = float(np.max(x_us) - np.min(x_us))
    if x_span <= 0:
        return None

    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_range = max(y_hi - y_lo, float(np.std(y)) * 2.0, 1e-6)
    amp_lim = max(3.0 * y_range, 1e-3)
    offset0 = float(np.median(y))
    amp0 = float(0.5 * (np.percentile(y, 95) - np.percentile(y, 5)))
    if np.isclose(amp0, 0.0):
        amp0 = 0.25 * y_range

    t2_min = max(x_span / 200.0, 1e-3)
    t2_max = max(x_span * 20.0, 10.0)
    t2_0 = max(x_span / 2.0, t2_min)

    # Baseline non-oscillatory model used as conservative fallback.
    exp_fit = None
    try:
        p0_exp = [amp0, t2_0, offset0]
        bounds_exp = (
            [-amp_lim, t2_min, y_lo - y_range],
            [amp_lim, t2_max, y_hi + y_range],
        )
        params_exp, _ = curve_fit(
            _ramsey_exp_decay,
            x_us,
            y,
            p0=p0_exp,
            bounds=bounds_exp,
            maxfev=80000,
        )
        y_exp = _ramsey_exp_decay(x_us, *params_exp)
        rss_exp = float(np.sum((y - y_exp) ** 2))
        exp_fit = {"params": params_exp, "rss": rss_exp}
    except (RuntimeError, ValueError):
        pass

    # Frequency guess: trust absolute detune when provided.
    if detune_mhz is not None and abs(float(detune_mhz)) > 0:
        freq_guess_mhz = abs(float(detune_mhz))
    else:
        dt = np.median(np.diff(x_us))
        if dt <= 0:
            freq_guess_mhz = 1.0 / x_span
        else:
            spectrum = np.abs(np.fft.rfft(y - np.mean(y)))
            freqs = np.fft.rfftfreq(len(y), d=float(dt))
            if len(spectrum) > 0:
                spectrum[0] = 0.0
            peak_idx = int(np.argmax(spectrum)) if len(spectrum) > 0 else 0
            freq_guess_mhz = float(freqs[peak_idx]) if peak_idx < len(freqs) else 0.0
            if freq_guess_mhz <= 0:
                freq_guess_mhz = 1.0 / x_span

    freq_min = 0.0
    freq_max = max(20.0 * freq_guess_mhz, 10.0 / x_span)
    p0_cos = [
        amp0,
        max(freq_guess_mhz, 1.0 / x_span),
        0.0,
        float(exp_fit["params"][1]) if exp_fit is not None else t2_0,
        float(exp_fit["params"][2]) if exp_fit is not None else offset0,
    ]
    bounds_cos = (
        [-amp_lim, freq_min, -2.0 * np.pi, t2_min, y_lo - y_range],
        [amp_lim, freq_max, 2.0 * np.pi, t2_max, y_hi + y_range],
    )

    cos_fit = None
    try:
        params_cos, _ = curve_fit(
            _ramsey_damped_cosine,
            x_us,
            y,
            p0=p0_cos,
            bounds=bounds_cos,
            maxfev=120000,
        )
        y_cos = _ramsey_damped_cosine(x_us, *params_cos)
        rss_cos = float(np.sum((y - y_cos) ** 2))
        cos_fit = {"params": params_cos, "rss": rss_cos}
    except (RuntimeError, ValueError):
        pass

    # Model selection: enforce detune-aware oscillatory fit when detune is explicit;
    # otherwise only accept sinusoid when it clearly outperforms simple decay.
    selected_model = None
    if detune_mhz is not None and abs(float(detune_mhz)) > 0:
        if cos_fit is not None:
            selected_model = "damped_cosine"
        elif exp_fit is not None:
            selected_model = "exp_decay"
    else:
        if cos_fit is not None and exp_fit is not None:
            fitted_freq = float(cos_fit["params"][1])
            min_meaningful_freq = 1.0 / x_span
            improves_over_exp = cos_fit["rss"] < 0.8 * exp_fit["rss"]
            oscillatory = fitted_freq >= min_meaningful_freq
            selected_model = "damped_cosine" if (improves_over_exp and oscillatory) else "exp_decay"
        elif cos_fit is not None:
            selected_model = "damped_cosine"
        elif exp_fit is not None:
            selected_model = "exp_decay"

    if selected_model is None:
        return None

    fit_x = np.linspace(float(np.min(x_us)), float(np.max(x_us)), max(500, len(x_us) * 20))
    if selected_model == "damped_cosine":
        params = np.asarray(cos_fit["params"], dtype=float)
        fit_y = _ramsey_damped_cosine(fit_x, *params)
        return {
            "model": selected_model,
            "params": params,
            "fit_x": fit_x,
            "fit_y": fit_y,
            "t2s_us": float(params[3]),
            "freq_mhz": float(params[1]),
            "offset": float(params[4]),
            "amplitude": float(params[0]),
            "phase_rad": float(params[2]),
        }

    params = np.asarray(exp_fit["params"], dtype=float)
    fit_y = _ramsey_exp_decay(fit_x, *params)
    return {
        "model": selected_model,
        "params": params,
        "fit_x": fit_x,
        "fit_y": fit_y,
        "t2s_us": float(params[1]),
        "freq_mhz": 0.0,
        "offset": float(params[2]),
        "amplitude": float(params[0]),
    }

traces = extract_standard_traces(data, x_axis=x_axis_us, use_counts_s=PLOT_USE_COUNTS_S)
contrast = traces.get("contrast")

fit_summary_text = None
fit_x = None
fit_y = None
if contrast is not None:
    fit = _fit_ramsey_contrast(x_axis_us, contrast, FREQ_DETUNE_MHZ)
    if fit is not None:
        fit_x = fit["fit_x"]
        fit_y = fit["fit_y"]
        if fit["model"] == "damped_cosine":
            print(
                "[ramsey] Contrast fit (damped cosine): "
                f"A={fit['amplitude']:.6g}, f={fit['freq_mhz']:.6g} MHz, "
                f"phi={fit['phase_rad']:.6g} rad, T2*={fit['t2s_us']:.3f} us, "
                f"C={fit['offset']:.6g}"
            )
            fit_summary_text = (
                f"Fit (damped cosine): A={fit['amplitude']:.4g}, "
                f"f={fit['freq_mhz']:.4g} MHz, T2*={fit['t2s_us']:.3f} us, "
                f"C={fit['offset']:.4g}"
            )
        else:
            print(
                "[ramsey] Contrast fit (exp decay fallback): "
                f"A={fit['amplitude']:.6g}, T2*={fit['t2s_us']:.3f} us, C={fit['offset']:.6g}"
            )
            fit_summary_text = (
                f"Fit (exp decay): A={fit['amplitude']:.4g}, "
                f"T2*={fit['t2s_us']:.3f} us, C={fit['offset']:.4g}"
            )
    else:
        print("[ramsey] Contrast fit skipped: fit did not converge.")
        fit_summary_text = "Fit: did not converge."

metadata = {
    "run_id": run_id,
    "mw_MHz": f"{config.mw_fMHz:.3f}",
    "detune_MHz": FREQ_DETUNE_MHZ,
    "gain": config.mw_gain,
    "pi2_ftsamp": config.mw_pi2_ftsamp,
    "pi2_ftns": f"{config.mw_pi2_ftns:.2f}",
    "sequence": "pi/2 - tau - pi/2",
    "reps": config.reps,
    "laser_mW": cfg["optics"]["excitation_laser_power_mW"],
    "units": "cts/s" if PLOT_USE_COUNTS_S else "raw",
}

if PLOT_DEBUG_RAW:
    plot_debug_traces(
        x_axis_us,
        traces,
        x_label=r"$\tau$ (us)",
        y_label="Counts/s" if PLOT_USE_COUNTS_S else "Counts",
        title=f"Ramsey Debug Raw | $\\pi/2 - \\tau - \\pi/2$ | {timestamp}",
        metadata=metadata,
        metadata_position=PLOT_METADATA_POSITION,
    )

fig, ax_left, _, _ = plot_contrast_twin(
    x_axis_us,
    traces,
    x_label=r"$\tau$ (us)",
    title=f"Ramsey | $\\pi/2 - \\tau - \\pi/2$ | {timestamp}",
    metadata=metadata,
    metadata_position=PLOT_METADATA_POSITION,
    fit_x=fit_x,
    fit_y=fit_y,
    fit_label="contrast fit",
)

if fit_summary_text is not None:
    ax_left.text(
        0.02,
        0.98,
        fit_summary_text,
        transform=ax_left.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[ramsey] Plot saved -> {plot_path}")