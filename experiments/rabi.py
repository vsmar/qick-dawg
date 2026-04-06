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

from qickdawg import RabiFineRes

from config import (
    load_config,
    build_nv_config,
    connect,
    save_experiment_hdf5,
)
from combine_utils import combine_chunk_hdf5_files
from plotting_utils import (
    extract_standard_traces,
    plot_debug_traces,
    plot_contrast_twin,
)
from experiment_helpers import (
    build_plot_metadata,
    build_standard_title,
    maybe_run_chunked_mode,
)

# =============================================================================
# EXPERIMENT PARAMETERS — edit these before each run
# =============================================================================

# Sweep bounds in nanoseconds.
# NVConfiguration handles conversion to ftsamp/treg companion units.
MW_DURATION_START_NS = 50     # ns
MW_DURATION_STOP_NS  = 6000    #2000    # ns
MW_DURATION_DELTA_NS = 50     # ns  (step size)

REPS          = 150000*2 #*3 # 4

RUN_MODE = "single"  # "single" or "chunked"
TARGET_TOTAL_REPS = 300_000
CHUNK_REPS = 50_000
ACQUIRE_PROGRESS = True
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = None  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition — set to "lower_dip", "upper_dip", or None to use config.yaml default.
TRANSITION    = None   # None = use calibration.default_transition

# Optional per-run overrides. If left None, values come from transition calibration.
OVERRIDE_FREQ_MHZ = None   # e.g. 1845.7
OVERRIDE_MW_GAIN  = 600   # e.g. 1200

GET_REFERENCE = True          # acquire reference readout with MW gain = 0
PLOT_USE_COUNTS_S = True
PLOT_DEBUG_RAW = True
PLOT_METADATA_POSITION = "top"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "rabi"


def _build_rabi_config(cfg: dict, reps: int):
    config = build_nv_config(cfg)

    active_transition = TRANSITION or cfg["calibration"]["default_transition"]
    t = cfg["calibration"][active_transition]

    # Resolve run parameters with precedence:
    # explicit file override -> selected/default transition values.
    config.mw_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["mw_fMHz"]
    config.mw_gain = OVERRIDE_MW_GAIN if OVERRIDE_MW_GAIN is not None else t["mw_gain"]

    config.reps = int(reps)
    config.get_reference = bool(GET_REFERENCE)

    config.add_linear_sweep(
        "mw_duration",
        "ftns",
        start=MW_DURATION_START_NS,
        stop=MW_DURATION_STOP_NS,
        delta=MW_DURATION_DELTA_NS,
    )
    return config, active_transition

# =============================================================================
# Setup
# =============================================================================

cfg    = load_config()
connect(cfg)

config, active_transition = _build_rabi_config(cfg, REPS)
print(
    f"[rabi] Sweep: {MW_DURATION_START_NS:.3f} -> {MW_DURATION_STOP_NS:.3f} ns "
    f"(delta {MW_DURATION_DELTA_NS:.3f} ns)"
)
print(f"[rabi] Active transition: {active_transition} | freq={config.mw_fMHz} MHz | gain={config.mw_gain}")


def _build_chunk_config_context(reps: int):
    chunk_cfg, chunk_transition = _build_rabi_config(cfg, reps)
    return (
        chunk_cfg,
        {
            "transition": chunk_transition,
            "mw_duration_start_ns": float(MW_DURATION_START_NS),
            "mw_duration_stop_ns": float(MW_DURATION_STOP_NS),
            "mw_duration_delta_ns": float(MW_DURATION_DELTA_NS),
        },
    )


if maybe_run_chunked_mode(
    run_mode=RUN_MODE,
    program_class=RabiFineRes,
    cfg_dict=cfg,
    build_config_for_chunk=_build_chunk_config_context,
    output_dir=OUTPUT_DIR,
    experiment_name="rabi_fine_res",
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
    piezo_initial_position_um=PIEZO_INITIAL_POSITION_UM,
    combine_filename="rabi_fine_res_combined.h5",
    combine_fn=combine_chunk_hdf5_files,
):
    raise SystemExit(0)

if RUN_MODE != "single":
    raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

# =============================================================================
# Acquire
# =============================================================================

prog = RabiFineRes(config)
data = prog.acquire(progress=bool(ACQUIRE_PROGRESS))

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    RabiFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="rabi_fine_res",
)
run_id = out_path.stem

print(f"[rabi] Saved → {out_path}")

# =============================================================================
# Plot
# =============================================================================

# Build plotting array from the canonical analyzed sweep axis.
if not hasattr(data, "mw_duration_ftns"):
    raise ValueError("RabiFineRes output missing expected sweep axis mw_duration_ftns.")
x_ns = np.asarray(data.mw_duration_ftns, dtype=float)
traces = extract_standard_traces(data, x_axis=x_ns, use_counts_s=PLOT_USE_COUNTS_S)
signal = traces["signal1"]
reference = traces["signal2"]
contrast = traces["contrast"]

if signal is None:
    raise ValueError("RabiFineRes output missing signal1/signal1_cts_s field.")


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

metadata = build_plot_metadata(
    program_class=RabiFineRes,
    config_obj=config,
    base_metadata={
        "run_id": run_id,
        "mw_MHz": f"{config.mw_fMHz:.3f}",
        "gain": config.mw_gain,
        "sequence": "mw_pulse - readout",
        "reps": config.reps,
        "laser_mW": cfg["optics"]["excitation_laser_power_mW"],
        "units": "cts/s" if PLOT_USE_COUNTS_S else "raw",
    },
)

if PLOT_DEBUG_RAW:
    plot_debug_traces(
        x_ns,
        traces,
        x_label="MW Pulse Duration (ns)",
        y_label="Counts/s" if PLOT_USE_COUNTS_S else "Counts",
        title=build_standard_title(
            experiment_label="Rabi Debug Raw",
            sequence_label="mw_pulse - readout",
            run_id=run_id,
        ),
        metadata=metadata,
        metadata_position=PLOT_METADATA_POSITION,
    )

fit_summary_text = None
fit_x = None
fit_y = None
if contrast is not None:
    fit = _fit_rabi_contrast(x_ns, contrast)
    if fit is not None:
        fit_x = fit["fit_x"]
        fit_y = fit["fit_y"]
        period_fit_ns = float(fit["params"][1])
        pi2_ns = period_fit_ns / 4.0
        rabi_freq_mhz = 1000.0 / period_fit_ns

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
    else:
        print("[rabi] Contrast fit skipped: fit did not converge.")
        fit_summary_text = "Fit: did not converge."

fig, ax_left, _, _ = plot_contrast_twin(
    x_ns,
    traces,
    x_label="MW Pulse Duration (ns)",
    title=build_standard_title(
        experiment_label="Rabi",
        sequence_label="mw_pulse - readout",
        run_id=run_id,
    ),
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
print(f"[rabi] Plot saved → {plot_path}")