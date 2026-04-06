"""
t1.py - T1 Time-Delay Sweep Experiment
======================================
Runs a T1 delay sweep using T1FineRes, saves data + full config to HDF5, and
plots MW on, MW off, and MW on-minus-off with an exponential fit for T1.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qickdawg.nvtestsuite.t1_fine_res import T1FineRes

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
    format_metadata_lines,
)
from experiment_helpers import (
    build_plot_metadata,
    build_standard_title,
    maybe_run_chunked_mode,
)

# =============================================================================
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

# T1 delay sweep bounds in ns (time units, not fine-time units).
T1_DELAY_START_TNS = 1_000.0
T1_DELAY_END_TNS = 3_000_000.0
SCALING_MODE = "exponential"  # "linear" or "exponential"

T1_DELAY_DELTA_TNS = 400.0  # Ignored when SCALING_MODE is "exponential"
SCALING_FACTOR = "5/4"  # Ignored when SCALING_MODE is "linear"

NSWEEP_POINTS = 120  # Used for linear sweeps.

REPS = 30_000

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 210_000
CHUNK_REPS = 30_000
ACQUIRE_PROGRESS = True
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = (-0.286, 1.159, -3.8239)  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition - set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI_FTSAMP = None
OVERRIDE_MW_PI_NS = None

GET_REFERENCE = True
PLOT_USE_COUNTS_S = True
PLOT_DEBUG_RAW = False
PLOT_METADATA_POSITION = "bottom"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "t1"


def _build_t1_config(cfg: dict, reps: int):
    config = build_nv_config(cfg)

    active_transition = TRANSITION or cfg["calibration"]["default_transition"]
    t = cfg["calibration"][active_transition]

    config.mw_gain = OVERRIDE_MW_GAIN if OVERRIDE_MW_GAIN is not None else t["mw_gain"]

    if OVERRIDE_MW_PI_FTSAMP is not None and OVERRIDE_MW_PI_NS is not None:
        raise ValueError("Set only one of OVERRIDE_MW_PI_FTSAMP or OVERRIDE_MW_PI_NS.")

    if OVERRIDE_MW_PI_FTSAMP is not None:
        config.mw_pi_ftsamp = int(OVERRIDE_MW_PI_FTSAMP)
        pi_source = f"OVERRIDE_MW_PI_FTSAMP={config.mw_pi_ftsamp}"
    elif OVERRIDE_MW_PI_NS is not None:
        config.mw_pi_ftns = float(OVERRIDE_MW_PI_NS)
        pi_source = f"OVERRIDE_MW_PI_NS={OVERRIDE_MW_PI_NS} ns"
    else:
        if t.get("mw_pi_ftsamp") is None:
            raise ValueError(
                "No calibration pi pulse found for this transition. "
                "Set OVERRIDE_MW_PI_FTSAMP (or OVERRIDE_MW_PI_NS)."
            )
        config.mw_pi_ftsamp = int(t["mw_pi_ftsamp"])
        pi_source = f"calibration.{active_transition}.mw_pi_ftsamp={config.mw_pi_ftsamp}"

    config.reps = int(reps)
    config.get_reference = bool(GET_REFERENCE)

    # Keep t1_delay delta/scaling populated for downstream code paths.
    config.t1_delay_delta_tns = float(T1_DELAY_DELTA_TNS)
    config.scaling_factor = SCALING_FACTOR

    if SCALING_MODE == "linear":
        config.add_linear_sweep(
            "t1_delay",
            "tns",
            start=float(T1_DELAY_START_TNS),
            stop=float(T1_DELAY_END_TNS),
            delta=float(T1_DELAY_DELTA_TNS),
        )
    elif SCALING_MODE == "exponential":
        config.add_exponential_sweep(
            "t1_delay",
            "tns",
            start=float(T1_DELAY_START_TNS),
            stop=float(T1_DELAY_END_TNS),
            scaling_factor=SCALING_FACTOR,
        )
    else:
        raise ValueError("SCALING_MODE must be 'linear' or 'exponential'.")

    return config, active_transition, pi_source

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, active_transition, pi_source = _build_t1_config(cfg, REPS)

if SCALING_MODE == "linear":
    print(
        f"[t1] Sweep: {config.t1_delay_start_tns:.3f} -> {config.t1_delay_end_tns:.3f} ns "
        f"({SCALING_MODE}, configured delta={config.t1_delay_delta_tns:.3f} ns, "
        f"points={config.nsweep_points})"
    )
else:
    print(
        f"[t1] Sweep: {config.t1_delay_start_tns:.3f} -> {config.t1_delay_end_tns:.3f} ns "
        f"({SCALING_MODE}, scaling_factor={SCALING_FACTOR}, points={config.nsweep_points})"
    )
print(f"[t1] Active transition: {active_transition} | mw_gain={config.mw_gain}")
print(f"[t1] pi pulse: {pi_source}")


def _build_chunk_config_context(reps: int):
    chunk_cfg, chunk_transition, chunk_pi_source = _build_t1_config(cfg, reps)
    return (
        chunk_cfg,
        {
            "transition": chunk_transition,
            "pi_source": chunk_pi_source,
            "scaling_mode": SCALING_MODE,
            "t1_delay_start_tns": float(T1_DELAY_START_TNS),
            "t1_delay_end_tns": float(T1_DELAY_END_TNS),
            "t1_delay_delta_tns": float(T1_DELAY_DELTA_TNS),
            "scaling_factor": SCALING_FACTOR,
        },
    )


if maybe_run_chunked_mode(
    run_mode=RUN_MODE,
    program_class=T1FineRes,
    cfg_dict=cfg,
    build_config_for_chunk=_build_chunk_config_context,
    output_dir=OUTPUT_DIR,
    experiment_name="t1_fine_res",
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
    piezo_initial_position_um=PIEZO_INITIAL_POSITION_UM,
    combine_filename="t1_fine_res_combined.h5",
    combine_fn=combine_chunk_hdf5_files,
):
    raise SystemExit(0)

if RUN_MODE != "single":
    raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

# =============================================================================
# Acquire
# =============================================================================

prog = T1FineRes(config)
data = prog.acquire(progress=bool(ACQUIRE_PROGRESS))

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    T1FineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="t1_fine_res",
)
run_id = out_path.stem

print(f"[t1] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

def _extract_t1_axis_us(acquired_data, active_config) -> np.ndarray:
    if hasattr(acquired_data, "t1_delay_tus"):
        return np.asarray(acquired_data.t1_delay_tus, dtype=float)
    if hasattr(acquired_data, "t1_delay_tns"):
        return np.asarray(acquired_data.t1_delay_tns, dtype=float) / 1000.0
    if hasattr(acquired_data, "t1_delay_treg"):
        treg = np.asarray(acquired_data.t1_delay_treg, dtype=float)
        return np.asarray([active_config.soccfg.cycles2us(v) for v in treg], dtype=float)
    raise ValueError("T1 output missing expected sweep axis (t1_delay_tus/tns/treg).")


x_us = _extract_t1_axis_us(data, config)
traces = extract_standard_traces(data, x_axis=x_us, use_counts_s=PLOT_USE_COUNTS_S)
signal_on = traces.get("signal1")
signal_off = traces.get("signal2")

if signal_on is None or signal_off is None:
    raise ValueError("T1 output missing signal1/signal2 channels for MW on/off plotting.")

diff_signal = signal_on - signal_off


def _exp_decay(x_us_arr, amplitude, t1_us, offset):
    x_us_arr = np.asarray(x_us_arr, dtype=float)
    safe_t1 = np.clip(float(t1_us), 1e-9, None)
    return offset + amplitude * np.exp(-np.clip(x_us_arr, 0.0, None) / safe_t1)


def _fit_t1_difference(x_us_arr, y):
    x_us_arr = np.asarray(x_us_arr, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    finite = np.isfinite(x_us_arr) & np.isfinite(y)
    x_us_arr = x_us_arr[finite]
    y = y[finite]
    if len(x_us_arr) < 5:
        return None

    order = np.argsort(x_us_arr)
    x_us_arr = x_us_arr[order]
    y = y[order]

    span = float(np.max(x_us_arr) - np.min(x_us_arr))
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
            _exp_decay,
            x_us_arr,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=60000,
        )
        fit_x = np.linspace(float(np.min(x_us_arr)), float(np.max(x_us_arr)), max(500, len(x_us_arr) * 20))
        fit_sigma = np.sqrt(np.diag(cov)) if cov is not None else np.full(len(params), np.nan)
        return {
            "params": params,
            "sigma": fit_sigma,
            "fit_x": fit_x,
            "fit_y": _exp_decay(fit_x, *params),
            "t1_us": float(params[1]),
            "t1_sigma_us": float(fit_sigma[1]) if len(fit_sigma) > 1 else np.nan,
        }
    except (RuntimeError, ValueError):
        return None


fit_summary_text = None
fit_x = None
fit_y = None
t1_result_text = None

fit = _fit_t1_difference(x_us, diff_signal)
if fit is not None:
    fit_x = fit["fit_x"]
    fit_y = fit["fit_y"]
    t1_us = float(fit["t1_us"])
    t1_sigma_us = float(fit["t1_sigma_us"])
    amp = float(fit["params"][0])
    offset = float(fit["params"][2])
    print(f"[t1] Difference fit: A={amp:.6g}, T1={t1_us:.3f} us, C={offset:.6g}")
    if np.isfinite(t1_sigma_us):
        t1_result_text = f"T1 = {t1_us:.3f} +/- {t1_sigma_us:.3f} us"
    else:
        t1_result_text = f"T1 = {t1_us:.3f} us"
    fit_summary_text = f"Fit: y = C + A*exp(-t/T1), A={amp:.4g}, C={offset:.4g}"
else:
    print("[t1] Difference fit skipped: fit did not converge.")
    fit_summary_text = "Fit: did not converge."
    t1_result_text = "T1: fit did not converge"

metadata = build_plot_metadata(
    program_class=T1FineRes,
    config_obj=config,
    base_metadata={
        "run_id": run_id,
        "gain": config.mw_gain,
        "pi_ftsamp": config.mw_pi_ftsamp,
        "pi_ftns": f"{config.mw_pi_ftns:.2f}",
        "sequence": "pi - delay - readout",
        "reps": config.reps,
        "laser_mW": cfg["optics"]["excitation_laser_power_mW"],
        "units": "cts/s" if PLOT_USE_COUNTS_S else "raw",
    },
)

if PLOT_DEBUG_RAW:
    plot_debug_traces(
        x_us,
        traces,
        x_label=r"$t_1$ delay (us)",
        y_label="Counts/s" if PLOT_USE_COUNTS_S else "Counts",
        title=build_standard_title(
            experiment_label="T1 Debug Raw",
            sequence_label="pi - delay - readout",
            run_id=run_id,
        ),
        metadata=metadata,
        metadata_position=PLOT_METADATA_POSITION,
    )

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)

ax_top.plot(x_us, signal_on, "o-", markersize=4.0, linewidth=1.2, alpha=0.9, color="tab:blue", label="MW on")
ax_top.plot(x_us, signal_off, "o-", markersize=4.0, linewidth=1.2, alpha=0.9, color="tab:orange", label="MW off")
ax_top.set_ylabel("Counts/s" if PLOT_USE_COUNTS_S else "Counts")
ax_top.set_title(
    build_standard_title(
        experiment_label="T1",
        sequence_label="pi - delay - readout",
        run_id=run_id,
        suffix="MW on/off and difference",
    )
)
ax_top.grid(alpha=0.25)
ax_top.legend(loc="best", framealpha=0.95)

ax_bottom.plot(
    x_us,
    diff_signal,
    "s--",
    markersize=4.2,
    linewidth=1.2,
    alpha=0.95,
    color="tab:green",
    label="MW on - MW off",
)
if fit_x is not None and fit_y is not None:
    ax_bottom.plot(fit_x, fit_y, "-", linewidth=2.0, color="black", alpha=0.95, label="exp fit")

ax_bottom.set_xlabel(r"$t_1$ delay (us)")
ax_bottom.set_ylabel("MW on - MW off")
ax_bottom.grid(alpha=0.25)
ax_bottom.legend(loc="best", framealpha=0.95)

ax_bottom.text(
    0.02,
    0.98,
    f"{t1_result_text}\n{fit_summary_text}",
    transform=ax_bottom.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray"),
)

metadata_line = format_metadata_lines(metadata)
fig.text(0.5, 0.01, metadata_line, ha="center", va="bottom", fontsize=8.5)
fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[t1] Plot saved -> {plot_path}")
