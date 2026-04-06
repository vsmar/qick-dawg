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

from qickdawg import PODMRFineRes

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

# Sweep bounds in MHz.
ODMR_START_MHZ = 1844.5
ODMR_STOP_MHZ = 1847.5
ODMR_DELTA_MHZ = 0.05

REPS = 200000

RUN_MODE = "single"  # "single" or "chunked"
TARGET_TOTAL_REPS = 600_000
CHUNK_REPS = 200_000
ACQUIRE_PROGRESS = True
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = None  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_MW_GAIN = 1200

# Set either ftsamp directly, or ns (which will be converted to ftsamp).
OVERRIDE_MW_PI_FTSAMP = None
OVERRIDE_MW_PI_NS = 950

GET_REFERENCE = True
PLOT_USE_COUNTS_S = True
PLOT_DEBUG_RAW = True
PLOT_METADATA_POSITION = "bottom"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "podmr"


def _build_podmr_config(cfg: dict, reps: int):
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
    config.add_linear_sweep("mw", "fMHz", start=ODMR_START_MHZ, stop=ODMR_STOP_MHZ, delta=ODMR_DELTA_MHZ)
    return config, active_transition, pi_source

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, active_transition, pi_source = _build_podmr_config(cfg, REPS)

print(
    f"[podmr] Sweep: {ODMR_START_MHZ:.3f} -> {ODMR_STOP_MHZ:.3f} MHz "
    f"(delta {ODMR_DELTA_MHZ:.3f} MHz)"
)
print(f"[podmr] Active transition: {active_transition} | mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}")
print(f"[podmr] pi pulse: {pi_source}")


def _build_chunk_config_context(reps: int):
    chunk_cfg, chunk_transition, chunk_pi_source = _build_podmr_config(cfg, reps)
    return (
        chunk_cfg,
        {
            "transition": chunk_transition,
            "pi_source": chunk_pi_source,
            "odmr_start_mhz": float(ODMR_START_MHZ),
            "odmr_stop_mhz": float(ODMR_STOP_MHZ),
            "odmr_delta_mhz": float(ODMR_DELTA_MHZ),
        },
    )


if maybe_run_chunked_mode(
    run_mode=RUN_MODE,
    program_class=PODMRFineRes,
    cfg_dict=cfg,
    build_config_for_chunk=_build_chunk_config_context,
    output_dir=OUTPUT_DIR,
    experiment_name="podmr_fine_res",
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
    piezo_initial_position_um=PIEZO_INITIAL_POSITION_UM,
    combine_filename="podmr_fine_res_combined.h5",
    combine_fn=combine_chunk_hdf5_files,
):
    raise SystemExit(0)

if RUN_MODE != "single":
    raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

# =============================================================================
# Acquire
# =============================================================================

prog = PODMRFineRes(config)
data = prog.acquire(progress=bool(ACQUIRE_PROGRESS))

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    PODMRFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="podmr_fine_res",
)
run_id = out_path.stem

print(f"[podmr] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

if not hasattr(data, "mw_fMHz"):
    raise ValueError("PODMRFineRes output missing expected sweep axis mw_fMHz.")
x_mhz = np.asarray(data.mw_fMHz, dtype=float)

traces = extract_standard_traces(data, x_axis=x_mhz, use_counts_s=PLOT_USE_COUNTS_S)
signal = traces["signal1"]
reference = traces["signal2"]
contrast = traces["contrast"]

if signal is None:
    raise ValueError("PODMRFineRes output missing signal1/signal1_cts_s field.")


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


fit_summary_text = None
fit_x = None
fit_y = None
dip_center_mhz = None
dip_y = None

if contrast is not None:
    fit = _fit_podmr_dip(x_mhz, contrast)
    if fit is not None:
        dip_center_mhz = float(fit["params"][1])
        dip_y = float(np.interp(dip_center_mhz, x_mhz, contrast))
        fit_x = fit["fit_x"]
        fit_y = fit["fit_y"]

        if fit["model"] == "laplace_dip":
            depth = float(fit["params"][0])
            width = float(fit["params"][2])
            baseline = float(fit["params"][3])
            fit_summary_text = (
                f"Laplace dip: center={dip_center_mhz:.3f} MHz, "
                f"depth={depth:.4g}, width={width:.4g} MHz, baseline={baseline:.4g}"
            )
            print(
                "[podmr] Contrast dip fit: "
                f"center={dip_center_mhz:.6f} MHz, depth={depth:.6g}, width={width:.6g} MHz, baseline={baseline:.6g}"
            )
        else:
            fit_summary_text = f"Argmin fallback dip center={dip_center_mhz:.3f} MHz"
            print(f"[podmr] Dip center (argmin fallback): {dip_center_mhz:.6f} MHz")

metadata = build_plot_metadata(
    program_class=PODMRFineRes,
    config_obj=config,
    base_metadata={
        "run_id": run_id,
        "mw_MHz": f"{config.mw_fMHz:.3f}",
        "gain": config.mw_gain,
        "pi_ftsamp": config.mw_pi_ftsamp,
        "mw_gain": config.mw_gain,
        "sequence": "laser - pi - readout",
        "reps": config.reps,
        "laser_mW": cfg["optics"]["excitation_laser_power_mW"],
        "units": "cts/s" if PLOT_USE_COUNTS_S else "raw",
    },
)

if PLOT_DEBUG_RAW:
    plot_debug_traces(
        x_mhz,
        traces,
        x_label="MW Frequency (MHz)",
        y_label="Counts/s" if PLOT_USE_COUNTS_S else "Counts",
        title=build_standard_title(
            experiment_label="PODMR Debug Raw",
            sequence_label="laser - pi - readout",
            run_id=run_id,
        ),
        metadata=metadata,
        metadata_position=PLOT_METADATA_POSITION,
    )

fig, ax_left, _, _ = plot_contrast_twin(
    x_mhz,
    traces,
    x_label="MW Frequency (MHz)",
    title=build_standard_title(
        experiment_label="PODMR",
        sequence_label="laser - pi - readout",
        run_id=run_id,
    ),
    metadata=metadata,
    metadata_position=PLOT_METADATA_POSITION,
    fit_x=fit_x,
    fit_y=fit_y,
    fit_label="contrast fit",
)

if dip_center_mhz is not None and dip_y is not None:
    ax_left.plot(
        [dip_center_mhz],
        [dip_y],
        "o",
        markersize=6,
        color="red",
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="dip center",
    )
    ax_left.annotate(
        f"{dip_center_mhz:.3f} MHz",
        xy=(dip_center_mhz, dip_y),
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="red",
    )
    handles, labels = ax_left.get_legend_handles_labels()
    ax_left.legend(handles, labels, loc="best", framealpha=0.95)

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
print(f"[podmr] Plot saved -> {plot_path}")
