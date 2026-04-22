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

from qickdawg.finetimingsuite.cpmg_xy_fine_res import CPMGXYFineRes

from config import (
    load_config,
    build_nv_config,
    connect,
    save_experiment_hdf5,
)
from combine_utils import combine_chunk_hdf5_files
from experiment_helpers import (
    build_plot_metadata,
    build_standard_title,
    maybe_run_chunked_mode,
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
TAU_STOP_FTNS = 9_000.0
TAU_DELTA_FTNS = 5

N_CPMG = 32  # 128
REPS = 2000

RUN_MODE = "single"  # "single" or "chunked"
TARGET_TOTAL_REPS = 100_000
CHUNK_REPS = 2_000
ACQUIRE_PROGRESS = True
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = None  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

GET_REFERENCE = False # True
PLOT_USE_COUNTS_S = True
PLOT_DEBUG_RAW = False
PLOT_METADATA_POSITION = "bottom"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cpmg"


def _build_cpmg_config(cfg: dict, reps: int):
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

    config.n_cpmg = int(N_CPMG)
    config.reps = int(reps)
    config.get_reference = bool(GET_REFERENCE)

    config.add_linear_sweep(
        "tau",
        "ftns",
        start=TAU_START_FTNS,
        stop=TAU_STOP_FTNS,
        delta=TAU_DELTA_FTNS,
    )

    return config, active_transition

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, active_transition = _build_cpmg_config(cfg, REPS)


def _build_chunk_config_context(reps: int):
    chunk_cfg, chunk_transition = _build_cpmg_config(cfg, reps)
    return (
        chunk_cfg,
        {
            "transition": chunk_transition,
            "n_cpmg": int(N_CPMG),
            "tau_start_ftns": float(TAU_START_FTNS),
            "tau_stop_ftns": float(TAU_STOP_FTNS),
            "tau_delta_ftns": float(TAU_DELTA_FTNS),
        },
    )

print(
    f"[cpmg] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
    f"(delta {config.tau_delta_ftns:.3f})"
)
print(
    f"[cpmg] Active transition: {active_transition} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}"
)
print(f"[cpmg] n_cpmg={config.n_cpmg}, mw_pi2_ftsamp={config.mw_pi2_ftsamp}, reps={config.reps}")

if maybe_run_chunked_mode(
    run_mode=RUN_MODE,
    program_class=CPMGXYFineRes,
    cfg_dict=cfg,
    build_config_for_chunk=_build_chunk_config_context,
    output_dir=OUTPUT_DIR,
    experiment_name="cpmg_xy_fine_res",
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
    piezo_initial_position_um=PIEZO_INITIAL_POSITION_UM,
    combine_filename="cpmg_xy_fine_res_combined.h5",
    combine_fn=combine_chunk_hdf5_files,
):
    raise SystemExit(0)

if RUN_MODE != "single":
    raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

# =============================================================================
# Acquire (single run mode)
# =============================================================================

if RUN_MODE == "single":
    prog = CPMGXYFineRes(config)
    data = prog.acquire(progress=bool(ACQUIRE_PROGRESS))
else:
    raise SystemExit(0)

# =============================================================================
# Save to HDF5 (single run mode)
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    CPMGXYFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="cpmg_xy_fine_res",
)
run_id = out_path.stem

print(f"[cpmg] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

if not hasattr(data, "tau_ftus"):
    raise ValueError("CPMG output missing expected sweep axis tau_ftus.")
x_axis = np.asarray(data.tau_ftus, dtype=float)

traces = extract_standard_traces(data, x_axis=x_axis, use_counts_s=PLOT_USE_COUNTS_S)
metadata = build_plot_metadata(
    program_class=CPMGXYFineRes,
    config_obj=config,
    base_metadata={
    "run_id": run_id,
    "mw_MHz": f"{config.mw_fMHz:.3f}",
    "gain": config.mw_gain,
    "pi2_ftsamp": config.mw_pi2_ftsamp,
    "pi2_ftns": f"{config.mw_pi2_ftns:.2f}",
    "sequence": "pi/2_y - {tau - pi_xy8 - tau}xN - pi/2_-y",
    "xy8_phase_order": "XYXYYXYX",
    "n_cpmg": config.n_cpmg,
    "reps": config.reps,
    "laser_mW": cfg["optics"]["excitation_laser_power_mW"],
    "units": "cts/s" if PLOT_USE_COUNTS_S else "raw",
    },
)

sequence_label = r"$\pi/2_y - \{\tau - \pi_{XY8} - \tau\}\times N - \pi/2_{-y}$"

if PLOT_DEBUG_RAW:
    plot_debug_traces(
        x_axis,
        traces,
        x_label=r"$\tau$ (ftus)",
        y_label="Counts/s" if PLOT_USE_COUNTS_S else "Counts",
        title=(
            build_standard_title(
                experiment_label="CPMG-XY Debug Raw",
                sequence_label=f"{sequence_label}; phase XYXYYXYX",
                run_id=run_id,
            )
        ),
        metadata=metadata,
        metadata_position=PLOT_METADATA_POSITION,
    )

fig, _, _, _ = plot_contrast_twin(
    x_axis,
    traces,
    x_label=r"$\tau$ (ftus)",
    title=build_standard_title(
        experiment_label="CPMG-XY",
        sequence_label=f"{sequence_label}; phase XYXYYXYX",
        run_id=run_id,
    ),
    metadata=metadata,
    metadata_position=PLOT_METADATA_POSITION,
)

plot_path = out_path.with_suffix(".png")
fig.savefig(plot_path, dpi=150)
plt.show()
print(f"[cpmg] Plot saved -> {plot_path}")