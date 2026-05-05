"""
podmr.py — Pulsed-ODMR Fine-Resolution Experiment
==================================================
Runs a PODMR frequency sweep, saves data + full config to HDF5, and plots
signal/reference/contrast with a dip fit on contrast ratio.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import PODMRFineRes, Visualizer

from config import (
    load_config,
    build_nv_config,
    connect,
    save_experiment_hdf5,
)
from experiment_helpers import (
    maybe_run_chunked_mode,
    build_common_config,
    make_chunked_plot_callback,
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

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "podmr"


def _build_podmr_config(cfg: dict, reps: int):
    config, active_transition, pi_source = build_common_config(
        cfg,
        reps,
        transition=TRANSITION,
        override_freq_mhz=None,
        override_mw_gain=OVERRIDE_MW_GAIN,
        override_mw_pi_ftsamp=OVERRIDE_MW_PI_FTSAMP,
        override_mw_pi_ftns=OVERRIDE_MW_PI_NS,
        get_reference=GET_REFERENCE,
    )

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
    plot_callback=make_chunked_plot_callback(Visualizer.plot_podmr, config=config, plot_kwargs={"fit": True, "view": "contrast"}),
    plot_filename="podmr_aggregated.png",
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

Visualizer.plot_podmr(data, cfg=config, fit=True, view="contrast")
