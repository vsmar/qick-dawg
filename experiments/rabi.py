"""
rabi.py — Rabi Fine-Resolution Experiment
==========================================
Runs a Rabi sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import RabiFineRes, Visualizer

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

# Sweep bounds in nanoseconds.
# NVConfiguration handles conversion to ftsamp/treg companion units.
MW_DURATION_START_NS = 5     # ns
MW_DURATION_STOP_NS  = 400    #2000    # ns
MW_DURATION_DELTA_NS = 0.2     # ns  (step size)

REPS          = 20000 #*3 # 4

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 500_000
CHUNK_REPS = 40_000
ACQUIRE_PROGRESS = True
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = (0.439, -6.934, -0.3742)  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition — set to "lower_dip", "upper_dip", or None to use config.yaml default.
TRANSITION    = None   # None = use calibration.default_transition

# Optional per-run overrides. If left None, values come from transition calibration.
OVERRIDE_FREQ_MHZ = None   # e.g. 1845.7
OVERRIDE_MW_GAIN  = None   # e.g. 1200

GET_REFERENCE = True          # acquire reference readout with MW gain = 0

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "rabi"


def _build_rabi_config(cfg: dict, reps: int):
    config, active_transition, _ = build_common_config(
        cfg,
        reps,
        transition=TRANSITION,
        override_freq_mhz=OVERRIDE_FREQ_MHZ,
        override_mw_gain=OVERRIDE_MW_GAIN,
        get_reference=GET_REFERENCE,
    )

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
    plot_callback=make_chunked_plot_callback(Visualizer.plot_rabi, config=config, plot_kwargs={"fit": True, "view": "contrast"}),
    plot_filename="rabi_aggregated.png",
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

Visualizer.plot_rabi(data, cfg=config, fit=True, view="contrast")