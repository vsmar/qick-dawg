"""
t1.py - T1 Time-Delay Sweep Experiment
======================================
Runs a T1 delay sweep using T1FineRes, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import T1FineRes, Visualizer

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
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

# T1 delay sweep bounds in microseconds.
T1_DELAY_START_TUS = 1.0
T1_DELAY_END_TUS = 3000.0

SCALING_MODE = "exponential"  # "linear" or "exponential"
T1_DELAY_DELTA_TUS = 400.0  # Ignored when SCALING_MODE is "exponential"
SCALING_FACTOR = "5/4"  # Ignored when SCALING_MODE is "linear"

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

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "t1"


def _build_t1_config(cfg: dict, reps: int):
    config, active_transition, pi_source = build_common_config(
        cfg,
        reps,
        transition=TRANSITION,
        override_mw_gain=OVERRIDE_MW_GAIN,
        override_mw_pi_ftsamp=OVERRIDE_MW_PI_FTSAMP,
        override_mw_pi_ftns=OVERRIDE_MW_PI_NS,
        get_reference=GET_REFERENCE,
    )

    # Keep delay delta populated for downstream code paths even with exponential sweep.
    config.delay_delta_tus = float(T1_DELAY_DELTA_TUS)
    config.scaling_factor = SCALING_FACTOR

    if SCALING_MODE == "linear":
        config.add_linear_sweep(
            "delay",
            "tus",
            start=T1_DELAY_START_TUS,
            stop=T1_DELAY_END_TUS,
            delta=T1_DELAY_DELTA_TUS,
        )
    elif SCALING_MODE == "exponential":
        config.add_exponential_sweep(
            "delay",
            "tus",
            start=T1_DELAY_START_TUS,
            stop=T1_DELAY_END_TUS,
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
        f"[t1] Sweep: {config.delay_start_tus:.3f} -> {config.delay_end_tus:.3f} us "
        f"({SCALING_MODE}, configured delta={config.delay_delta_tus:.3f})"
    )
else:
    print(
        f"[t1] Sweep: {config.delay_start_tus:.3f} -> {config.delay_end_tus:.3f} us "
        f"({SCALING_MODE}, scaling_factor={SCALING_FACTOR})"
    )
print(f"[t1] Active transition: {active_transition} | pi pulse: {pi_source}")
print(f"[t1] mw_gain={config.mw_gain}, reps={config.reps}")


def _build_chunk_config_context(reps: int):
    chunk_cfg, chunk_transition, chunk_pi_source = _build_t1_config(cfg, reps)
    return (
        chunk_cfg,
        {
            "transition": chunk_transition,
            "pi_source": chunk_pi_source,
            "scaling_mode": SCALING_MODE,
            "delay_start_tus": float(T1_DELAY_START_TUS),
            "delay_end_tus": float(T1_DELAY_END_TUS),
            "delay_delta_tus": float(T1_DELAY_DELTA_TUS),
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
    plot_callback=make_chunked_plot_callback(Visualizer.plot_t1, config=config, plot_kwargs={"fit": True, "view": "contrast"}),
    plot_filename="t1_aggregated.png",
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

Visualizer.plot_t1(data, cfg=config, fit=True, view="contrast")
