"""
ramsey.py - Ramsey Fine-Resolution Experiment
=============================================
Runs a Ramsey fine-resolution sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import RamseyFineRes, Visualizer

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

# Sweep bounds in fine-time nanoseconds (ftns).
TAU_START_FTNS = 200.0
TAU_END_FTNS = 15_000.0

SCALING_MODE = "linear" # "linear" or "exponential"
TAU_DELTA_FTNS = 100.0   # Ignored when scaling_mode is 'exponential'
SCALING_FACTOR = "5/4"   # Ignored when scaling_mode is 'linear'

REPS = 80_000

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 240_000
CHUNK_REPS = 80_000
ACQUIRE_PROGRESS = True
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = (0.8991, -6.7037, 0.0069)  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition - set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

FREQ_DETUNE_MHZ = None

GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ramsey"


def _build_ramsey_config(cfg: dict, reps: int):
    config, active_transition, _ = build_common_config(
        cfg,
        reps,
        transition=TRANSITION,
        override_freq_mhz=OVERRIDE_FREQ_MHZ,
        override_mw_gain=OVERRIDE_MW_GAIN,
        override_mw_pi_ftsamp=OVERRIDE_MW_PI2_FTSAMP,
        override_mw_pi_ftns=OVERRIDE_MW_PI2_FTNS,
        get_reference=GET_REFERENCE,
    )

    # apply detune if requested
    config.mw_fMHz += FREQ_DETUNE_MHZ if FREQ_DETUNE_MHZ is not None else 0.0

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

    return config, active_transition

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, active_transition = _build_ramsey_config(cfg, REPS)

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


def _build_chunk_config_context(reps: int):
    chunk_cfg, chunk_transition = _build_ramsey_config(cfg, reps)
    return (
        chunk_cfg,
        {
            "transition": chunk_transition,
            "scaling_mode": SCALING_MODE,
            "tau_start_ftns": float(TAU_START_FTNS),
            "tau_end_ftns": float(TAU_END_FTNS),
            "tau_delta_ftns": float(TAU_DELTA_FTNS),
            "scaling_factor": SCALING_FACTOR,
        },
    )


if maybe_run_chunked_mode(
    run_mode=RUN_MODE,
    program_class=RamseyFineRes,
    cfg_dict=cfg,
    build_config_for_chunk=_build_chunk_config_context,
    output_dir=OUTPUT_DIR,
    experiment_name="ramsey_fine_res",
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
    piezo_initial_position_um=PIEZO_INITIAL_POSITION_UM,
    plot_callback=make_chunked_plot_callback(Visualizer.plot_ramsey, config=config, plot_kwargs={"fit": True, "view": "contrast", "fit_mode": "oscillatory"}),
    plot_filename="ramsey_aggregated.png",
):
    raise SystemExit(0)

if RUN_MODE != "single":
    raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

# =============================================================================
# Acquire
# =============================================================================

prog = RamseyFineRes(config)
data = prog.acquire(progress=bool(ACQUIRE_PROGRESS))

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

Visualizer.plot_ramsey(data, cfg=config, fit=True, view="contrast", fit_mode="oscillatory")
