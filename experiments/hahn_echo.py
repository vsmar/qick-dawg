"""
hahn_echo.py � Hahn Echo Fine-Resolution Experiment
====================================================
Runs a Hahn echo tau sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import HahnEchoFineRes, Visualizer

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
# EXPERIMENT PARAMETERS � edit these before each run
# =============================================================================

# Sweep bounds in fine-time nanoseconds (ftns).
TAU_START_FTNS = 200.0
TAU_END_FTNS = 800_000.0
TAU_DELTA_FTNS = 0.0 # Ignored when scaling_mode is 'exponential'
SCALING_FACTOR = "9/8"

REPS = 50_000

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 600_000
CHUNK_REPS = 30_000
ACQUIRE_PROGRESS = True
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = (-0.5888, 0.8658, -3.890)  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition � set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "hahn_echo"


def _build_hahn_echo_config(cfg: dict, reps: int):
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

    # Keep tau delta populated for downstream code paths even with exponential sweep.
    config.tau_delta_ftns = float(TAU_DELTA_FTNS)
    config.add_exponential_sweep(
        "tau",
        "ftns",
        start=TAU_START_FTNS,
        stop=TAU_END_FTNS,
        scaling_factor=SCALING_FACTOR,
    )
    return config, active_transition

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, active_transition = _build_hahn_echo_config(cfg, REPS)

print(
    f"[hahn_echo] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
    f"(exponential, scaling_factor={SCALING_FACTOR})"
)
print(
    f"[hahn_echo] Active transition: {active_transition} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}"
)
print(f"[hahn_echo] mw_pi2_ftsamp={config.mw_pi2_ftsamp}, reps={config.reps}")


def _build_chunk_config_context(reps: int):
    chunk_cfg, chunk_transition = _build_hahn_echo_config(cfg, reps)
    return (
        chunk_cfg,
        {
            "transition": chunk_transition,
            "tau_start_ftns": float(TAU_START_FTNS),
            "tau_end_ftns": float(TAU_END_FTNS),
            "tau_delta_ftns": float(TAU_DELTA_FTNS),
            "scaling_factor": SCALING_FACTOR,
        },
    )


if maybe_run_chunked_mode(
    run_mode=RUN_MODE,
    program_class=HahnEchoFineRes,
    cfg_dict=cfg,
    build_config_for_chunk=_build_chunk_config_context,
    output_dir=OUTPUT_DIR,
    experiment_name="hahn_echo_fine_res",
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
    piezo_initial_position_um=PIEZO_INITIAL_POSITION_UM,
    plot_callback=make_chunked_plot_callback(Visualizer.plot_hahnecho, config=config, plot_kwargs={"fit": True, "view": "contrast"}),
    plot_filename="hahn_echo_aggregated.png",
):
    raise SystemExit(0)

if RUN_MODE != "single":
    raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

# =============================================================================
# Acquire
# =============================================================================

prog = HahnEchoFineRes(config)
data = prog.acquire(progress=bool(ACQUIRE_PROGRESS))

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    HahnEchoFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="hahn_echo_fine_res",
)
run_id = out_path.stem

print(f"[hahn_echo] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

Visualizer.plot_hahnecho(data, cfg=config, fit=True, view="contrast")
