"""
cpmg.py — CPMG-XY Fine-Resolution Experiment
============================================
Runs a CPMG-XY sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import CPMGXYFineRes, Visualizer

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

# Sweep bounds in fine-time nanoseconds (ftns).
TAU_START_FTNS = 200.0
TAU_STOP_FTNS = 4_000.0
TAU_DELTA_FTNS = 2

N_CPMG = 32  # 128
REPS = 2000

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 100_000
CHUNK_REPS = 4_000
ACQUIRE_PROGRESS = False
# Required in chunked mode to avoid startup piezo reset behavior.
PIEZO_INITIAL_POSITION_UM = (0.83627, -6.7762, 0.00793)  # e.g. (-2.2799, 0.7189, -2.8593)

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

GET_REFERENCE = False # True

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cpmg"


def _build_cpmg_config(cfg: dict, reps: int):
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

    config.n_cpmg = int(N_CPMG)
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
    plot_callback=make_chunked_plot_callback(Visualizer.plot_cpmg, config=config, plot_kwargs={"view": "contrast"}),
    plot_filename="cpmg_aggregated.png",
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

Visualizer.plot_cpmg(data, cfg=config, view="contrast")