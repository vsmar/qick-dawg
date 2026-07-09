"""
ramsey.py - Ramsey Fine-Resolution Experiment
=============================================
Runs a Ramsey fine-resolution sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import CPMGXYFineRes, Visualizer

from experiments.helpers.config import (
    load_config,
    connect,
)
from experiments.helpers.config_builders import build_common_config
from experiments.helpers.experiment_runner import run_experiment
from experiments.helpers.experiment_specs import ChunkSpec, ExperimentSpec, PlotSpec
from experiments.helpers.plotting import make_chunked_plot_callback

# =============================================================================
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

# Sweep bounds in fine-time nanoseconds (ftns).
TAU_START_FTNS = 100.0
TAU_END_FTNS = 30_000.0

SCALING_MODE = "linear" # "linear" or "exponential"
TAU_DELTA_FTNS = 100.0   # Ignored when scaling_mode is 'exponential'
SCALING_FACTOR = "3/2"   # Ignored when scaling_mode is 'linear'

REPS = 100_000

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 300_000
CHUNK_REPS = 100_000
ACQUIRE_PROGRESS = True
SAVED_AXIS_KEY = "tau_ftns"

# Transition - set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_FMHZ = 1838.15-2
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

FREQ_DETUNE_FMHZ = None

GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "ramsey"


def _build_ramsey_config(cfg: dict, reps: int):
    config, active_transition, _ = build_common_config(
        cfg,
        reps,
        transition=TRANSITION,
        override_freq_fMHz=OVERRIDE_FREQ_FMHZ,
        override_mw_gain=OVERRIDE_MW_GAIN,
        override_mw_pi_ftsamp=OVERRIDE_MW_PI2_FTSAMP,
        override_mw_pi_ftns=OVERRIDE_MW_PI2_FTNS,
        get_reference=GET_REFERENCE,
    )

    # apply detune if requested
    config.mw_fMHz += FREQ_DETUNE_FMHZ if FREQ_DETUNE_FMHZ is not None else 0.0

    # Keep tau delta populated for downstream code paths even with exponential sweep.
    config.tau_delta_ftns = float(TAU_DELTA_FTNS)
    config.scaling_factor = SCALING_FACTOR

    config.n_cpmg = 0

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

    context = {
        "transition": active_transition,
        "sweep": {
            "axis": "tau",
            "unit": "ftns",
            "start": float(TAU_START_FTNS),
            "stop": float(TAU_END_FTNS),
            "delta": float(TAU_DELTA_FTNS),
            "scaling_factor": SCALING_FACTOR,
            "mode": SCALING_MODE,
        },
        "freq_detune_fMHz": FREQ_DETUNE_FMHZ,
    }

    return config, context

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, context = _build_ramsey_config(cfg, REPS)

# Use CPMG implementation with n=0 which is equivalent to a Ramsey experiment
config.n_cpmg = 0

if SCALING_MODE == "linear":
    sweep_line = (
        f"[ramsey] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
        f"({SCALING_MODE}, configured delta={config.tau_delta_ftns:.3f})"
    )
else:
    sweep_line = (
        f"[ramsey] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
        f"({SCALING_MODE}, scaling_factor={SCALING_FACTOR})"
    )

header_lines = [
    sweep_line,
    f"[ramsey] Active transition: {context['transition']} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}",
    f"[ramsey] mw_pi2_ftsamp={config.mw_pi2_ftsamp}, reps={config.reps}",
]

plot = PlotSpec(
    chunked_callback=make_chunked_plot_callback(
        Visualizer.plot_ramsey,
        config=config,
        plot_kwargs={"fit": True, "view": "contrast", "fit_mode": "oscillatory"},
    ),
    chunked_filename="ramsey_aggregated.png",
    sweep_axis_key=SAVED_AXIS_KEY,
    single_plotter=Visualizer.plot_ramsey,
    single_plot_kwargs={"fit": True, "view": "contrast", "fit_mode": "oscillatory"},
)
chunk = ChunkSpec(
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
)
spec = ExperimentSpec(
    name="ramsey_fine_res",
    program_class=CPMGXYFineRes,
    cfg_dict=cfg,
    output_dir=OUTPUT_DIR,
    run_mode=RUN_MODE,
    single_reps=int(REPS),
    build_config_for_reps=lambda reps: _build_ramsey_config(cfg, reps),
    chunk=chunk,
    plot=plot,
    header_lines=header_lines,
    initial_config=config,
    initial_context=context,
)

result = run_experiment(spec)

if result.get("mode") == "chunked":
    raise SystemExit(0)

print(f"[ramsey] Saved -> {result['out_path']}")
