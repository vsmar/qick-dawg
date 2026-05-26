"""
cpmg.py — CPMG-XY Fine-Resolution Experiment
============================================
Runs a CPMG-XY sweep, saves data + full config to HDF5, plots result.

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
# EXPERIMENT PARAMETERS — edit these before each run
# =============================================================================

# Sweep bounds in fine-time nanoseconds (ftns).
TAU_START_FTNS = 5_000.0
TAU_STOP_FTNS = 6_000.0
TAU_DELTA_FTNS = 3

N_CPMG = 32  # 128
REPS = 2000

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 200_000
CHUNK_REPS = 10_000
ACQUIRE_PROGRESS = True #False

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_FREQ_FMHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

GET_REFERENCE = False # True

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "cpmg"


def _build_cpmg_config(cfg: dict, reps: int):
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

    config.n_cpmg = int(N_CPMG)
    config.add_linear_sweep(
        "tau",
        "ftns",
        start=TAU_START_FTNS,
        stop=TAU_STOP_FTNS,
        delta=TAU_DELTA_FTNS,
    )

    context = {
        "transition": active_transition,
        "n_cpmg": int(N_CPMG),
        "sweep": {
            "axis": "tau",
            "unit": "ftns",
            "start": float(TAU_START_FTNS),
            "stop": float(TAU_STOP_FTNS),
            "delta": float(TAU_DELTA_FTNS),
            "mode": "linear",
        },
    }

    return config, context

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, context = _build_cpmg_config(cfg, REPS)

header_lines = [
    f"[cpmg] Sweep: {config.tau_start_ftns:.3f} -> {config.tau_end_ftns:.3f} ftns "
    f"(delta {config.tau_delta_ftns:.3f})",
    f"[cpmg] Active transition: {context['transition']} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}",
    f"[cpmg] n_cpmg={config.n_cpmg}, mw_pi2_ftsamp={config.mw_pi2_ftsamp}, reps={config.reps}",
]

plot = PlotSpec(
    chunked_callback=make_chunked_plot_callback(Visualizer.plot_cpmg, config=config, plot_kwargs={"view": "contrast"}),
    chunked_filename="cpmg_aggregated.png",
    sweep_axis_key="tau_ftns",
    single_plotter=Visualizer.plot_cpmg,
    single_plot_kwargs={"view": "contrast"},
)
chunk = ChunkSpec(
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
)
spec = ExperimentSpec(
    name="cpmg_xy_fine_res",
    program_class=CPMGXYFineRes,
    cfg_dict=cfg,
    output_dir=OUTPUT_DIR,
    run_mode=RUN_MODE,
    single_reps=int(REPS),
    build_config_for_reps=lambda reps: _build_cpmg_config(cfg, reps),
    chunk=chunk,
    plot=plot,
    header_lines=header_lines,
    initial_config=config,
    initial_context=context,
)

result = run_experiment(spec)

if result.get("mode") == "chunked":
    raise SystemExit(0)

print(f"[cpmg] Saved -> {result['out_path']}")
