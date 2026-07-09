"""
rabi.py — Rabi Fine-Resolution Experiment
==========================================
Runs a Rabi sweep, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import RabiFineRes, Visualizer

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

# Sweep bounds in nanoseconds.
# NVConfiguration handles conversion to ftsamp/treg companion units.
MW_DURATION_START_FTNS = 10000     # ftns
MW_DURATION_STOP_FTNS  = 10500    #2000    # ftns
MW_DURATION_DELTA_FTNS = 2     # ftns  (step size)

REPS = 100_000 #*3 # 4

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 300_000
CHUNK_REPS = 100_000
ACQUIRE_PROGRESS = True

# Transition — set to "lower_dip", "upper_dip", or None to use config.yaml default.
TRANSITION    = None   # None = use calibration.default_transition

# Optional per-run overrides. If left None, values come from transition calibration.
OVERRIDE_FREQ_FMHZ = 1838.15-0.6   # e.g. 1845.7
OVERRIDE_MW_GAIN  = None   # e.g. 1200

GET_REFERENCE = True          # acquire reference readout with MW gain = 0

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "rabi"


def _build_rabi_config(cfg: dict, reps: int):
    config, active_transition, _ = build_common_config(
        cfg,
        reps,
        transition=TRANSITION,
        override_freq_fMHz=OVERRIDE_FREQ_FMHZ,
        override_mw_gain=OVERRIDE_MW_GAIN,
        get_reference=GET_REFERENCE,
    )

    config.add_linear_sweep(
        "mw_duration",
        "ftns",
        start=MW_DURATION_START_FTNS,
        stop=MW_DURATION_STOP_FTNS,
        delta=MW_DURATION_DELTA_FTNS,
    )
    context = {
        "transition": active_transition,
        "sweep": {
            "axis": "mw_duration",
            "unit": "ftns",
            "start": float(MW_DURATION_START_FTNS),
            "stop": float(MW_DURATION_STOP_FTNS),
            "delta": float(MW_DURATION_DELTA_FTNS),
            "mode": "linear",
        },
    }
    return config, context

# =============================================================================
# Setup
# =============================================================================

cfg    = load_config()
connect(cfg)

config, context = _build_rabi_config(cfg, REPS)

header_lines = [
    f"[rabi] Sweep: {MW_DURATION_START_FTNS:.3f} -> {MW_DURATION_STOP_FTNS:.3f} ftns "
    f"(delta {MW_DURATION_DELTA_FTNS:.3f} ftns)",
    f"[rabi] Active transition: {context['transition']} | "
    f"freq={config.mw_fMHz} MHz | gain={config.mw_gain}",
]

plot = PlotSpec(
    chunked_callback=make_chunked_plot_callback(
        Visualizer.plot_rabi,
        config=config,
        plot_kwargs={"fit": True, "view": "contrast"},
    ),
    chunked_filename="rabi_aggregated.png",
    sweep_axis_key="mw_duration_ftns",
    single_plotter=Visualizer.plot_rabi,
    single_plot_kwargs={"fit": True, "view": "contrast"},
)
chunk = ChunkSpec(
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
)
spec = ExperimentSpec(
    name="rabi_fine_res",
    program_class=RabiFineRes,
    cfg_dict=cfg,
    output_dir=OUTPUT_DIR,
    run_mode=RUN_MODE,
    single_reps=int(REPS),
    build_config_for_reps=lambda reps: _build_rabi_config(cfg, reps),
    chunk=chunk,
    plot=plot,
    header_lines=header_lines,
    initial_config=config,
    initial_context=context,
)

result = run_experiment(spec)

if result.get("mode") == "chunked":
    raise SystemExit(0)

print(f"[rabi] Saved -> {result['out_path']}")
