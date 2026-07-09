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

# Sweep bounds in MHz.
ODMR_START_fMHz = 3893 # 1842 
ODMR_STOP_fMHz = 3900 # 1849
ODMR_DELTA_fMHz = 0.10

REPS = 800_000

RUN_MODE = "single"  # "single" or "chunked"
TARGET_TOTAL_REPS = 1_200_000
CHUNK_REPS = 400_000
ACQUIRE_PROGRESS = True

# Transition — set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = "upper_dip"

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_MW_GAIN = 1200 #500 # 2000

# Set either ftsamp directly, or ns (which will be converted to ftsamp).
OVERRIDE_MW_PI_FTSAMP = None
OVERRIDE_MW_PI_NS = 2500 # 2300

GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "podmr"


def _build_podmr_config(cfg: dict, reps: int):
    config, active_transition, pi_source = build_common_config(
        cfg,
        reps,
        transition=TRANSITION,
        override_freq_fMHz=None,
        override_mw_gain=OVERRIDE_MW_GAIN,
        override_mw_pi_ftsamp=OVERRIDE_MW_PI_FTSAMP,
        override_mw_pi_ftns=OVERRIDE_MW_PI_NS,
        get_reference=GET_REFERENCE,
    )

    config.add_linear_sweep("mw", "fMHz", start=ODMR_START_fMHz, stop=ODMR_STOP_fMHz, delta=ODMR_DELTA_fMHz)
    context = {
        "transition": active_transition,
        "pulse_sources": {"pi": pi_source},
        "sweep": {
            "axis": "mw",
            "unit": "fMHz",
            "start": float(ODMR_START_fMHz),
            "stop": float(ODMR_STOP_fMHz),
            "delta": float(ODMR_DELTA_fMHz),
            "mode": "linear",
        },
    }
    return config, context

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, context = _build_podmr_config(cfg, REPS)

header_lines = [
    f"[podmr] Sweep: {ODMR_START_fMHz:.3f} -> {ODMR_STOP_fMHz:.3f} fMHz "
    f"(delta {ODMR_DELTA_fMHz:.3f} fMHz)",
    f"[podmr] Active transition: {context['transition']} | "
    f"mw_fMHz={config.mw_fMHz} MHz | mw_gain={config.mw_gain}",
    f"[podmr] pi pulse: {context['pulse_sources']['pi']}",
]

plot = PlotSpec(
    chunked_callback=make_chunked_plot_callback(
        Visualizer.plot_podmr,
        config=config,
        plot_kwargs={"fit": True, "view": "contrast"},
    ),
    chunked_filename="podmr_aggregated.png",
    sweep_axis_key="mw_fMHz",
    single_plotter=Visualizer.plot_podmr,
    single_plot_kwargs={"fit": True, "view": "contrast"},
)
chunk = ChunkSpec(
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
)
spec = ExperimentSpec(
    name="podmr_fine_res",
    program_class=PODMRFineRes,
    cfg_dict=cfg,
    output_dir=OUTPUT_DIR,
    run_mode=RUN_MODE,
    single_reps=int(REPS),
    build_config_for_reps=lambda reps: _build_podmr_config(cfg, reps),
    chunk=chunk,
    plot=plot,
    header_lines=header_lines,
    initial_config=config,
    initial_context=context,
)

result = run_experiment(spec)

if result.get("mode") == "chunked":
    raise SystemExit(0)

print(f"[podmr] Saved -> {result['out_path']}")
