"""
t1.py - T1 Time-Delay Sweep Experiment
======================================
Runs a T1 delay sweep using T1FineRes, saves data + full config to HDF5, plots result.

Edit the EXPERIMENT PARAMETERS block before each run.
Everything else is pulled from config.yaml via config.py.
"""

from pathlib import Path

from qickdawg.finetimingsuite import T1FineRes, Visualizer

from experiments.helpers.config import load_config, connect
from experiments.helpers.config_builders import build_common_config
from experiments.helpers.experiment_runner import run_experiment
from experiments.helpers.experiment_specs import ChunkSpec, ExperimentSpec, PlotSpec
from experiments.helpers.plotting import make_chunked_plot_callback

# =============================================================================
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

# T1 delay sweep bounds in microseconds.
T1_DELAY_START_TUS = 1.0
T1_DELAY_END_TUS = 3000 #00.0

SCALING_MODE = "exponential"  # "linear" or "exponential"
T1_DELAY_DELTA_TUS = 400.0  # Ignored when SCALING_MODE is "exponential"
SCALING_FACTOR = "5/4"  # Ignored when SCALING_MODE is "linear"

REPS = 30_000

RUN_MODE = "chunked"  # "single" or "chunked"
TARGET_TOTAL_REPS = 240_000
CHUNK_REPS = 30_000
ACQUIRE_PROGRESS = True

# Transition - set to "lower_dip", "upper_dip", or None to use config default.
TRANSITION = None

# Optional per-run overrides. If None, values come from selected transition.
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI_FTSAMP = None
OVERRIDE_MW_PI_NS = None

GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "t1"


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

    context = {
        "transition": active_transition,
        "pulse_sources": {"pi": pi_source},
        "sweep": {
            "axis": "delay",
            "unit": "tus",
            "start": float(T1_DELAY_START_TUS),
            "stop": float(T1_DELAY_END_TUS),
            "delta": float(T1_DELAY_DELTA_TUS),
            "scaling_factor": SCALING_FACTOR,
            "mode": SCALING_MODE,
        },
    }

    return config, context

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)
config, context = _build_t1_config(cfg, REPS)

if SCALING_MODE == "linear":
    sweep_line = (
        f"[t1] Sweep: {config.delay_start_tus:.3f} -> {config.delay_end_tus:.3f} us "
        f"({SCALING_MODE}, configured delta={config.delay_delta_tus:.3f})"
    )
else:
    sweep_line = (
        f"[t1] Sweep: {config.delay_start_tus:.3f} -> {config.delay_end_tus:.3f} us "
        f"({SCALING_MODE}, scaling_factor={SCALING_FACTOR})"
    )

header_lines = [
    sweep_line,
    f"[t1] Active transition: {context['transition']} | pi pulse: {context['pulse_sources']['pi']}",
    f"[t1] mw_gain={config.mw_gain}, reps={config.reps}",
]

plot = PlotSpec(
    chunked_callback=make_chunked_plot_callback(
        Visualizer.plot_t1,
        config=config,
        plot_kwargs={"fit": True, "view": "contrast"},
    ),
    chunked_filename="t1_aggregated.png",
    sweep_axis_key="delay_tus",
    single_plotter=Visualizer.plot_t1,
    single_plot_kwargs={"fit": True, "view": "contrast"},
)
chunk = ChunkSpec(
    target_total_reps=int(TARGET_TOTAL_REPS),
    chunk_reps=int(CHUNK_REPS),
    acquire_progress=bool(ACQUIRE_PROGRESS),
)
spec = ExperimentSpec(
    name="t1_fine_res",
    program_class=T1FineRes,
    cfg_dict=cfg,
    output_dir=OUTPUT_DIR,
    run_mode=RUN_MODE,
    single_reps=int(REPS),
    build_config_for_reps=lambda reps: _build_t1_config(cfg, reps),
    chunk=chunk,
    plot=plot,
    header_lines=header_lines,
    initial_config=config,
    initial_context=context,
)

result = run_experiment(spec)

if result.get("mode") == "chunked":
    raise SystemExit(0)

print(f"[t1] Saved -> {result['out_path']}")
