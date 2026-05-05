"""
counting_duration.py
====================
Run a counting-duration fine-resolution sweep, save data, and generate
SNR vs integration time visualization.

Edit the EXPERIMENT PARAMETERS block before each run.
"""

from pathlib import Path
from copy import copy

import numpy as np
from tqdm.auto import tqdm

from qickdawg.finetimingsuite import CountingDurationFineRes, Visualizer

from config import (
    load_config,
    build_nv_config,
    connect,
    save_experiment_hdf5,
)
from experiment_helpers import build_common_config

# =============================================================================
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

READOUT_OFFSET_START_TNS = 1000
READOUT_OFFSET_STOP_TNS = 2500
READOUT_OFFSET_STEP_TNS = 10

REPS = 200_000
READOUT_INTEGRATION_TNS = 10
GET_REFERENCE = True

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "counting_duration"


# =============================================================================
# Acquisition helpers
# =============================================================================

def _as_scalar(value):
    """Extract scalar from array-like value."""
    arr = np.asarray(value).squeeze()
    if arr.size == 1:
        return float(arr)
    return float(np.mean(arr))


def _acquire_full_res(config, delays_tns, show_progress=True):
    """Acquire counting duration sweep over readout offsets."""
    results = None

    total_steps = len(delays_tns)
    sweep_iter = tqdm(
        delays_tns,
        total=total_steps,
        desc="Counting-duration sweep",
        unit="step",
        disable=not show_progress,
    )

    for idx, delay_tns in enumerate(sweep_iter, start=1):
        config.laser_readout_offset_tns = int(delay_tns)

        prog = CountingDurationFineRes(config)
        d = prog.acquire()

        if results is None:
            results = copy(d)
            for key in d.keys():
                results[key] = np.empty(0, dtype=float)
            results.delay = np.empty(0, dtype=float)

        for key, value in d.items():
            results[key] = np.append(results[key], _as_scalar(value))

        results.delay = np.append(results.delay, config.laser_readout_offset_tus)

        sweep_iter.set_postfix({
            "offset_ns": int(delay_tns),
            "done": f"{idx}/{total_steps}",
        }, refresh=False)

    return results

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)

config, _, _ = build_common_config(cfg, REPS, get_reference=GET_REFERENCE)
config.readout_integration_tns = int(READOUT_INTEGRATION_TNS)

# Create sweep array
delays_tns = np.arange(READOUT_OFFSET_START_TNS, READOUT_OFFSET_STOP_TNS + 1, READOUT_OFFSET_STEP_TNS)
print(f"[counting_duration] Sweep: {len(delays_tns)} points from {READOUT_OFFSET_START_TNS} to {READOUT_OFFSET_STOP_TNS} ns")
print(f"[counting_duration] Reps: {REPS}, Integration: {READOUT_INTEGRATION_TNS} ns")

# =============================================================================
# Acquire
# =============================================================================

data = _acquire_full_res(config, delays_tns, show_progress=True)

# =============================================================================
# Save to HDF5
# =============================================================================

out_path, timestamp = save_experiment_hdf5(
    CountingDurationFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="counting_duration_fine_res",
)
run_id = out_path.stem

print(f"[counting_duration] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

Visualizer.plot_experiment(
    data,
    {
        "name": "Counting Duration",
        "x_key": "delay",
        "x_label": "Readout Offset (�s)",
        "y_label": "Photon Rate (cts/s)",
        "traces": {
            "Signal": "signal1_cts_s",
            "Reference": "signal2_cts_s",
        },
    },
    cfg=config,
    fit=False,
    view="raw"
)
