"""
qdlscan_reopt_hook_example.py

Minimal orchestration hook for a workflow where qick and qdlscan are independent:
1) Manual optimize once before starting this script.
2) Run a short qick experiment.
3) Pause, laser on, run XYZ reoptimization, laser off.
4) Start the next short qick experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import importlib
import importlib.resources
import yaml


import numpy as np

def _bootstrap_qdlutils_import() -> None:
    """
    Make qdlutils importable when this script is run from another repo.

    Resolution order:
    1) Existing environment/site-packages
    2) QDLUTILS_SRC environment variable (path to qdl-utils/src)
    3) Common local path near this repo: ../../../../qdl-utils/src
    """
    try:
        import qdlutils  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    candidates = []

    env_src = os.environ.get("QDLUTILS_SRC")
    if env_src:
        candidates.append(Path(env_src))

    here = Path(__file__).resolve()
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "qdl-utils" / "src")

    for candidate in candidates:
        if (candidate / "qdlutils").is_dir():
            sys.path.insert(0, str(candidate))
            break

    try:
        import qdlutils  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import qdlutils. Install it into this environment "
            "(for example: pip install -e C:/Users/QT3 User Facility/qdl-utils) "
            "or set QDLUTILS_SRC to the qdl-utils/src folder."
        ) from exc


_bootstrap_qdlutils_import()

from qdlutils.applications.qdlscan.reoptimizer import AxisOptimizationResult, Reoptimizer

# Axis order and scan parameters
AXIS_ORDER = ("x", "y", "z")
REOPT_PASSES = 3
SCAN_RANGES_UM = {"x": 2.0, "y": 2.0, "z": 10.0}
SCAN_PIXELS = {"x": 100, "y": 100, "z": 100}
SCAN_TIMES_S = {"x": 3.0, "y": 3.0, "z": 3.0}

# Orchestration timing
SHORT_EXPERIMENT_DURATION_S = 20 * 60
POST_EXPERIMENT_PAUSE_S = 2.0
LASER_SETTLE_S = 1
REOPT_PAUSE_AFTER_S = 1

# Optional software limits; set to None if not needed
AXIS_LIMITS_UM = {
    "x": (-40.0, 40.0),
    "y": (-40.0, 40.0),
    "z": (-40.0, 40.0),
}


def get_scan_controller():

    # Load the same default qdlscan config used by the GUI app
    config_pkg = "qdlutils.applications.qdlscan.config_files"
    yaml_name = "qdlscan_base.yaml"
    yaml_path = importlib.resources.files(config_pkg).joinpath(yaml_name)

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    app_name = list(config.keys())[0]
    app_cfg = config[app_name]

    hw = app_cfg["ApplicationController"]["hardware"]

    def build_component(component_name):
        c = app_cfg[component_name]
        module = importlib.import_module(c["import_path"])
        cls = getattr(module, c["class_name"])
        obj = cls()
        obj.configure(c["configure"])
        return obj

    counter = build_component(hw["counter"])
    x_axis = build_component(hw["x_axis_control"])
    y_axis = build_component(hw["y_axis_control"])
    z_axis = build_component(hw["z_axis_control"])

    controller_cfg = app_cfg["ApplicationController"]["configure"]
    module = importlib.import_module(app_cfg["ApplicationController"]["import_path"])
    cls = getattr(module, app_cfg["ApplicationController"]["class_name"])

    controller = cls(
        x_axis_controller=x_axis,
        y_axis_controller=y_axis,
        z_axis_controller=z_axis,
        counter_controller=counter,
        **controller_cfg,
    )
    return controller


def run_short_qick_experiment(cycle_index: int, duration_seconds: float) -> None:
    """
    Replace this with your qick experiment call.

    Example shape:
    - Construct experiment program
    - Acquire for the desired short runtime
    - Save data and return
    """
    print(f"[qick] cycle={cycle_index} start")
    time.sleep(duration_seconds)
    print(f"[qick] cycle={cycle_index} done")


def laser_on() -> None:
    """
    Replace this with your hardware-specific laser enable call.
    """
    print("[laser] on")


def laser_off() -> None:
    """
    Replace this with your hardware-specific laser disable call.
    """
    print("[laser] off")


def run_reoptimization_step(
    reoptimizer: Reoptimizer,
    n_passes: int = REOPT_PASSES,
) -> List[Tuple[int, str, AxisOptimizationResult]]:
    """
    Runs repeated ordered optimization passes around the current position.

    Example with n_passes=3 and AXIS_ORDER=("x", "y", "z"):
    x,y,z,x,y,z,x,y,z
    """
    all_results: List[Tuple[int, str, AxisOptimizationResult]] = []

    for pass_idx in range(n_passes):
        for axis in AXIS_ORDER:
            result = reoptimizer.optimize_axis(
                axis=axis,
                scan_range=SCAN_RANGES_UM[axis],
                n_pixels=SCAN_PIXELS[axis],
                scan_time=SCAN_TIMES_S[axis],
                move_to_optimum=True,
            )
            all_results.append((pass_idx + 1, axis, result))

    return all_results


def print_reopt_results(results: List[Tuple[int, str, AxisOptimizationResult]], cycle_index: int) -> None:
    print(f"[reopt] cycle={cycle_index}")
    for pass_index, axis, r in results:
        peak_idx = int(np.argmax(r.data_count_rates))
        peak_rate = float(r.data_count_rates[peak_idx])
        peak_pos = float(r.data_positions[peak_idx])

        start_idx = int(np.argmin(np.abs(r.data_positions - r.start_position)))
        final_idx = int(np.argmin(np.abs(r.data_positions - r.final_position)))
        start_rate_est = float(r.data_count_rates[start_idx])
        final_rate_est = float(r.data_count_rates[final_idx])
        delta_rate = final_rate_est - start_rate_est

        print(
            f"  pass={pass_index} axis={axis} start={r.start_position:.4f} um "
            f"final={r.final_position:.4f} um "
            f"method={r.fit_method} fit_success={r.fit_success}"
        )
        print(
            f"    counts: start~{start_rate_est:.1f} cts/s "
            f"final~{final_rate_est:.1f} cts/s "
            f"delta={delta_rate:+.1f} cts/s "
            f"peak={peak_rate:.1f} cts/s @ {peak_pos:.4f} um"
        )


# ---------------------------------------------------------------------------
# Minimal integration pattern for an experiment loop
# ---------------------------------------------------------------------------

def run_experiment_reopt_loop(total_cycles: int = 6) -> None:
    """
    Workflow:
    - You manually optimize before running this loop.
    - For each cycle: short qick experiment -> laser on -> reoptimize -> laser off.
    """
    controller = get_scan_controller()

    reoptimizer = Reoptimizer(
        application_controller=controller,
        axis_limits=AXIS_LIMITS_UM,
        optimization_method="gaussian",
    )

    for i in range(total_cycles):
        run_short_qick_experiment(
            cycle_index=i,
            duration_seconds=SHORT_EXPERIMENT_DURATION_S,
        )

        time.sleep(POST_EXPERIMENT_PAUSE_S)

        laser_on()
        time.sleep(LASER_SETTLE_S)

        try:
            results = run_reoptimization_step(reoptimizer)
            print_reopt_results(results, cycle_index=i)
        finally:
            # Always force laser-off even if reoptimization fails.
            laser_off()

        time.sleep(REOPT_PAUSE_AFTER_S)

    print("[loop] complete")


if __name__ == "__main__":
    run_experiment_reopt_loop(total_cycles=6)
