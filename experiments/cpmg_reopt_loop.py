"""
cpmg_reopt_loop.py

Long-run CPMG orchestrator with periodic qdlscan reoptimization.

Design goals:
1) Save each CPMG chunk as an independent HDF5 file to avoid losing whole runs.
2) Record reoptimization outcomes and simple success metrics after each chunk.
3) Keep integration simple with existing script-style experiments.

Typical usage:
    python experiments/cpmg_reopt_loop.py
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
from copy import copy
from datetime import UTC, datetime
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qickdawg as qd
from qickdawg.nvtestsuite.cpmg_xy_subnano_fine_res import CPMGXYFineRes

from config import (
    build_nv_config,
    connect,
    load_config,
    save_experiment_hdf5,
)
from cpmg_combine_utils import combine_chunk_hdf5_files
from plotting_utils import plot_contrast_twin, plot_debug_traces


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

    candidates: List[Path] = []

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


# =============================================================================
# CPMG chunk parameters (same spirit as experiments/cpmg.py)
# =============================================================================
TAU_START_FTNS = 200.0
TAU_STOP_FTNS = 13_000.0
TAU_DELTA_FTNS = 5

N_CPMG = 32
GET_REFERENCE = False # True
TRANSITION = None

OVERRIDE_FREQ_MHZ = None
OVERRIDE_MW_GAIN = None
OVERRIDE_MW_PI2_FTSAMP = None
OVERRIDE_MW_PI2_FTNS = None

# If TARGET_TOTAL_REPS is not divisible by CHUNK_REPS, the final chunk uses remainder reps.
TARGET_TOTAL_REPS = 27_000
CHUNK_REPS = 1_000


def _env_flag_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Respect shell-level progress toggles for long terminal runs.
ACQUIRE_PROGRESS = not _env_flag_true("TQDM_DISABLE", default=False)
ACQUIRE_MAX_RETRIES = 5
ACQUIRE_RETRY_DELAY_S = 5.0

# =============================================================================
# Reoptimization parameters
# =============================================================================
REOPT_EVERY_N_CHUNKS = 1
REOPT_PASSES = 2
AXIS_ORDER = ("x", "y", "z")
SCAN_RANGES_UM = {"x": 2.0, "y": 2.0, "z": 10.0}
SCAN_PIXELS = {"x": 100, "y": 100, "z": 100}
SCAN_TIMES_S = {"x": 5.0, "y": 5.0, "z": 5.0}
AXIS_LIMITS_UM = {
    "x": (-40.0, 40.0),
    "y": (-40.0, 40.0),
    "z": (-40.0, 40.0),
}

# Startup and move-safety controls for reoptimization
INITIAL_POSITION_UM: Optional[Tuple[float, float, float]] = (-2.298223, -0.038856, -4.01095)
ALLOW_UNSEEDED_REOPT_START = False
REQUIRE_FIT_SUCCESS_FOR_AXIS_MOVE = True
MIN_AXIS_IMPROVEMENT_CTS_S = 0.0

# Reopt quality gate (for metadata only; acquisition can continue despite failure)
MIN_FIT_SUCCESS_FRACTION = 0.66
MIN_MEAN_DELTA_CTS = -200.0

# Orchestration timing
POST_EXPERIMENT_PAUSE_S = 2.0
LASER_SETTLE_S = 1
REOPT_PAUSE_AFTER_S = 1

# Safety behavior
MAX_CONSECUTIVE_REOPT_FAILURES = 3
ABORT_IF_REOPT_CONTROLLER_UNAVAILABLE = True

# Logging/output behavior
VERBOSE_LOGS = False
ENABLE_COMBINE_AT_END = True

# =============================================================================
# Output locations
# =============================================================================

OUTPUT_ROOT = Path(__file__).parent.parent / "data" / "cpmg_reopt"
MANIFEST_FILENAME = "run_manifest.jsonl"
SUMMARY_FILENAME = "run_summary.csv"
COMBINED_FILENAME = "cpmg_xy_fine_res_combined.h5"
PLOTS_DIRNAME = "plots"


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        value = s
    try:
        return float(value)
    except Exception:
        return None


def _is_retryable_acquire_error(exc: Exception) -> bool:
    """Heuristic for transient acquisition/poll failures."""
    msg = str(exc)
    if "NoneType" in msg and "subscriptable" in msg:
        return True
    if "data size mismatch" in msg:
        return True
    if "got too much data" in msg:
        return True
    return False


def _load_seed_position_from_previous_run(output_root: Path) -> Optional[Tuple[float, float, float]]:
    """
    Load most recent saved reoptimization XYZ from prior run summaries.

    This provides startup seeding when DAQ position readback channels are unavailable.
    """
    if not output_root.exists():
        return None

    run_dirs = [
        p for p in output_root.iterdir()
        if p.is_dir() and p.name.startswith("cpmg_reopt_run_")
    ]
    run_dirs.sort(reverse=True)

    for run_dir in run_dirs:
        summary_path = run_dir / SUMMARY_FILENAME
        if not summary_path.exists():
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue

        for row in reversed(rows):
            x = _to_optional_float(row.get("reopt_x_um"))
            y = _to_optional_float(row.get("reopt_y_um"))
            z = _to_optional_float(row.get("reopt_z_um"))
            if x is not None and y is not None and z is not None:
                if VERBOSE_LOGS:
                    print(
                        "[reopt] loaded seed position from previous run summary: "
                        f"x={x:.4f}, y={y:.4f}, z={z:.4f} um "
                        f"({summary_path})"
                    )
                return (x, y, z)

    return None


# =============================================================================
# Integration hooks (edit these for your setup)
# =============================================================================

def get_scan_controller():
    """
    Return your configured qdlscan ScanController instance.

    Replace this with lab-specific controller construction/retrieval.
    """

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

    # Read current physical positions before creating ScanController.
    # ScanController.__init__ zeroes each axis to establish internal state,
    # so we restore these values immediately afterward.
    measured_start_pos: Optional[Tuple[float, float, float]] = None
    try:
        measured_start_pos = (
            float(x_axis.get_current_position()),
            float(y_axis.get_current_position()),
            float(z_axis.get_current_position()),
        )
        if VERBOSE_LOGS:
            print(
                "[reopt] measured start position before controller init: "
                f"x={measured_start_pos[0]:.4f}, y={measured_start_pos[1]:.4f}, z={measured_start_pos[2]:.4f} um"
            )
    except Exception as exc:
        if VERBOSE_LOGS:
            print(f"[reopt] could not read current stage position before init: {exc}")

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

    auto_seed_pos = _load_seed_position_from_previous_run(OUTPUT_ROOT)
    restore_pos = measured_start_pos if measured_start_pos is not None else (INITIAL_POSITION_UM or auto_seed_pos)
    if restore_pos is not None:
        for axis_name, axis_pos in zip(("x", "y", "z"), restore_pos):
            controller.set_axis(axis=axis_name, position=float(axis_pos))
        if VERBOSE_LOGS:
            print(
                "[reopt] restored controller start position: "
                f"x={restore_pos[0]:.4f}, y={restore_pos[1]:.4f}, z={restore_pos[2]:.4f} um"
            )
    else:
        message = (
            "No measured/manual start position available; controller may remain near startup zero. "
            "Set INITIAL_POSITION_UM=(x,y,z) or configure axis read_channel in qdlscan config."
        )
        if not ALLOW_UNSEEDED_REOPT_START:
            raise RuntimeError(message)
        print(f"[reopt] warning: {message}")

    return controller

def laser_on(config_obj) -> None:
    """Enable laser for reoptimization scans."""
    qd.laser_on(copy(config_obj))


def laser_off(config_obj) -> None:
    """Disable laser after reoptimization scans."""
    qd.laser_off(copy(config_obj))


# =============================================================================
# Helpers
# =============================================================================

def _to_json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays to native JSON-serializable values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _append_manifest_line(manifest_path: Path, payload: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_json_safe(payload), sort_keys=True) + "\n")


def _write_summary_csv(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = sorted({k for row in rows for k in row.keys()})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_json_safe(row))


def _data_get_array(data: Any, names: List[str]) -> Optional[np.ndarray]:
    """Read a 1D array from an analyzed data object via attrs or key lookup."""
    for name in names:
        value = getattr(data, name, None)
        if value is None and hasattr(data, "keys"):
            try:
                if name in data.keys():
                    value = data[name]
            except Exception:
                value = None

        if value is not None:
            arr = np.asarray(value, dtype=float).reshape(-1)
            return arr
    return None


def _extract_tau_axis_ftns(data: Any) -> Optional[np.ndarray]:
    return _data_get_array(data, ["tau_ftns"])


def _extract_chunk_cts_series(data: Any) -> Dict[str, np.ndarray]:
    series: Dict[str, np.ndarray] = {}
    mapping = {
        "signal1_cts_s": ["signal1_cts_s", "signal1"],
        "signal2_cts_s": ["signal2_cts_s", "signal2"],
        "reference1_cts_s": ["reference1_cts_s", "reference1"],
        "reference2_cts_s": ["reference2_cts_s", "reference2"],
    }
    for out_name, candidates in mapping.items():
        arr = _data_get_array(data, candidates)
        if arr is not None:
            series[out_name] = arr
    return series


def _save_chunk_cts_plot(
    tau_ftns: np.ndarray,
    chunk_series: Dict[str, np.ndarray],
    cycle_index: int,
    run_id: str,
    out_path: Path,
) -> bool:
    """Save per-chunk debug plot with all available raw channels."""
    traces = {
        "signal1": chunk_series.get("signal1_cts_s"),
        "signal2": chunk_series.get("signal2_cts_s"),
        "reference1": chunk_series.get("reference1_cts_s"),
        "reference2": chunk_series.get("reference2_cts_s"),
        "contrast": None,
    }

    valid_count = 0
    for key in ("signal1", "signal2", "reference1", "reference2"):
        arr = traces[key]
        if arr is None:
            continue
        if arr.shape != tau_ftns.shape:
            traces[key] = None
            continue
        valid_count += 1

    if valid_count == 0:
        return False

    fig, _ = plot_debug_traces(
        tau_ftns,
        traces,
        x_label=r"$\tau$ (ftns)",
        y_label="counts/s",
        title=(
            f"CPMG chunk {cycle_index}: raw cts/s traces | "
            f"$\\pi/2_y - \\{{\\tau - \\pi_{{XY8}} - \\tau\\}}\\times N - \\pi/2_{{-y}}$ "
            f"| phase XYXYYXYX"
        ),
        metadata={
            "run_id": run_id,
            "cycle": cycle_index,
            "units": "cts/s",
            "sequence": "pi/2_y - {tau - pi_xy8 - tau}xN - pi/2_-y",
            "xy8_phase_order": "XYXYYXYX",
        },
        metadata_position="bottom",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _save_cumulative_plot(
    tau_ftns: np.ndarray,
    signal_avg: np.ndarray,
    denominator_avg: np.ndarray,
    cycle_index: int,
    run_id: str,
    out_path: Path,
    denominator_label: str,
) -> None:
    """Save cumulative weighted-average contrast-focused twin plot."""
    signal_over_steady_state = signal_avg / np.clip(denominator_avg, 1e-12, None)

    traces = {
        "signal1": signal_avg,
        "signal2": denominator_avg,
        "reference1": None,
        "reference2": None,
        "contrast": signal_over_steady_state,
    }

    fig, _, _, _ = plot_contrast_twin(
        tau_ftns,
        traces,
        x_label=r"$\tau$ (ftns)",
        title=(
            f"CPMG cumulative through chunk {cycle_index} | "
            f"$\\pi/2_y - \\{{\\tau - \\pi_{{XY8}} - \\tau\\}}\\times N - \\pi/2_{{-y}}$ "
            f"| phase XYXYYXYX"
        ),
        metadata={
            "run_id": run_id,
            "cycle": cycle_index,
            "units": "cts/s",
            "sequence": "pi/2_y - {tau - pi_xy8 - tau}xN - pi/2_-y",
            "xy8_phase_order": "XYXYYXYX",
        },
        metadata_position="bottom",
        raw_alpha=0.3,
        contrast_label="signal/steady_state",
        left_ylabel="Signal/steady_state",
        signal1_label="signal1",
        signal2_label=denominator_label,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_tau_ftns() -> Tuple[float, float, float]:
    if TAU_DELTA_FTNS <= 0:
        raise ValueError("TAU_DELTA_FTNS must be > 0.")
    return float(TAU_START_FTNS), float(TAU_STOP_FTNS), float(TAU_DELTA_FTNS)


def build_cpmg_config_for_chunk(cfg: dict, reps: int) -> Tuple[Any, str]:
    """Build per-chunk NV configuration."""
    config = build_nv_config(cfg)

    active_transition = TRANSITION or cfg["calibration"]["default_transition"]
    t = cfg["calibration"][active_transition]

    config.mw_fMHz = OVERRIDE_FREQ_MHZ if OVERRIDE_FREQ_MHZ is not None else t["mw_fMHz"]
    config.mw_gain = OVERRIDE_MW_GAIN if OVERRIDE_MW_GAIN is not None else t["mw_gain"]

    if OVERRIDE_MW_PI2_FTSAMP is not None:
        if OVERRIDE_MW_PI2_FTNS is not None:
            raise ValueError("Set only one of OVERRIDE_MW_PI2_FTSAMP or OVERRIDE_MW_PI2_FTNS.")
        config.mw_pi2_ftsamp = int(OVERRIDE_MW_PI2_FTSAMP)
    elif OVERRIDE_MW_PI2_FTNS is not None:
        config.mw_pi2_ftns = float(OVERRIDE_MW_PI2_FTNS)
    elif t.get("mw_pi2_ftsamp") is not None:
        config.mw_pi2_ftsamp = int(t["mw_pi2_ftsamp"])
    else:
        raise ValueError(
            "No calibration pi/2 pulse found for this transition. "
            "Set OVERRIDE_MW_PI2_FTSAMP or provide calibration.<transition>.mw_pi2_ftsamp."
        )

    config.n_cpmg = int(N_CPMG)
    config.reps = int(reps)
    config.get_reference = bool(GET_REFERENCE)

    tau_start_ftns, tau_stop_ftns, tau_delta_ftns = _resolve_tau_ftns()

    config.add_linear_sweep(
        "tau",
        "ftns",
        start=tau_start_ftns,
        stop=tau_stop_ftns,
        delta=tau_delta_ftns,
    )

    return config, active_transition


def compute_chunk_signal_metrics(data: Any) -> Dict[str, Optional[float]]:
    """Compute simple chunk-level metrics for post-run quality tracking."""
    signal_raw = getattr(data, "signal1_cts_s", getattr(data, "signal1", None))
    reference_raw = getattr(data, "signal2_cts_s", getattr(data, "signal2", None))

    signal = np.asarray(signal_raw, dtype=float) if signal_raw is not None else None
    reference = np.asarray(reference_raw, dtype=float) if reference_raw is not None else None

    metrics: Dict[str, Optional[float]] = {
        "signal_mean_cts_s": None,
        "signal_median_cts_s": None,
        "reference_mean_cts_s": None,
        "contrast_mean": None,
    }

    if signal is not None and signal.size > 0:
        metrics["signal_mean_cts_s"] = float(np.mean(signal))
        metrics["signal_median_cts_s"] = float(np.median(signal))

    if reference is not None and reference.size > 0:
        metrics["reference_mean_cts_s"] = float(np.mean(reference))
        denom = np.clip(reference, 1e-12, None)
        if signal is not None and signal.shape == reference.shape:
            metrics["contrast_mean"] = float(np.mean(signal / denom))

    return metrics


def run_reoptimization_step(
    reoptimizer: Reoptimizer,
    n_passes: int = REOPT_PASSES,
) -> List[Tuple[int, str, AxisOptimizationResult]]:
    """Run ordered XYZ passes with guarded move acceptance per axis."""
    all_results: List[Tuple[int, str, AxisOptimizationResult]] = []

    for pass_idx in range(n_passes):
        for axis in AXIS_ORDER:
            result = reoptimizer.optimize_axis(
                axis=axis,
                scan_range=SCAN_RANGES_UM[axis],
                n_pixels=SCAN_PIXELS[axis],
                scan_time=SCAN_TIMES_S[axis],
                move_to_optimum=False,
            )

            start_idx = int(np.argmin(np.abs(result.data_positions - result.start_position)))
            final_idx = int(np.argmin(np.abs(result.data_positions - result.final_position)))
            peak_idx = int(np.argmax(result.data_count_rates))

            start_rate = float(result.data_count_rates[start_idx])
            final_rate = float(result.data_count_rates[final_idx])
            peak_rate = float(result.data_count_rates[peak_idx])
            peak_pos = float(result.data_positions[peak_idx])

            accepted_position = float(result.start_position)
            decision_reason = "keep_start"

            if (not REQUIRE_FIT_SUCCESS_FOR_AXIS_MOVE) or bool(result.fit_success):
                if peak_rate >= start_rate + MIN_AXIS_IMPROVEMENT_CTS_S:
                    accepted_position = peak_pos
                    decision_reason = "move_peak"
                elif final_rate >= start_rate + MIN_AXIS_IMPROVEMENT_CTS_S:
                    accepted_position = float(result.final_position)
                    decision_reason = "move_fit"
            else:
                decision_reason = "reject_fit_failed"

            reoptimizer.application_controller.set_axis(axis=axis, position=accepted_position)
            result.final_position = accepted_position
            result.message = f"{result.message} | decision={decision_reason}"

            all_results.append((pass_idx + 1, axis, result))

    return all_results


def summarize_reopt_results(results: List[Tuple[int, str, AxisOptimizationResult]]) -> Dict[str, Any]:
    """
    Summarize optimization quality from per-axis scans.

    Uses nearest sampled points to estimate start/final rates.
    """
    if not results:
        return {
            "axes_total": 0,
            "fit_success_fraction": 0.0,
            "mean_delta_cts_s": None,
            "mean_final_cts_s": None,
            "is_success": False,
        }

    fit_success = 0
    delta_rates: List[float] = []
    final_rates: List[float] = []

    for _pass_index, _axis, r in results:
        if bool(r.fit_success):
            fit_success += 1

        start_idx = int(np.argmin(np.abs(r.data_positions - r.start_position)))
        final_idx = int(np.argmin(np.abs(r.data_positions - r.final_position)))

        start_rate = float(r.data_count_rates[start_idx])
        final_rate = float(r.data_count_rates[final_idx])

        delta_rates.append(final_rate - start_rate)
        final_rates.append(final_rate)

    fit_fraction = fit_success / len(results)
    mean_delta = float(np.mean(delta_rates)) if delta_rates else None
    mean_final = float(np.mean(final_rates)) if final_rates else None

    is_success = (
        fit_fraction >= MIN_FIT_SUCCESS_FRACTION
        and (mean_delta is not None and mean_delta >= MIN_MEAN_DELTA_CTS)
    )

    return {
        "axes_total": len(results),
        "fit_success_fraction": fit_fraction,
        "mean_delta_cts_s": mean_delta,
        "mean_final_cts_s": mean_final,
        "is_success": bool(is_success),
    }


def _print_reopt_detail(cycle_index: int, results: List[Tuple[int, str, AxisOptimizationResult]], summary: Dict[str, Any]) -> None:
    print(f"[reopt] cycle={cycle_index} axes={summary['axes_total']} fit_ok={summary['fit_success_fraction']:.2f}")
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
            f"final={r.final_position:.4f} um method={r.fit_method} fit_success={r.fit_success}"
        )
        print(
            f"    counts: start~{start_rate_est:.1f} cts/s final~{final_rate_est:.1f} cts/s "
            f"delta={delta_rate:+.1f} cts/s peak={peak_rate:.1f} cts/s @ {peak_pos:.4f} um"
        )


def _format_duration_s(seconds: float) -> str:
    """Render seconds as a compact h/m/s string."""
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


# =============================================================================
# Run loop
# =============================================================================

def run_cpmg_reopt_loop() -> Dict[str, Any]:
    if CHUNK_REPS <= 0:
        raise ValueError("CHUNK_REPS must be > 0.")
    if TARGET_TOTAL_REPS <= 0:
        raise ValueError("TARGET_TOTAL_REPS must be > 0.")

    cfg = load_config()
    connect(cfg)

    if VERBOSE_LOGS:
        if INITIAL_POSITION_UM is not None:
            print(
                "[reopt] startup reminder: using INITIAL_POSITION_UM="
                f"({INITIAL_POSITION_UM[0]:.4f}, {INITIAL_POSITION_UM[1]:.4f}, {INITIAL_POSITION_UM[2]:.4f}) um"
            )
            print(
                "[reopt] confirm this seed is current for this session; "
                "update INITIAL_POSITION_UM after long idle periods or realignments."
            )
        else:
            print(
                "[reopt] startup reminder: INITIAL_POSITION_UM is unset; "
                "script will use measured position or previous-run seed if available."
            )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"cpmg_reopt_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / MANIFEST_FILENAME
    summary_path = run_dir / SUMMARY_FILENAME
    plots_dir = run_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        controller = get_scan_controller()
        reoptimizer = Reoptimizer(
            application_controller=controller,
            axis_limits=AXIS_LIMITS_UM,
            optimization_method="gaussian",
        )
        reopt_ready = True
    except Exception as exc:
        if ABORT_IF_REOPT_CONTROLLER_UNAVAILABLE:
            raise RuntimeError(
                "Failed to construct qdlscan Reoptimizer controller. "
                f"Reason: {exc}"
            ) from exc

        print(f"[reopt] disabled: controller unavailable ({exc})")
        reoptimizer = None
        reopt_ready = False

    planned_chunks = int(math.ceil(TARGET_TOTAL_REPS / CHUNK_REPS))
    remaining_reps = int(TARGET_TOTAL_REPS)

    pre_chunk_reopt_status = "manual_initial"
    pre_chunk_reopt_score: Optional[float] = None
    consecutive_reopt_failures = 0

    cycle_rows: List[Dict[str, Any]] = []
    cumulative_tau_ftns: Optional[np.ndarray] = None
    cumulative_signal1_weighted: Optional[np.ndarray] = None
    cumulative_denominator_weighted: Optional[np.ndarray] = None
    cumulative_weight = 0.0

    print(
        f"[loop] run_id={run_id} target_total_reps={TARGET_TOTAL_REPS} "
        f"chunk_reps={CHUNK_REPS} planned_chunks={planned_chunks} "
        f"acquire_progress={ACQUIRE_PROGRESS}"
    )

    run_start_monotonic = time.perf_counter()

    for cycle_index in range(planned_chunks):
        chunk_reps = int(min(CHUNK_REPS, remaining_reps))
        chunk_start_utc = datetime.now(UTC).isoformat(timespec="seconds")
        chunk_start_monotonic = time.perf_counter()

        config, active_transition = build_cpmg_config_for_chunk(cfg, chunk_reps)

        print(
            f"[cpmg] cycle={cycle_index} reps={chunk_reps} "
            f"transition={active_transition} n_cpmg={config.n_cpmg}"
        )

        prog = CPMGXYFineRes(config)
        last_acquire_error: Optional[Exception] = None
        data = None

        for attempt in range(1, ACQUIRE_MAX_RETRIES + 2):
            try:
                data = prog.acquire(progress=ACQUIRE_PROGRESS)
                if attempt > 1:
                    print(f"[acquire] cycle={cycle_index} recovered on attempt {attempt}")
                break
            except Exception as exc:
                last_acquire_error = exc
                retryable = _is_retryable_acquire_error(exc)
                has_next_attempt = attempt < (ACQUIRE_MAX_RETRIES + 1)

                _append_manifest_line(
                    manifest_path,
                    {
                        "event": "acquire_error",
                        "run_id": run_id,
                        "cycle_index": cycle_index,
                        "attempt": attempt,
                        "retryable": retryable,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

                if not (retryable and has_next_attempt):
                    print(
                        f"[acquire] cycle={cycle_index} failed on attempt {attempt}; "
                        f"retryable={retryable}. Aborting run."
                    )
                    raise

                print(
                    f"[acquire] cycle={cycle_index} transient error on attempt {attempt} "
                    f"({type(exc).__name__}: {exc}); retrying in {ACQUIRE_RETRY_DELAY_S:.1f}s"
                )
                time.sleep(ACQUIRE_RETRY_DELAY_S)
                # Rebuild program to avoid stale hardware/readout state between attempts.
                prog = CPMGXYFineRes(config)

        if data is None:
            # Defensive guard: should be unreachable if exceptions are raised correctly.
            raise RuntimeError(f"acquire() returned no data for cycle={cycle_index}") from last_acquire_error

        custom_attrs = {
            "loop_run_id": run_id,
            "loop_cycle_index": cycle_index,
            "loop_total_cycles_planned": planned_chunks,
            "loop_chunk_reps": chunk_reps,
            "loop_target_total_reps": int(TARGET_TOTAL_REPS),
            "loop_pre_chunk_reopt_status": pre_chunk_reopt_status,
            "loop_pre_chunk_reopt_mean_delta_cts_s": pre_chunk_reopt_score,
        }

        out_path, timestamp = save_experiment_hdf5(
            CPMGXYFineRes,
            config,
            cfg,
            data,
            run_dir,
            experiment_name="cpmg_xy_fine_res_chunk",
            custom_attrs=custom_attrs,
        )

        chunk_metrics = compute_chunk_signal_metrics(data)
        tau_ftns = _extract_tau_axis_ftns(data)
        chunk_series = _extract_chunk_cts_series(data)

        chunk_plot_path = plots_dir / f"chunk_cts_cycle_{cycle_index:03d}.png"
        cumulative_plot_path = plots_dir / f"cumulative_cycle_{cycle_index:03d}.png"

        chunk_plot_saved = False
        cumulative_plot_saved = False

        if tau_ftns is not None:
            chunk_plot_saved = _save_chunk_cts_plot(
                tau_ftns=tau_ftns,
                chunk_series=chunk_series,
                cycle_index=cycle_index,
                run_id=run_id,
                out_path=chunk_plot_path,
            )

            s1 = chunk_series.get("signal1_cts_s")
            if bool(config.get_reference):
                denominator = chunk_series.get("signal2_cts_s")
                denominator_label = "signal2"
            else:
                denominator = chunk_series.get("reference1_cts_s")
                denominator_label = "reference1"

            if (
                s1 is not None
                and denominator is not None
                and s1.shape == tau_ftns.shape
                and denominator.shape == tau_ftns.shape
            ):
                if cumulative_tau_ftns is None:
                    cumulative_tau_ftns = tau_ftns.copy()
                    cumulative_signal1_weighted = np.zeros_like(s1, dtype=float)
                    cumulative_denominator_weighted = np.zeros_like(denominator, dtype=float)

                if cumulative_tau_ftns.shape == tau_ftns.shape and np.allclose(cumulative_tau_ftns, tau_ftns):
                    assert cumulative_signal1_weighted is not None
                    assert cumulative_denominator_weighted is not None

                    cumulative_signal1_weighted += chunk_reps * s1
                    cumulative_denominator_weighted += chunk_reps * denominator
                    cumulative_weight += float(chunk_reps)

                    if cumulative_weight > 0:
                        signal_avg = cumulative_signal1_weighted / cumulative_weight
                        denominator_avg = cumulative_denominator_weighted / cumulative_weight
                        _save_cumulative_plot(
                            tau_ftns=cumulative_tau_ftns,
                            signal_avg=signal_avg,
                            denominator_avg=denominator_avg,
                            cycle_index=cycle_index,
                            run_id=run_id,
                            out_path=cumulative_plot_path,
                            denominator_label=denominator_label,
                        )
                        cumulative_plot_saved = True

        cycle_record: Dict[str, Any] = {
            "run_id": run_id,
            "cycle_index": cycle_index,
            "timestamp": timestamp,
            "chunk_start_utc": chunk_start_utc,
            "chunk_end_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunk_reps": chunk_reps,
            "remaining_reps_after_chunk": int(remaining_reps - chunk_reps),
            "chunk_file": str(out_path),
            "chunk_plot_file": str(chunk_plot_path) if chunk_plot_saved else None,
            "cumulative_plot_file": str(cumulative_plot_path) if cumulative_plot_saved else None,
            "reopt_x_um": None,
            "reopt_y_um": None,
            "reopt_z_um": None,
            "pre_chunk_reopt_status": pre_chunk_reopt_status,
            "pre_chunk_reopt_mean_delta_cts_s": pre_chunk_reopt_score,
        }
        cycle_record.update(chunk_metrics)

        _append_manifest_line(manifest_path, {"event": "chunk_saved", **cycle_record})

        remaining_reps -= chunk_reps

        should_reopt_now = (
            reopt_ready
            and (remaining_reps > 0)
            and (REOPT_EVERY_N_CHUNKS > 0)
            and ((cycle_index + 1) % REOPT_EVERY_N_CHUNKS == 0)
        )

        if should_reopt_now:
            time.sleep(POST_EXPERIMENT_PAUSE_S)
            laser_on(config)
            time.sleep(LASER_SETTLE_S)

            try:
                assert reoptimizer is not None
                reopt_results = run_reoptimization_step(reoptimizer)
                reopt_summary = summarize_reopt_results(reopt_results)
                if VERBOSE_LOGS:
                    _print_reopt_detail(cycle_index, reopt_results, reopt_summary)

                cycle_record.update(
                    {
                        "reopt_status": "success" if reopt_summary["is_success"] else "weak",
                        "reopt_fit_success_fraction": reopt_summary["fit_success_fraction"],
                        "reopt_mean_delta_cts_s": reopt_summary["mean_delta_cts_s"],
                        "reopt_mean_final_cts_s": reopt_summary["mean_final_cts_s"],
                    }
                )

                try:
                    pos = reoptimizer.application_controller.get_position()
                    cycle_record["reopt_x_um"] = float(pos[0])
                    cycle_record["reopt_y_um"] = float(pos[1])
                    cycle_record["reopt_z_um"] = float(pos[2])
                except Exception:
                    pass

                pre_chunk_reopt_status = cycle_record["reopt_status"]
                pre_chunk_reopt_score = reopt_summary["mean_delta_cts_s"]
                consecutive_reopt_failures = 0 if reopt_summary["is_success"] else (consecutive_reopt_failures + 1)
            except Exception as exc:
                cycle_record.update(
                    {
                        "reopt_status": "failed",
                        "reopt_error": str(exc),
                        "reopt_fit_success_fraction": None,
                        "reopt_mean_delta_cts_s": None,
                        "reopt_mean_final_cts_s": None,
                    }
                )
                pre_chunk_reopt_status = "failed"
                pre_chunk_reopt_score = None
                consecutive_reopt_failures += 1
                print(f"[reopt] cycle={cycle_index} failed: {exc}")
            finally:
                laser_off(config)
                time.sleep(REOPT_PAUSE_AFTER_S)
        else:
            cycle_record["reopt_status"] = "skipped"

        cycle_rows.append(cycle_record)

        chunk_elapsed_s = time.perf_counter() - chunk_start_monotonic

        elapsed_s = time.perf_counter() - run_start_monotonic
        completed_chunks = cycle_index + 1
        avg_cycle_s = elapsed_s / completed_chunks
        chunks_remaining = planned_chunks - completed_chunks
        eta_remaining_s = max(0.0, avg_cycle_s * chunks_remaining)
        eta_finish_local = datetime.now().timestamp() + eta_remaining_s
        eta_finish_local_text = datetime.fromtimestamp(eta_finish_local).strftime("%Y-%m-%d %H:%M:%S")

        cycle_record["loop_elapsed_s"] = round(elapsed_s, 3)
        cycle_record["loop_avg_cycle_s"] = round(avg_cycle_s, 3)
        cycle_record["chunk_elapsed_s"] = round(chunk_elapsed_s, 3)
        cycle_record["loop_eta_remaining_s"] = round(eta_remaining_s, 3)
        cycle_record["loop_eta_finish_local"] = eta_finish_local_text
        _write_summary_csv(summary_path, cycle_rows)

        print(
            f"[loop] cycle={cycle_index} saved={out_path.name} "
            f"pre_next_reopt_status={pre_chunk_reopt_status} "
            f"chunk_elapsed={_format_duration_s(chunk_elapsed_s)} "
            f"elapsed={_format_duration_s(elapsed_s)} "
            f"eta_remaining={_format_duration_s(eta_remaining_s)} "
            f"eta_finish_local={eta_finish_local_text}"
        )

        if consecutive_reopt_failures >= MAX_CONSECUTIVE_REOPT_FAILURES:
            print(
                f"[loop] aborting: {consecutive_reopt_failures} consecutive reopt failures "
                f"(threshold={MAX_CONSECUTIVE_REOPT_FAILURES})"
            )
            break

    combined_path: Optional[Path] = None
    if ENABLE_COMBINE_AT_END:
        combined_path = run_dir / COMBINED_FILENAME
        combine_chunk_hdf5_files(summary_path, combined_path)

    run_summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "combined_path": str(combined_path) if combined_path is not None else None,
    }
    _append_manifest_line(manifest_path, {"event": "run_complete", **run_summary})

    print(f"[loop] complete run_id={run_id}")
    print(f"[loop] summary_csv={summary_path}")
    if combined_path is not None:
        print(f"[loop] combined_h5={combined_path}")

    return run_summary


if __name__ == "__main__":
    run_cpmg_reopt_loop()
