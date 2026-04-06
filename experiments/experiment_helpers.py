"""
Shared helpers for experiment scripts.

This module centralizes:
- Plot metadata assembly (including required_cfg parameters)
- Standardized plot title formatting with run identifier
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config import add_unit_pair_expansions, collect_required_cfg_attrs


def _to_plot_value(value: Any) -> Any:
    """Convert values into compact, plot-friendly scalar/string forms."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, (list, tuple)):
        if len(value) <= 4:
            return ",".join(str(_to_plot_value(v)) for v in value)
        return f"len={len(value)}"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_plot_value(value.item())
        if value.size <= 4:
            return ",".join(str(_to_plot_value(v)) for v in value.reshape(-1).tolist())
        return f"shape={value.shape}"
    return value


def build_plot_metadata(
    *,
    program_class: Any,
    config_obj: Any,
    base_metadata: Optional[Mapping[str, Any]] = None,
    cfg_prefix: str = "cfg.",
) -> Dict[str, Any]:
    """
    Merge caller metadata with expanded required_cfg values.

    Parameters
    ----------
    program_class:
        Experiment class that defines required_cfg.
    config_obj:
        Active NVConfiguration object for the run.
    base_metadata:
        User-provided metadata first in output ordering.
    cfg_prefix:
        Prefix used for required_cfg keys in plot annotations.
    """
    merged: Dict[str, Any] = {}
    if base_metadata:
        for key, value in base_metadata.items():
            converted = _to_plot_value(value)
            if converted is not None:
                merged[str(key)] = converted

    required_keys = getattr(program_class, "required_cfg", [])
    required_attrs = collect_required_cfg_attrs(config_obj, required_keys)
    required_attrs = add_unit_pair_expansions(required_attrs, config_obj)

    for key in sorted(required_attrs.keys()):
        converted = _to_plot_value(required_attrs[key])
        if converted is None:
            continue
        merged[f"{cfg_prefix}{key}"] = converted

    return merged


def build_standard_title(
    *,
    experiment_label: str,
    sequence_label: str,
    run_id: str,
    suffix: Optional[str] = None,
) -> str:
    """Construct standardized plot title containing sequence and run identifier."""
    parts = [experiment_label.strip(), sequence_label.strip(), f"run={run_id}"]
    if suffix:
        parts.append(str(suffix).strip())
    return " | ".join(p for p in parts if p)


def load_reopt_callback(module_name: Optional[str], function_name: str):
    """Load a user-provided reoptimization callback by module/function name."""
    if not module_name:
        return None
    module = importlib.import_module(module_name)
    callback = getattr(module, function_name)
    if not callable(callback):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return callback


def maybe_run_chunked_mode(
    *,
    run_mode: str,
    program_class: Any,
    cfg_dict: Dict[str, Any],
    build_config_for_chunk: Any,
    output_dir: Any,
    experiment_name: str,
    target_total_reps: int,
    chunk_reps: int,
    acquire_progress: bool,
    piezo_initial_position_um: Optional[tuple[float, float, float]],
    combine_filename: str,
    combine_fn: Any = None,
) -> bool:
    """
    Run chunked mode and return True when caller should exit script early.

    Returns False when run_mode is "single".
    """
    if run_mode == "single":
        return False
    if run_mode != "chunked":
        raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

    from chunked_runner import run_chunked_experiment

    from reopt_hook import create_reopt_callback

    if piezo_initial_position_um is None:
        raise ValueError(
            "Chunked mode requires PIEZO_INITIAL_POSITION_UM=(x, y, z) to avoid piezo reset-at-start behavior."
        )

    reopt_callback = create_reopt_callback(initial_position_um=piezo_initial_position_um)

    run_summary = run_chunked_experiment(
        program_class=program_class,
        cfg_dict=cfg_dict,
        build_config_for_chunk=build_config_for_chunk,
        output_dir=output_dir,
        experiment_name=experiment_name,
        target_total_reps=int(target_total_reps),
        chunk_reps=int(chunk_reps),
        acquire_progress=bool(acquire_progress),
        reopt_every_n_chunks=1,
        reopt_callback=reopt_callback,
        enable_combine_at_end=True,
        combine_filename=combine_filename,
        combine_fn=combine_fn,
    )
    print(f"[{experiment_name}] Chunked run complete: {run_summary}")
    return True
