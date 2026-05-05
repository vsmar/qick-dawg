"""
Shared helpers for experiment scripts.

This module centralizes:
- Plot metadata assembly (including required_cfg parameters)
- Standardized plot title formatting with run identifier
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from config import add_unit_pair_expansions, collect_required_cfg_attrs, build_nv_config


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


def load_aggregated_h5_namespace(h5_path: Path) -> SimpleNamespace:
    """Load /aggregated/data from a unified HDF5 file into an attribute namespace."""
    h5py = importlib.import_module("h5py")

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find HDF5 file: {h5_path.resolve()}")

    proxy = SimpleNamespace()
    with h5py.File(h5_path, "r") as hf:
        if "aggregated" not in hf or "data" not in hf["aggregated"]:
            raise ValueError(f"{h5_path.name} does not contain /aggregated/data")

        data_grp = hf["aggregated/data"]
        for key, dataset in data_grp.items():
            setattr(proxy, key, np.asarray(dataset[()]))

    return proxy


def make_chunked_plot_callback(
    plotter: Callable[..., Any],
    *,
    config: Any,
    plot_kwargs: Optional[Dict[str, Any]] = None,
):
    """Create a reusable callback that plots from a unified chunked HDF5 file."""
    plot_kwargs = dict(plot_kwargs or {})

    def _callback(output_h5_path: Path):
        proxy = load_aggregated_h5_namespace(output_h5_path)
        return plotter(proxy, cfg=config, show=False, **plot_kwargs)

    return _callback


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
    plot_callback: Optional[Any] = None,
    plot_filename: str = "aggregated_plot.png",
    combine_filename: str = "combined.h5",  # Deprecated, ignored
    combine_fn: Any = None,  # Deprecated, ignored
) -> bool:
    """
    Run chunked mode and return True when caller should exit script early.

    Returns False when run_mode is "single".
    
    Notes
    -----
    Output is a single timestamped HDF5 file: {output_dir}/{experiment_name}_{run_id}.h5
    This file contains:
      - /metadata/ with run-level attributes
      - /aggregated/data/ with weighted-average signal/reference/contrast (computed in real-time)
      - /chunks/chunk_NNN/ with per-chunk data and metadata
    
    The combine_filename and combine_fn parameters are deprecated and ignored.
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
        plot_callback=plot_callback,
        plot_filename=plot_filename,
        enable_combine_at_end=False,  # No longer needed; aggregation is automatic
        combine_filename=combine_filename,  # Ignored
        combine_fn=None,  # Ignored
    )
    print(f"[{experiment_name}] Chunked run complete: {run_summary}")
    return True


def build_common_config(
    cfg: Dict[str, Any],
    reps: int,
    *,
    transition: Optional[str] = None,
    override_freq_mhz: Optional[float] = None,
    override_mw_gain: Optional[float] = None,
    override_mw_pi_ftsamp: Optional[int] = None,
    override_mw_pi_ftns: Optional[float] = None,
    get_reference: bool = True,
) -> tuple:
    """Build a common NVConfiguration with shared override handling.

    This centralizes the common logic used across experiment scripts:
    - selects active transition from config when not provided
    - applies overrides for mw frequency, gain, and pi pulse
    - sets `reps` and `get_reference`

    Returns (config, active_transition, pi_source_str)
    """
    cfg = dict(cfg)
    config = build_nv_config(cfg)

    active_transition = transition or cfg["calibration"]["default_transition"]
    t = cfg["calibration"][active_transition]

    config.mw_fMHz = override_freq_mhz if override_freq_mhz is not None else t["mw_fMHz"]
    config.mw_gain = override_mw_gain if override_mw_gain is not None else t["mw_gain"]

    if override_mw_pi_ftsamp is not None and override_mw_pi_ftns is not None:
        raise ValueError("Set only one of override_mw_pi_ftsamp or override_mw_pi_ftns.")

    if override_mw_pi_ftsamp is not None:
        config.mw_pi_ftsamp = int(override_mw_pi_ftsamp)
        pi_source = f"override_mw_pi_ftsamp={config.mw_pi_ftsamp}"
    elif override_mw_pi_ftns is not None:
        config.mw_pi_ftns = float(override_mw_pi_ftns)
        pi_source = f"override_mw_pi_ftns={config.mw_pi_ftns}"
    else:
        if t.get("mw_pi_ftsamp") is None:
            # leave pi unspecified; caller may handle missing pi
            pi_source = "calibration:missing_pi"
        else:
            config.mw_pi_ftsamp = int(t["mw_pi_ftsamp"])
            pi_source = f"calibration.{active_transition}.mw_pi_ftsamp={config.mw_pi_ftsamp}"

    config.reps = int(reps)
    config.get_reference = bool(get_reference)

    return config, active_transition, pi_source
