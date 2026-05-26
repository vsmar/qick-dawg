"""Plot helpers for experiment runs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from experiments.helpers.config import add_unit_pair_expansions, collect_required_cfg_attrs
from experiments.helpers.h5_namespace import load_aggregated_h5_namespace


def _to_plot_value(value: Any) -> Any:
    """Convert values into compact, plot-friendly scalar/string forms."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        converted = float(value)
        if np.isnan(converted) or np.isinf(converted):
            return None
        return converted
    if isinstance(value, (list, tuple)):
        if len(value) <= 4:
            return ",".join(str(_to_plot_value(item)) for item in value)
        return f"len={len(value)}"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_plot_value(value.item())
        if value.size <= 4:
            return ",".join(str(_to_plot_value(item)) for item in value.reshape(-1).tolist())
        return f"shape={value.shape}"
    return value


def build_plot_metadata(
    *,
    program_class: Any,
    config_obj: Any,
    base_metadata: Optional[Mapping[str, Any]] = None,
    cfg_prefix: str = "cfg.",
) -> Dict[str, Any]:
    """Merge caller metadata with expanded required_cfg values."""
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
        if converted is not None:
            merged[f"{cfg_prefix}{key}"] = converted

    return merged


def build_standard_title(
    *,
    experiment_label: str,
    sequence_label: str,
    run_id: str,
    suffix: Optional[str] = None,
) -> str:
    """Construct a standardized plot title containing the run identifier."""
    parts = [experiment_label.strip(), sequence_label.strip(), f"run={run_id}"]
    if suffix:
        parts.append(str(suffix).strip())
    return " | ".join(part for part in parts if part)


def make_chunked_plot_callback(
    plotter: Callable[..., Any],
    *,
    config: Any,
    plot_kwargs: Optional[Dict[str, Any]] = None,
):
    """Create a callback that plots from a unified chunked HDF5 file."""
    plot_kwargs = dict(plot_kwargs or {})

    def _callback(output_h5_path):
        proxy = load_aggregated_h5_namespace(output_h5_path)
        return plotter(proxy, cfg=config, show=False, **plot_kwargs)

    return _callback
