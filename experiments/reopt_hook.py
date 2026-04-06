"""
Default reoptimization hook for chunked experiments.

Chunked mode uses this automatically via maybe_run_chunked_mode.
Single-run mode never calls this hook.
"""

from __future__ import annotations

from copy import copy
import importlib
import importlib.resources
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import qickdawg as qd
import yaml


def _bootstrap_qdlutils_import() -> None:
    try:
        import qdlutils  # type: ignore[import-not-found]  # noqa: F401

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
        import qdlutils  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import qdlutils. Install it into this environment or set QDLUTILS_SRC."
        ) from exc


_bootstrap_qdlutils_import()


_REOPTIMIZER: Optional[Any] = None


def _build_scan_controller(initial_position_um: Tuple[float, float, float]):
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

    # Explicitly restore known start position to avoid startup reset behavior.
    for axis_name, axis_pos in zip(("x", "y", "z"), initial_position_um):
        controller.set_axis(axis=axis_name, position=float(axis_pos))

    return controller


def _get_or_create_reoptimizer(initial_position_um: Tuple[float, float, float]) -> Any:
    global _REOPTIMIZER
    if _REOPTIMIZER is not None:
        return _REOPTIMIZER

    reoptimizer_module = importlib.import_module("qdlutils.applications.qdlscan.reoptimizer")
    reoptimizer_cls = getattr(reoptimizer_module, "Reoptimizer")

    controller = _build_scan_controller(initial_position_um=initial_position_um)
    _REOPTIMIZER = reoptimizer_cls(
        application_controller=controller,
        axis_limits={"x": (-40.0, 40.0), "y": (-40.0, 40.0), "z": (-40.0, 40.0)},
        optimization_method="gaussian",
    )
    return _REOPTIMIZER


def _laser_on(config_obj: Any) -> None:
    qd.laser_on(copy(config_obj))


def _laser_off(config_obj: Any) -> None:
    qd.laser_off(copy(config_obj))


def create_reopt_callback(
    *,
    initial_position_um: Tuple[float, float, float],
    axis_order: Tuple[str, str, str] = ("x", "y", "z"),
    scan_ranges_um: Optional[Dict[str, float]] = None,
    scan_pixels: Optional[Dict[str, int]] = None,
    scan_times_s: Optional[Dict[str, float]] = None,
) -> Any:
    """Create a chunk-reopt callback bound to a fixed initial piezo position."""

    def _callback(context: Dict[str, Any]) -> Dict[str, Any]:
        ranges = scan_ranges_um or {"x": 2.0, "y": 2.0, "z": 10.0}
        pixels = scan_pixels or {"x": 100, "y": 100, "z": 100}
        times_s = scan_times_s or {"x": 5.0, "y": 5.0, "z": 5.0}

        reoptimizer = _get_or_create_reoptimizer(initial_position_um=initial_position_um)

        config_obj = context.get("config")
        if config_obj is not None:
            _laser_on(config_obj)
            time.sleep(1.0)

        fit_success_count = 0
        total_axes = 0

        try:
            for axis in axis_order:
                result = reoptimizer.optimize_axis(
                    axis=axis,
                    scan_range=ranges[axis],
                    n_pixels=pixels[axis],
                    scan_time=times_s[axis],
                    move_to_optimum=True,
                )
                total_axes += 1
                if bool(result.fit_success):
                    fit_success_count += 1
        finally:
            if config_obj is not None:
                _laser_off(config_obj)

        pos = reoptimizer.application_controller.get_position()
        fit_fraction = (fit_success_count / total_axes) if total_axes > 0 else 0.0
        return {
            "status": "success" if fit_fraction > 0.0 else "weak",
            "fit_success_fraction": fit_fraction,
            "x_um": float(pos[0]),
            "y_um": float(pos[1]),
            "z_um": float(pos[2]),
        }

    return _callback
