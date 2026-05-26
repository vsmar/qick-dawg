"""
Default reoptimization hook for chunked experiments.

Chunked mode uses this automatically via maybe_run_chunked_mode.
Single-run mode never calls this hook.
"""

from __future__ import annotations

from copy import copy
import importlib
import importlib.resources
import json
import os
import subprocess
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import qickdawg as qd
import yaml


DEFAULT_REOPT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_reopt.yaml"


def _bootstrap_qdlutils_import() -> None:
    try:
        import qdlutils  # type: ignore[import-not-found]  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    candidates = []

    env_src = os.environ.get("QDLUTILS_SRC")
    if env_src:
        # Allow QDLUTILS_SRC to point either at the parent 'src' directory
        # or directly at the 'qdlutils' package directory.
        p = Path(os.path.expandvars(os.path.expanduser(env_src)))
        candidates.append(p)
        if p.name == "qdlutils":
            candidates.append(p.parent)

    here = Path(__file__).resolve()
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "qdl-utils" / "src")

    # Common developer location on this machine
    home_qdl = Path.home() / "qdl-utils" / "src"
    candidates.append(home_qdl)

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


def _find_external_reopt_python() -> Optional[Path]:
    """Return a Python executable that can import qdlutils+nidaqmx, if configured."""
    env_python = os.environ.get("REOPT_PYTHON_EXE") or os.environ.get("QDLUTILS_PYTHON")
    if env_python:
        candidate = Path(os.path.expandvars(os.path.expanduser(env_python)))
        if candidate.exists():
            return candidate

    default_candidate = Path.home() / "qdl-utils" / "vqlm-home-venv" / "Scripts" / "python.exe"
    if default_candidate.exists():
        return default_candidate

    return None


def _run_reopt_in_external_python(
    *,
    python_exe: Path,
    initial_position_um: Tuple[float, float, float],
    axis_order: Tuple[str, str, str],
    scan_ranges_um: Dict[str, float],
    scan_pixels: Dict[str, int],
    scan_times_s: Dict[str, float],
) -> Dict[str, Any]:
    """Run the qdlutils reoptimizer in a separate Python interpreter."""
    payload = {
        "initial_position_um": [float(v) for v in initial_position_um],
        "axis_order": list(axis_order),
        "scan_ranges_um": {str(k): float(v) for k, v in scan_ranges_um.items()},
        "scan_pixels": {str(k): int(v) for k, v in scan_pixels.items()},
        "scan_times_s": {str(k): float(v) for k, v in scan_times_s.items()},
    }

    script = r'''
import importlib
import importlib.resources
import json
import sys

import numpy as np
import yaml

from qdlutils.applications.qdlscan.reoptimizer import Reoptimizer


def build_scan_controller(initial_position_um):
    config_pkg = "qdlutils.applications.qdlscan.config_files"
    yaml_name = "qdlscan_base.yaml"
    yaml_path = importlib.resources.files(config_pkg).joinpath(yaml_name)

    with open(yaml_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

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

    for axis_name, axis_pos in zip(("x", "y", "z"), initial_position_um):
        controller.set_axis(axis=axis_name, position=float(axis_pos))

    return controller


def main():
    payload = json.loads(sys.stdin.read())
    controller = build_scan_controller(tuple(payload["initial_position_um"]))
    reoptimizer = Reoptimizer(
        application_controller=controller,
        axis_limits={"x": (-40.0, 40.0), "y": (-40.0, 40.0), "z": (-40.0, 40.0)},
        optimization_method="gaussian",
    )

    results = {}
    fit_success_count = 0
    total_axes = 0

    for axis in payload["axis_order"]:
        result = reoptimizer.optimize_axis(
            axis=axis,
            scan_range=float(payload["scan_ranges_um"][axis]),
            n_pixels=int(payload["scan_pixels"][axis]),
            scan_time=float(payload["scan_times_s"][axis]),
            move_to_optimum=True,
        )
        total_axes += 1
        fit_success_count += int(bool(result.fit_success))
        results[axis] = {
            "fit_success": bool(result.fit_success),
            "final_position": float(result.final_position),
            "fit_method": result.fit_method,
            "message": result.message,
        }

    pos = reoptimizer.application_controller.get_position()
    fit_fraction = (fit_success_count / total_axes) if total_axes > 0 else 0.0
    print(json.dumps({
        "status": "success" if fit_fraction > 0.0 else "weak",
        "fit_success_fraction": fit_fraction,
        "x_um": float(pos[0]),
        "y_um": float(pos[1]),
        "z_um": float(pos[2]),
        "axis_results": results,
    }))


if __name__ == "__main__":
    main()
'''

    completed = subprocess.run(
        [str(python_exe), "-c", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"External reoptimizer failed with exit code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )

    output = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "{}"
    return json.loads(output)


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


def load_reopt_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load the reoptimization YAML configuration."""
    path = Path(config_path) if config_path is not None else DEFAULT_REOPT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def save_reopt_config(config_path: Path | str, payload: Dict[str, Any]) -> None:
    """Write the reoptimization YAML configuration."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def create_reopt_callback(
    *,
    initial_position_um: Tuple[float, float, float],
    config_path: Path | str | None = None,
    axis_order: Tuple[str, str, str] = ("x", "y", "z"),
    scan_ranges_um: Optional[Dict[str, float]] = None,
    scan_pixels: Optional[Dict[str, int]] = None,
    scan_times_s: Optional[Dict[str, float]] = None,
) -> Any:
    """Create a chunk-reopt callback bound to a fixed initial piezo position."""

    config_path_obj = Path(config_path) if config_path is not None else DEFAULT_REOPT_CONFIG_PATH
    reopt_cfg = load_reopt_config(config_path_obj)

    config_section = reopt_cfg.get("reoptimization", {}) if isinstance(reopt_cfg.get("reoptimization", {}), dict) else {}

    stored_position = config_section.get("piezo_initial_position_um")
    if isinstance(stored_position, (list, tuple)) and len(stored_position) == 3:
        try:
            initial_position_um = (float(stored_position[0]), float(stored_position[1]), float(stored_position[2]))
        except Exception:
            pass

    file_scan_ranges = config_section.get("scan_ranges_um") if isinstance(config_section.get("scan_ranges_um"), dict) else None
    file_scan_pixels = config_section.get("scan_pixels") if isinstance(config_section.get("scan_pixels"), dict) else None
    file_scan_times = config_section.get("scan_times_s") if isinstance(config_section.get("scan_times_s"), dict) else None

    if scan_ranges_um is None and file_scan_ranges is not None:
        scan_ranges_um = {str(k): float(v) for k, v in file_scan_ranges.items()}
    if scan_pixels is None and file_scan_pixels is not None:
        scan_pixels = {str(k): int(v) for k, v in file_scan_pixels.items()}
    if scan_times_s is None and file_scan_times is not None:
        scan_times_s = {str(k): float(v) for k, v in file_scan_times.items()}

    def _callback(context: Dict[str, Any]) -> Dict[str, Any]:
        ranges = scan_ranges_um or {"x": 2.0, "y": 2.0, "z": 10.0}
        pixels = scan_pixels or {"x": 100, "y": 100, "z": 100}
        times_s = scan_times_s or {"x": 5.0, "y": 5.0, "z": 5.0}

        config_obj = context.get("config")
        if config_obj is not None:
            _laser_on(config_obj)
            time.sleep(1.0)

        try:
            try:
                reoptimizer = _get_or_create_reoptimizer(initial_position_um=initial_position_um)
            except ModuleNotFoundError:
                external_python = _find_external_reopt_python()
                if external_python is None:
                    return {
                        "status": "unavailable",
                        "error": "Reoptimization requires qdlutils+nidaqmx, but no compatible interpreter was found.",
                    }

                external_result = _run_reopt_in_external_python(
                    python_exe=external_python,
                    initial_position_um=initial_position_um,
                    axis_order=axis_order,
                    scan_ranges_um=ranges,
                    scan_pixels=pixels,
                    scan_times_s=times_s,
                )
                return external_result

            fit_success_count = 0
            total_axes = 0

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

        current_payload = load_reopt_config(config_path_obj)
        current_payload.setdefault("reoptimization", {})
        current_payload["reoptimization"]["piezo_initial_position_um"] = [float(pos[0]), float(pos[1]), float(pos[2])]
        current_payload["reoptimization"]["last_reopt_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        current_payload["reoptimization"]["last_status"] = "success" if fit_fraction > 0.0 else "weak"
        save_reopt_config(config_path_obj, current_payload)

        return {
            "status": "success" if fit_fraction > 0.0 else "weak",
            "fit_success_fraction": fit_fraction,
            "x_um": float(pos[0]),
            "y_um": float(pos[1]),
            "z_um": float(pos[2]),
        }

    return _callback
