"""
Chunked experiment runner with unified HDF5 output.

The runner owns three responsibilities:
- archive each chunk under ``/chunks/chunk_NNNN``
- maintain a weighted ``/summary_data`` aggregate for plotting
- invoke optional reoptimization between chunks
"""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from experiments.helpers.config import normalize_acquired_data
from experiments.helpers.data_manager import DataManager


def _to_json_safe(value: Any) -> Any:
    """Convert values into JSON-serializable primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        if np.isnan(as_float) or np.isinf(as_float):
            return None
        return as_float
    if isinstance(value, np.ndarray):
        return [_to_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_summary_csv(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a compact CSV summary for debugging and traceability."""
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_json_safe(row))


def _append_manifest_line(manifest_path: Path, payload: Dict[str, Any]) -> None:
    """Append a JSONL manifest line."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json_safe(payload), sort_keys=True) + "\n")


def run_chunked_experiment(
    *,
    program_class: Any,
    cfg_dict: Dict[str, Any],
    build_config_for_chunk: Callable[[int], Tuple[Any, Dict[str, Any]]],
    output_dir: Path,
    experiment_name: str,
    target_total_reps: int,
    chunk_reps: int,
    acquire_progress: bool = True,
    reopt_every_n_chunks: int = 0,
    reopt_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    plot_callback: Optional[Callable[[Path], Any]] = None,
    plot_filename: str = "aggregated_plot.png",
    manifest_filename: str = "run_manifest.jsonl",
    summary_filename: str = "run_summary.csv",
    sweep_axis_key: Optional[str] = None,
    metadata_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run an experiment in chunks with a single unified HDF5 output.

    Each chunk is acquired in memory and merged directly into the main file.
    No per-chunk HDF5 files are created.
    """
    if chunk_reps <= 0:
        raise ValueError("chunk_reps must be > 0")
    if target_total_reps <= 0:
        raise ValueError("target_total_reps must be > 0")

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / f".{experiment_name}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_h5_path = run_dir / f"{experiment_name}_{run_id}.h5"
    manifest_path = run_dir / manifest_filename
    summary_path = run_dir / summary_filename

    data_manager = DataManager(output_h5_path, experiment_name, run_id)

    planned_chunks = int(math.ceil(target_total_reps / chunk_reps))
    remaining_reps = int(target_total_reps)
    rows: List[Dict[str, Any]] = []
    plot_path: Optional[Path] = None
    last_reopt_result: Dict[str, Any] = {"status": "not_run"}

    def _merge_metadata(primary: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if primary:
            merged.update(primary)
        if extra:
            merged.update(extra)
        return merged or None

    config, config_context = build_config_for_chunk(chunk_reps)
    initial_metadata = _merge_metadata(config_context, metadata_attrs)
    data_manager.write_initial_metadata(cfg_dict, program_class, config, sweep_axis_key, initial_metadata)

    for chunk_index in range(planned_chunks):
        reps_this_chunk = int(min(chunk_reps, remaining_reps))
        chunk_start_utc = datetime.now(UTC).isoformat(timespec="seconds")

        config, config_context = build_config_for_chunk(reps_this_chunk)
        program = program_class(config)
        data = program.acquire(progress=bool(acquire_progress))

        chunk_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        data_arrays, cts_s_arrays, axis_name, axis_data, axis_variants = normalize_acquired_data(
            data,
            sweep_axis_key=sweep_axis_key,
        )

        if chunk_index == 0:
            data_manager.write_sweep_axis(axis_name, axis_data, axis_variants)
            data_manager.initialize_summary_from_chunk(
                data_arrays=data_arrays,
                cts_s_arrays=cts_s_arrays,
                chunk_reps=reps_this_chunk,
            )
        else:
            data_manager.update_weighted_summary(
                data_arrays=data_arrays,
                cts_s_arrays=cts_s_arrays,
                chunk_reps=reps_this_chunk,
            )

        data_manager.store_chunk_archive(
            chunk_index=chunk_index,
            chunk_timestamp=chunk_timestamp,
            chunk_reps=reps_this_chunk,
            data_arrays=data_arrays,
            cts_s_arrays=cts_s_arrays,
            axis_name=axis_name,
            axis_data=axis_data,
            axis_variants=axis_variants,
            config_context=config_context,
        )

        rows.append(
            {
                "chunk_index": chunk_index,
                "chunk_reps": reps_this_chunk,
                "chunk_start_utc": chunk_start_utc,
                "chunk_timestamp": chunk_timestamp,
                "remaining_after_chunk": remaining_reps - reps_this_chunk,
            }
        )
        remaining_reps -= reps_this_chunk

        should_reopt = (
            reopt_callback is not None
            and reopt_every_n_chunks > 0
            and (chunk_index + 1) % int(reopt_every_n_chunks) == 0
            and chunk_index + 1 < planned_chunks
        )
        if should_reopt:
            reopt_context = {
                "run_id": run_id,
                "chunk_index": chunk_index,
                "next_chunk_index": chunk_index + 1,
                "chunk_reps": reps_this_chunk,
                "remaining_reps": remaining_reps,
                "experiment_name": experiment_name,
                "output_h5_path": output_h5_path,
                "output_dir": output_dir,
                "config": config,
                "config_context": config_context,
                "summary_rows": list(rows),
                "sweep_axis_key": sweep_axis_key,
            }
            try:
                last_reopt_result = dict(reopt_callback(reopt_context) or {"status": "ok"})
                rows[-1]["reopt_status"] = last_reopt_result.get("status", "ok")
                
                # Update piezo position in config_reopt.yaml if reopt succeeded
                if "x_um" in last_reopt_result and "y_um" in last_reopt_result and "z_um" in last_reopt_result:
                    reopt_config_path = Path(__file__).parent.parent / "config" / "config_reopt.yaml"
                    if reopt_config_path.exists():
                        try:
                            with open(reopt_config_path, "r") as f:
                                reopt_config = yaml.safe_load(f) or {}
                            if "reoptimization" not in reopt_config:
                                reopt_config["reoptimization"] = {}
                            reopt_config["reoptimization"]["piezo_initial_position_um"] = [
                                float(last_reopt_result["x_um"]),
                                float(last_reopt_result["y_um"]),
                                float(last_reopt_result["z_um"]),
                            ]
                            reopt_config["reoptimization"]["last_reopt_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
                            reopt_config["reoptimization"]["last_status"] = last_reopt_result.get("status", "ok")
                            with open(reopt_config_path, "w") as f:
                                yaml.dump(reopt_config, f, default_flow_style=False)
                        except Exception as exc:
                            print(f"Warning: Failed to update config_reopt.yaml: {exc}")
                
                _append_manifest_line(
                    manifest_path,
                    {
                        "event": "reopt_complete",
                        "run_id": run_id,
                        "chunk_index": chunk_index,
                        "result": last_reopt_result,
                    },
                )
            except Exception as exc:
                _append_manifest_line(
                    manifest_path,
                    {
                        "event": "reopt_failed",
                        "run_id": run_id,
                        "chunk_index": chunk_index,
                        "error": str(exc),
                    },
                )
                raise

    _write_summary_csv(summary_path, rows)

    data_manager.finalize_run("complete", planned_chunks)

    if plot_callback is not None:
        try:
            plot_result = plot_callback(output_h5_path)
            if hasattr(plot_result, "savefig"):
                plot_path = run_dir / plot_filename
                plot_result.savefig(plot_path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            _append_manifest_line(
                manifest_path,
                {
                    "event": "final_plot_failed",
                    "run_id": run_id,
                    "error": str(exc),
                },
            )
            raise

    _append_manifest_line(
        manifest_path,
        {
            "event": "run_complete",
            "run_id": run_id,
            "output_h5": str(output_h5_path),
            "plot_path": str(plot_path) if plot_path is not None else None,
            "n_chunks": planned_chunks,
            "total_reps": target_total_reps,
        },
    )

    return {
        "run_id": run_id,
        "output_h5_path": str(output_h5_path),
        "output_dir": str(output_dir),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "n_chunks_completed": planned_chunks,
    }

