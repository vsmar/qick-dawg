"""
Generic chunked experiment runner with unified HDF5 output.

Output file: {experiment_name}_{run_id}.h5
Structure:
  /metadata/
    - run_id, experiment_name, target_total_reps, chunk_reps, planned_chunks attributes
  /aggregated/data/
    - shared sweep axis + weighted-mean signal/reference/contrast (updated real-time)
  /chunks/chunk_NNN/
    - metadata/ (chunk_index, chunk_reps, timestamps, reopt_status)
    - data/ (raw signal/reference/contrast for this chunk)
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np

from config import save_experiment_hdf5



def _to_json_safe(value: Any) -> Any:
    """Convert values into JSON-serializable primitives."""
    try:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(value, np.ndarray):
            return [_to_json_safe(v) for v in value.tolist()]
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _extract_data_arrays(hdf5_group: h5py.Group) -> Dict[str, np.ndarray]:
    """Extract all datasets from a group into a flat dict."""
    arrays = {}
    
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            leaf = Path(name).name
            arrays[leaf] = np.asarray(obj[()], dtype=float)
    
    hdf5_group.visititems(visitor)
    return arrays


def _update_aggregated_data(
    hf: h5py.File,
    chunk_data_arrays: Dict[str, np.ndarray],
    chunk_reps: int,
    axis_name: Optional[str] = None,
) -> None:
    """Update /aggregated/data with weighted-mean of new chunk.
    
    On first chunk, initialize aggregated datasets.
    On subsequent chunks, add weighted contribution and update total_weight.
    """
    if "aggregated" not in hf:
        hf.create_group("aggregated")
    
    agg_grp = hf["aggregated"]
    
    if "data" not in agg_grp:
        # First chunk: initialize aggregated data
        data_grp = agg_grp.create_group("data")
        
        # Find and store the axis
        if axis_name is None:
            # Try common names
            for candidate in ["mw_duration_ftns", "tau_ftns", "tau_ns", "sweep_pts"]:
                if candidate in chunk_data_arrays:
                    axis_name = candidate
                    break
        
        if axis_name and axis_name in chunk_data_arrays:
            data_grp.create_dataset(axis_name, data=chunk_data_arrays[axis_name])
        
        # Initialize signal/reference/contrast as zeros (will accumulate)
        for key in ("signal1_cts_s", "signal1", "reference1_cts_s", "reference1", "signal2_cts_s", "signal2", "contrast"):
            if key in chunk_data_arrays:
                data_grp.create_dataset(key, data=np.zeros_like(chunk_data_arrays[key]))
        
        agg_grp.attrs["total_weight"] = chunk_reps
    else:
        agg_grp.attrs["total_weight"] = float(agg_grp.attrs.get("total_weight", 0)) + chunk_reps
    
    # Accumulate weighted data
    data_grp = agg_grp["data"]
    for key, arr in chunk_data_arrays.items():
        # Skip axis datasets
        if key in ("mw_duration_ftns", "tau_ftns", "tau_ns", "sweep_pts", "mw_duration_ns", "tau_tus"):
            continue
        # Skip if not in aggregated (shouldn't happen, but be safe)
        if key not in data_grp:
            continue
        
        current = np.asarray(data_grp[key], dtype=float)
        data_grp[key][()] = current + chunk_reps * arr




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
    enable_combine_at_end: bool = False,
    combine_filename: str = "combined.h5",  # Deprecated, kept for API compatibility
    combine_fn: Optional[Callable[[Path, Path], None]] = None,  # Deprecated
    manifest_filename: str = "run_manifest.jsonl",
    summary_filename: str = "run_summary.csv",
) -> Dict[str, Any]:
    """
    Run an experiment in chunks with unified HDF5 output and real-time aggregation.

    Output file: {output_dir}/{experiment_name}_{run_id}.h5
    
    Parameters
    ----------
    build_config_for_chunk(reps) -> (config, context_dict)
        Returns configuration and metadata dict for this chunk.
    reopt_callback(context) -> dict or None
        Optional reoptimization hook called between chunks.
        Returns dict with reopt results (status, x_um, y_um, z_um, etc.).
    
    Returns
    -------
    dict with keys: run_id, output_h5_path, output_dir, n_chunks_completed
    """
    if chunk_reps <= 0:
        raise ValueError("chunk_reps must be > 0")
    if target_total_reps <= 0:
        raise ValueError("target_total_reps must be > 0")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f".{experiment_name}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Single unified output file stored with the run artifacts
    output_h5_path = run_dir / f"{experiment_name}_{run_id}.h5"

    manifest_path = run_dir / manifest_filename
    summary_path = run_dir / summary_filename

    planned_chunks = int(math.ceil(target_total_reps / chunk_reps))
    remaining_reps = int(target_total_reps)
    rows: List[Dict[str, Any]] = []
    pre_chunk_reopt_status = "not_run"

    # Create main HDF5 file with metadata group
    with h5py.File(output_h5_path, "w") as hf:
        meta_grp = hf.create_group("metadata")
        meta_grp.attrs["run_id"] = str(run_id)
        meta_grp.attrs["experiment_name"] = str(experiment_name)
        meta_grp.attrs["target_total_reps"] = int(target_total_reps)
        meta_grp.attrs["chunk_reps"] = int(chunk_reps)
        meta_grp.attrs["planned_chunks"] = int(planned_chunks)
        meta_grp.attrs["timestamp_start"] = datetime.now(UTC).isoformat(timespec="seconds")

    # Process each chunk
    for chunk_index in range(planned_chunks):
        reps_this_chunk = int(min(chunk_reps, remaining_reps))
        chunk_start_utc = datetime.now(UTC).isoformat(timespec="seconds")

        config, config_context = build_config_for_chunk(reps_this_chunk)
        prog = program_class(config)
        data = prog.acquire(progress=acquire_progress)

        # Save chunk to temporary location (will extract arrays for unified .h5)
        temp_chunk_path = run_dir / f"temp_chunk_{chunk_index}.h5"
        chunk_out_path, timestamp = save_experiment_hdf5(
            program_class,
            config,
            cfg_dict,
            data,
            run_dir,
            experiment_name=f"{experiment_name}_chunk",
            custom_attrs={
                "loop_run_id": run_id,
                "loop_chunk_index": chunk_index,
                "loop_chunk_reps": reps_this_chunk,
                "loop_target_total_reps": int(target_total_reps),
                "loop_pre_chunk_reopt_status": pre_chunk_reopt_status,
            },
        )

        # Extract data from chunk and write to unified HDF5
        chunk_data_arrays: Dict[str, np.ndarray] = {}
        with h5py.File(chunk_out_path, "r") as chunk_hf:
            if "data" in chunk_hf:
                chunk_data_arrays = _extract_data_arrays(chunk_hf["data"])
        
        chunk_end_utc = datetime.now(UTC).isoformat(timespec="seconds")

        # Write chunk to unified .h5 and update aggregated
        with h5py.File(output_h5_path, "a") as hf:
            # Create chunk group
            chunk_grp = hf.create_group(f"chunks/chunk_{chunk_index:04d}")
            meta_chunk_grp = chunk_grp.create_group("metadata")
            data_chunk_grp = chunk_grp.create_group("data")
            
            # Store chunk metadata as attributes
            meta_chunk_grp.attrs["chunk_index"] = int(chunk_index)
            meta_chunk_grp.attrs["chunk_reps"] = int(reps_this_chunk)
            meta_chunk_grp.attrs["chunk_start_utc"] = str(chunk_start_utc)
            meta_chunk_grp.attrs["chunk_end_utc"] = str(chunk_end_utc)
            meta_chunk_grp.attrs["pre_chunk_reopt_status"] = str(pre_chunk_reopt_status)
            
            # Store chunk config context as JSON
            if config_context:
                meta_chunk_grp.attrs["config_context"] = json.dumps(_to_json_safe(config_context))
            
            # Write chunk data
            for key, arr in chunk_data_arrays.items():
                data_chunk_grp.create_dataset(key, data=arr)
            
            # Update aggregated data in real-time
            _update_aggregated_data(hf, chunk_data_arrays, reps_this_chunk)

        # Build summary row for debug CSV
        row: Dict[str, Any] = {
            "chunk_index": chunk_index,
            "chunk_reps": reps_this_chunk,
            "chunk_start_utc": chunk_start_utc,
            "chunk_end_utc": chunk_end_utc,
            "pre_chunk_reopt_status": pre_chunk_reopt_status,
        }
        row.update(config_context or {})

        remaining_reps -= reps_this_chunk

        # Optional reoptimization between chunks
        should_reopt = (
            reopt_callback is not None
            and reopt_every_n_chunks > 0
            and remaining_reps > 0
            and ((chunk_index + 1) % reopt_every_n_chunks == 0)
        )

        if should_reopt:
            context = {
                "run_id": run_id,
                "chunk_index": chunk_index,
                "chunk_reps": reps_this_chunk,
                "remaining_reps": remaining_reps,
                "config_context": config_context,
                "config": config,
            }
            try:
                reopt_result = reopt_callback(context) or {}
                row.update({f"reopt_{k}": v for k, v in reopt_result.items()})
                pre_chunk_reopt_status = str(reopt_result.get("status", "ok"))
                
                # Store reopt result in chunk metadata
                with h5py.File(output_h5_path, "a") as hf:
                    chunk_meta = hf[f"chunks/chunk_{chunk_index:04d}/metadata"]
                    chunk_meta.attrs["reopt_status"] = str(reopt_result.get("status", "ok"))
                    if "x_um" in reopt_result:
                        chunk_meta.attrs["reopt_x_um"] = float(reopt_result["x_um"])
                    if "y_um" in reopt_result:
                        chunk_meta.attrs["reopt_y_um"] = float(reopt_result["y_um"])
                    if "z_um" in reopt_result:
                        chunk_meta.attrs["reopt_z_um"] = float(reopt_result["z_um"])
            except Exception as exc:
                row["reopt_error"] = str(exc)
                pre_chunk_reopt_status = "failed"
                with h5py.File(output_h5_path, "a") as hf:
                    chunk_meta = hf[f"chunks/chunk_{chunk_index:04d}/metadata"]
                    chunk_meta.attrs["reopt_status"] = "failed"
                    chunk_meta.attrs["reopt_error"] = str(exc)
        else:
            row["reopt_status"] = "skipped"

        rows.append(row)

        # Clean up temporary chunk file
        if chunk_out_path.exists():
            chunk_out_path.unlink()

    # Finalize aggregated data: divide by total_weight
    with h5py.File(output_h5_path, "a") as hf:
        if "aggregated" in hf:
            total_weight = float(hf["aggregated"].attrs.get("total_weight", target_total_reps))
            if total_weight > 0:
                agg_data_grp = hf["aggregated/data"]
                for key in agg_data_grp.keys():
                    # Skip axis datasets
                    if key in ("mw_duration_ftns", "tau_ftns", "tau_ns", "sweep_pts", "mw_duration_ns", "tau_tus"):
                        continue
                    agg_data_grp[key][()] = np.asarray(agg_data_grp[key]) / total_weight
        
        # Update finish timestamp
        hf["metadata"].attrs["timestamp_end"] = datetime.now(UTC).isoformat(timespec="seconds")
        hf["metadata"].attrs["n_chunks_completed"] = int(planned_chunks)

    plot_path = None
    if plot_callback is not None:
        try:
            plot_result = plot_callback(output_h5_path)
            if hasattr(plot_result, "savefig"):
                plot_path = run_dir / plot_filename
                plot_result.savefig(plot_path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            _append_manifest_line(manifest_path, {
                "event": "plot_failed",
                "run_id": run_id,
                "error": str(exc),
            })
            raise

    # Optional: Write debug CSV summary (for backward compatibility and debugging)
    if rows:
        _write_summary_csv(summary_path, rows)
    
    # Log manifest event
    _append_manifest_line(manifest_path, {
        "event": "run_complete",
        "run_id": run_id,
        "output_h5": str(output_h5_path),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "n_chunks": planned_chunks,
        "total_reps": target_total_reps,
    })

    run_summary = {
        "run_id": run_id,
        "output_h5_path": str(output_h5_path),
        "output_dir": str(output_dir),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "n_chunks_completed": planned_chunks,
    }
    return run_summary


def _write_summary_csv(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write debug summary CSV (for backward compatibility)."""
    if not rows:
        return

    import csv

    fieldnames = sorted({k for row in rows for k in row.keys()})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_json_safe(row))


def _append_manifest_line(manifest_path: Path, payload: Dict[str, Any]) -> None:
    """Append debug manifest JSONL line (for backward compatibility)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_json_safe(payload), sort_keys=True) + "\n")

