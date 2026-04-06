"""
Generic chunked experiment runner with optional between-chunk reoptimization hook.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import save_experiment_hdf5


def _to_json_safe(value: Any) -> Any:
    """Convert values into JSON-serializable primitives."""
    try:
        import numpy as np

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


def _append_manifest_line(manifest_path: Path, payload: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_json_safe(payload), sort_keys=True) + "\n")


def _write_summary_csv(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_json_safe(row))


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
    enable_combine_at_end: bool = False,
    combine_filename: str = "combined.h5",
    combine_fn: Optional[Callable[[Path, Path], None]] = None,
    manifest_filename: str = "run_manifest.jsonl",
    summary_filename: str = "run_summary.csv",
) -> Dict[str, Any]:
    """
    Run an experiment in rep chunks and optionally invoke reoptimization between chunks.

    Notes
    -----
    - build_config_for_chunk(reps) must return (config, context_dict)
    - reopt_callback(context) can return any key/values; they will be persisted in summary
    """
    if chunk_reps <= 0:
        raise ValueError("chunk_reps must be > 0")
    if target_total_reps <= 0:
        raise ValueError("target_total_reps must be > 0")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{experiment_name}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / manifest_filename
    summary_path = run_dir / summary_filename

    planned_chunks = int(math.ceil(target_total_reps / chunk_reps))
    remaining_reps = int(target_total_reps)
    rows: List[Dict[str, Any]] = []

    pre_chunk_reopt_status = "not_run"

    for chunk_index in range(planned_chunks):
        reps_this_chunk = int(min(chunk_reps, remaining_reps))
        chunk_start_utc = datetime.now(UTC).isoformat(timespec="seconds")

        config, config_context = build_config_for_chunk(reps_this_chunk)
        prog = program_class(config)
        data = prog.acquire(progress=acquire_progress)

        out_path, timestamp = save_experiment_hdf5(
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

        row: Dict[str, Any] = {
            "run_id": run_id,
            "chunk_index": chunk_index,
            "timestamp": timestamp,
            "chunk_start_utc": chunk_start_utc,
            "chunk_end_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunk_reps": reps_this_chunk,
            "remaining_reps_after_chunk": int(remaining_reps - reps_this_chunk),
            "chunk_file": str(out_path),
            "pre_chunk_reopt_status": pre_chunk_reopt_status,
        }
        row.update(config_context)

        _append_manifest_line(manifest_path, {"event": "chunk_saved", **row})

        remaining_reps -= reps_this_chunk

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
                "chunk_file": str(out_path),
                "chunk_reps": reps_this_chunk,
                "remaining_reps": remaining_reps,
                "config_context": config_context,
                "config": config,
            }
            try:
                reopt_result = reopt_callback(context) or {}
                row.update({f"reopt_{k}": v for k, v in reopt_result.items()})
                pre_chunk_reopt_status = str(reopt_result.get("status", "ok"))
            except Exception as exc:
                row["reopt_status"] = "failed"
                row["reopt_error"] = str(exc)
                pre_chunk_reopt_status = "failed"
        else:
            row["reopt_status"] = "skipped"

        rows.append(row)
        _write_summary_csv(summary_path, rows)

    combined_path: Optional[Path] = None
    if enable_combine_at_end and combine_fn is not None:
        combined_path = run_dir / combine_filename
        combine_fn(summary_path, combined_path)

    run_summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "combined_path": str(combined_path) if combined_path is not None else None,
    }
    _append_manifest_line(manifest_path, {"event": "run_complete", **run_summary})
    return run_summary
