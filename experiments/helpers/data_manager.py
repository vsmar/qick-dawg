"""
data_manager.py — HDF5 Data Management for Experiments
========================================================
Handles the creation, layout, and writing of HDF5 files for
both single-run and chunked experiments.

Core Responsibilities:
- Create a standardized HDF5 file layout.
- Write and manage metadata, including experiment configuration.
- Store sweep axis data.
- Manage data for individual chunks in chunked runs.
- Aggregate and store summary data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import yaml

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


def _flatten_metadata(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        key_str = str(key)
        dotted = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, Mapping):
            flat.update(_flatten_metadata(value, dotted))
        else:
            flat[dotted] = value
    return flat


def _coerce_attr_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        if np.isnan(as_float) or np.isinf(as_float):
            return None
        return as_float
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.dtype.kind in ("i", "u", "f", "b"):
            return arr
        if arr.dtype.kind in ("U", "S"):
            return arr.astype("U")
        return json.dumps(_to_json_safe(value), sort_keys=True)
    if isinstance(value, str):
        return value
    return json.dumps(_to_json_safe(value), sort_keys=True)


def _axis_needs_store(root_axis_group: h5py.Group, name: str, data: np.ndarray) -> bool:
    if name not in root_axis_group:
        return True
    existing = np.asarray(root_axis_group[name][()])
    incoming = np.asarray(data)
    if existing.shape != incoming.shape:
        return True
    if np.issubdtype(existing.dtype, np.floating) or np.issubdtype(incoming.dtype, np.floating):
        return not np.allclose(existing, incoming, equal_nan=True)
    return not np.array_equal(existing, incoming)

def _write_arrays_to_group(
    destination: h5py.Group,
    arrays: Dict[str, np.ndarray],
    *,
    as_float: bool,
) -> None:
    """Write arrays into a group, replacing existing datasets with same names."""
    for name, arr in arrays.items():
        if name in destination:
            del destination[name]
        payload = np.asarray(arr, dtype=float) if as_float else np.asarray(arr)
        destination.create_dataset(name, data=payload)

class DataManager:
    """
    Manages the HDF5 file for an experiment run.
    """
    def __init__(self, output_h5_path: Path, experiment_name: str, run_id: str):
        self.output_h5_path = output_h5_path
        self.experiment_name = experiment_name
        self.run_id = run_id
        self._ensure_unified_structure()

    def _ensure_unified_structure(self):
        """Create the root groups used by the unified run file."""
        with h5py.File(self.output_h5_path, 'a') as hf:
            hf.require_group("metadata")
            hf.require_group("axis")
            summary_grp = hf.require_group("summary_data")
            summary_grp.require_group("data")
            summary_grp.require_group("cts_s")
            hf.require_group("chunks")

    def write_initial_metadata(self, cfg_dict: dict, program_class: Any, config_obj: Any, sweep_axis_key: str | None, custom_attrs: dict | None):
        """Writes the initial metadata to the HDF5 file."""
        from experiments.helpers.config import collect_required_cfg_attrs, add_unit_pair_expansions
        
        with h5py.File(self.output_h5_path, 'a') as hf:
            meta = hf["metadata"]
            
            local_dt = datetime.now().astimezone()
            utc_dt = datetime.now(UTC)
            timestamp_local = local_dt.strftime("%Y%m%d_%H%M%S")
            timestamp_utc = utc_dt.strftime("%Y%m%d_%H%M%S")

            config_yaml = yaml.dump(cfg_dict, sort_keys=False)
            required_attrs = collect_required_cfg_attrs(config_obj, getattr(program_class, "required_cfg", []))
            required_attrs = add_unit_pair_expansions(required_attrs, config_obj)

            experiment_attrs: Dict[str, Any] = {}
            experiment_attrs.update(_flatten_metadata(required_attrs))
            if custom_attrs:
                experiment_attrs.update(_flatten_metadata(custom_attrs))
            if sweep_axis_key:
                experiment_attrs["sweep_axis_key"] = sweep_axis_key

            experiment_attrs["run_id"] = self.run_id
            experiment_attrs["timestamp"] = timestamp_local
            experiment_attrs["timestamp_local"] = timestamp_local
            experiment_attrs["timestamp_utc"] = timestamp_utc
            experiment_attrs["timestamp_local_iso"] = local_dt.isoformat(timespec="seconds")
            experiment_attrs["timestamp_utc_iso"] = utc_dt.isoformat(timespec="seconds")
            experiment_attrs["experiment_name"] = self.experiment_name
            experiment_attrs["excitation_laser_power_mW"] = cfg_dict.get("optics", {}).get("excitation_laser_power_mW")
            sample_cfg = cfg_dict.get("sample", {})
            experiment_attrs["sample_id"] = sample_cfg.get("sample_id")
            experiment_attrs["sil_id"] = sample_cfg.get("sil_id")
            experiment_attrs["sample_notes"] = sample_cfg.get("notes")

            for key, value in experiment_attrs.items():
                coerced = _coerce_attr_value(value)
                if coerced is not None:
                    meta.attrs[key] = coerced
            meta.attrs["config_yaml"] = config_yaml

    def write_sweep_axis(
        self,
        axis_name: str | None,
        axis_data: np.ndarray | None,
        axis_variants: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Writes the sweep axis data to the HDF5 file."""
        payload: Dict[str, np.ndarray] = {}
        if axis_variants:
            payload.update(axis_variants)
        if axis_name and axis_data is not None and axis_name not in payload:
            payload[axis_name] = axis_data
        if not payload:
            return

        with h5py.File(self.output_h5_path, 'a') as hf:
            axis_grp = hf.require_group("axis")
            for name, data in payload.items():
                if name in axis_grp:
                    del axis_grp[name]
                axis_grp.create_dataset(name, data=np.asarray(data))

    def initialize_summary_from_chunk(self, data_arrays: Dict[str, np.ndarray], cts_s_arrays: Dict[str, np.ndarray], chunk_reps: int):
        """Populate the summary groups from the first chunk data."""
        with h5py.File(self.output_h5_path, 'a') as hf:
            data_grp = hf["summary_data/data"]
            cts_grp = hf["summary_data/cts_s"]

            _write_arrays_to_group(data_grp, data_arrays, as_float=True)
            _write_arrays_to_group(cts_grp, cts_s_arrays, as_float=True)

            hf["metadata"].attrs["total_weight"] = float(chunk_reps)

    def update_weighted_summary(self, data_arrays: Dict[str, np.ndarray], cts_s_arrays: Dict[str, np.ndarray], chunk_reps: int):
        """Update the running aggregation stored in /summary_data."""
        with h5py.File(self.output_h5_path, 'a') as hf:
            metadata = hf["metadata"].attrs
            previous_weight = float(metadata.get("total_weight", 0.0))
            new_weight = previous_weight + float(chunk_reps)

            data_grp = hf["summary_data/data"]
            cts_grp = hf["summary_data/cts_s"]

            # Total counts: sum across chunks
            for key, arr in data_arrays.items():
                if key not in data_grp:
                    data_grp.create_dataset(key, data=np.asarray(arr, dtype=float))
                    continue
                old_total = np.asarray(data_grp[key][()], dtype=float)
                new_total = old_total + np.asarray(arr, dtype=float)
                data_grp[key][()] = new_total

            # Count-rate traces: weighted average across chunks
            for key, arr in cts_s_arrays.items():
                if key not in cts_grp:
                    cts_grp.create_dataset(key, data=np.asarray(arr, dtype=float))
                    continue
                old_mean = np.asarray(cts_grp[key][()], dtype=float)
                new_mean = (old_mean * previous_weight + np.asarray(arr, dtype=float) * float(chunk_reps)) / new_weight
                cts_grp[key][()] = new_mean

            metadata["total_weight"] = new_weight

    def store_chunk_archive(
        self,
        chunk_index: int,
        chunk_timestamp: str,
        chunk_reps: int,
        data_arrays: Dict[str, np.ndarray],
        cts_s_arrays: Dict[str, np.ndarray],
        axis_name: str | None,
        axis_data: np.ndarray | None,
        axis_variants: Optional[Dict[str, np.ndarray]],
        config_context: Dict[str, Any] | None,
    ):
        """Store full per-chunk payload under /chunks/chunk_NNNN in the main file."""
        with h5py.File(self.output_h5_path, 'a') as hf:
            chunk_grp = hf.require_group(f"chunks/chunk_{chunk_index:04d}")
            chunk_grp.attrs["index"] = int(chunk_index)
            chunk_grp.attrs["timestamp"] = chunk_timestamp
            chunk_grp.attrs["reps"] = int(chunk_reps)

            data_grp = chunk_grp.require_group("data")
            cts_grp = chunk_grp.require_group("cts_s")
            axis_grp = chunk_grp.require_group("axis")

            _write_arrays_to_group(data_grp, data_arrays, as_float=False)
            _write_arrays_to_group(cts_grp, cts_s_arrays, as_float=False)

            axis_payload: Dict[str, np.ndarray] = {}
            if axis_variants:
                axis_payload.update(axis_variants)
            if axis_name and axis_data is not None and axis_name not in axis_payload:
                axis_payload[axis_name] = axis_data

            if axis_payload:
                root_axis_grp = hf.require_group("axis")
                for name, data in axis_payload.items():
                    if _axis_needs_store(root_axis_grp, name, data):
                        if name in axis_grp:
                            del axis_grp[name]
                        axis_grp.create_dataset(name, data=np.asarray(data))

            if config_context:
                flat_context = _flatten_metadata(config_context)
                for key, value in flat_context.items():
                    coerced = _coerce_attr_value(value)
                    if coerced is not None:
                        chunk_grp.attrs[f"ctx_{key}"] = coerced
            
            hf["metadata"].attrs["timestamp_last_chunk"] = chunk_timestamp
            hf["metadata"].attrs["last_chunk_index"] = int(chunk_index)

    def finalize_run(self, status: str, completed_chunks: int):
        """Writes final metadata to the HDF5 file."""
        with h5py.File(self.output_h5_path, 'a') as hf:
            hf["metadata"].attrs["timestamp_end"] = datetime.now(UTC).isoformat(timespec="seconds")
            hf["metadata"].attrs["status"] = status
            hf["metadata"].attrs["completed_chunks"] = int(completed_chunks)
