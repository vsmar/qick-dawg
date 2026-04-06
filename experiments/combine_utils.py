"""
Generic utilities for combining chunk-level experiment HDF5 outputs.

Combined output conventions:
- /data holds regular-run-compatible aggregated datasets.
- /chunks/chunk_XXXX/data stores each chunk raw data arrays.
- /chunks/chunk_XXXX/experiment stores chunk-level metadata attrs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


AXIS_CANDIDATES = [
    "tau_ftns",
    "tau_ftus",
    "tau_tus",
    "mw_duration_ftns",
    "mw_duration_ns",
    "mw_fMHz",
    "t1_delay_tns",
    "t1_delay_tus",
]

SIGNAL_CANDIDATES = ["signal1_cts_s", "signal1"]
REFERENCE_CANDIDATES = ["signal2_cts_s", "signal2"]
CONTRAST_CANDIDATES = ["contrast"]


def _read_chunk_experiment_attrs(chunk_file: Path) -> Dict[str, object]:
    attrs: Dict[str, object] = {}
    with h5py.File(chunk_file, "r") as hf:
        if "experiment" in hf:
            for key, value in hf["experiment"].attrs.items():
                attrs[str(key)] = value
    return attrs


def _first_dataset(dgrp: h5py.Group, candidates: List[str]) -> tuple[Optional[str], Optional[np.ndarray]]:
    for name in candidates:
        if name in dgrp:
            return name, np.asarray(dgrp[name], dtype=float)
    return None, None


def combine_chunk_hdf5_files(summary_csv_path: Path, out_path: Path) -> None:
    rows: List[Dict[str, str]] = []
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    valid_rows = [r for r in rows if r.get("chunk_file") and Path(r["chunk_file"]).exists()]
    if not valid_rows:
        print("[combine] no valid chunk files found; skipping combined output")
        return

    axis_name = None
    x_axis = None
    weighted_signal = None
    weighted_reference = None
    weighted_contrast = None
    total_weight = 0.0

    chunk_sources: List[Dict[str, object]] = []

    for row in valid_rows:
        chunk_file = Path(row["chunk_file"])
        weight = float(row.get("chunk_reps", 0) or 0)
        if weight <= 0:
            continue

        with h5py.File(chunk_file, "r") as hf:
            if "data" not in hf:
                continue
            dgrp = hf["data"]

            this_axis_name, this_x = _first_dataset(dgrp, AXIS_CANDIDATES)
            if this_axis_name is None or this_x is None:
                raise ValueError(f"Chunk file missing known sweep axis dataset: {chunk_file}")

            _, this_signal = _first_dataset(dgrp, SIGNAL_CANDIDATES)
            if this_signal is None:
                raise ValueError(f"Chunk file missing signal dataset: {chunk_file}")

            _, this_reference = _first_dataset(dgrp, REFERENCE_CANDIDATES)
            _, this_contrast = _first_dataset(dgrp, CONTRAST_CANDIDATES)

        if axis_name is None:
            axis_name = this_axis_name
            x_axis = this_x
            weighted_signal = np.zeros_like(this_signal, dtype=float)
            weighted_reference = np.zeros_like(this_reference, dtype=float) if this_reference is not None else None
            weighted_contrast = np.zeros_like(this_contrast, dtype=float) if this_contrast is not None else None
        else:
            if this_axis_name != axis_name:
                raise ValueError("Cannot combine chunks with different sweep axis names.")
            if this_x.shape != x_axis.shape or not np.allclose(this_x, x_axis):
                raise ValueError("Cannot combine chunks with different sweep axis values.")
            if this_signal.shape != weighted_signal.shape:
                raise ValueError("Cannot combine chunks with different signal shapes.")
            if (this_reference is None) != (weighted_reference is None):
                weighted_reference = None
            if weighted_reference is not None and this_reference is not None and this_reference.shape != weighted_reference.shape:
                raise ValueError("Cannot combine chunks with different reference shapes.")
            if (this_contrast is None) != (weighted_contrast is None):
                weighted_contrast = None
            if weighted_contrast is not None and this_contrast is not None and this_contrast.shape != weighted_contrast.shape:
                raise ValueError("Cannot combine chunks with different contrast shapes.")

        weighted_signal += weight * this_signal
        if weighted_reference is not None and this_reference is not None:
            weighted_reference += weight * this_reference
        if weighted_contrast is not None and this_contrast is not None:
            weighted_contrast += weight * this_contrast
        total_weight += weight

        chunk_sources.append(
            {
                "chunk_file": str(chunk_file),
                "chunk_reps": weight,
                "chunk_index": int(float(row.get("cycle_index", row.get("chunk_index", len(chunk_sources))) or 0)),
                "chunk_start_utc": row.get("chunk_start_utc"),
                "chunk_end_utc": row.get("chunk_end_utc"),
            }
        )

    if total_weight <= 0 or x_axis is None or weighted_signal is None or axis_name is None:
        print("[combine] no weighted chunk data available; skipping combined output")
        return

    signal_avg = weighted_signal / total_weight
    reference_avg = (weighted_reference / total_weight) if weighted_reference is not None else None

    if weighted_contrast is not None:
        contrast_avg = weighted_contrast / total_weight
    elif reference_avg is not None:
        contrast_avg = signal_avg / np.clip(reference_avg, 1e-12, None)
    else:
        contrast_avg = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        dgrp = hf.create_group("data")
        dgrp.create_dataset(axis_name, data=x_axis)
        dgrp.create_dataset("signal1_cts_s", data=signal_avg)
        if reference_avg is not None:
            dgrp.create_dataset("signal2_cts_s", data=reference_avg)
        if contrast_avg is not None:
            dgrp.create_dataset("contrast", data=contrast_avg)

        exp = hf.create_group("experiment")
        exp.attrs["combined_from_summary_csv"] = str(summary_csv_path)
        exp.attrs["n_chunks_combined"] = len(chunk_sources)
        exp.attrs["weighted_by"] = "chunk_reps"

        chunks_grp = hf.create_group("chunks")
        for idx, src in enumerate(chunk_sources):
            chunk_file = Path(str(src["chunk_file"]))
            chunk_id = int(src.get("chunk_index", idx))
            chunk_name = f"chunk_{chunk_id:04d}"

            cg = chunks_grp.create_group(chunk_name)
            cg.attrs["source_file"] = str(chunk_file)
            cg.attrs["chunk_index"] = chunk_id
            cg.attrs["chunk_reps"] = float(src.get("chunk_reps", 0.0))
            if src.get("chunk_start_utc"):
                cg.attrs["chunk_start_utc"] = str(src["chunk_start_utc"])
            if src.get("chunk_end_utc"):
                cg.attrs["chunk_end_utc"] = str(src["chunk_end_utc"])

            exp_attrs = _read_chunk_experiment_attrs(chunk_file)
            if exp_attrs:
                chunk_exp = cg.create_group("experiment")
                for key, value in exp_attrs.items():
                    chunk_exp.attrs[key] = value

            with h5py.File(chunk_file, "r") as chunk_h5:
                if "data" in chunk_h5:
                    chunk_data = cg.create_group("data")
                    for key, ds in chunk_h5["data"].items():
                        chunk_data.create_dataset(key, data=np.asarray(ds))
                    for attr_key, attr_val in chunk_h5["data"].attrs.items():
                        chunk_data.attrs[attr_key] = attr_val

    print(f"[combine] wrote {out_path}")
