"""
Utilities for combining chunk-level CPMG HDF5 outputs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


def combine_chunk_hdf5_files(summary_csv_path: Path, out_path: Path) -> None:
    """
    Combine chunk-level HDF5 files into a single weighted-average dataset.

    Weights are chunk reps, so this approximates an equivalent single run with
    TARGET_TOTAL_REPS when all chunks share the same sweep axis.
    """
    rows: List[Dict[str, str]] = []
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    valid_rows = [r for r in rows if r.get("chunk_file") and Path(r["chunk_file"]).exists()]
    if not valid_rows:
        print("[combine] no valid chunk files found; skipping combined output")
        return

    x_axis = None
    weighted_signals = None
    weighted_refs = None
    total_weight = 0.0

    for row in valid_rows:
        chunk_file = Path(row["chunk_file"])
        weight = float(row.get("chunk_reps", 0) or 0)
        if weight <= 0:
            continue

        with h5py.File(chunk_file, "r") as hf:
            if "data" not in hf:
                continue

            dgrp = hf["data"]

            if "tau_ftns" not in dgrp:
                raise ValueError(f"Chunk file missing required tau_ftns axis dataset: {chunk_file}")
            this_x = np.asarray(dgrp["tau_ftns"], dtype=float)

            if "signal1_cts_s" in dgrp:
                this_signal = np.asarray(dgrp["signal1_cts_s"], dtype=float)
            elif "signal1" in dgrp:
                this_signal = np.asarray(dgrp["signal1"], dtype=float)
            else:
                raise ValueError(f"Chunk file missing signal dataset: {chunk_file}")

            if "signal2_cts_s" in dgrp:
                this_ref = np.asarray(dgrp["signal2_cts_s"], dtype=float)
            elif "signal2" in dgrp:
                this_ref = np.asarray(dgrp["signal2"], dtype=float)
            else:
                this_ref = None

        if x_axis is None:
            x_axis = this_x
            weighted_signals = np.zeros_like(this_signal, dtype=float)
            weighted_refs = np.zeros_like(this_ref, dtype=float) if this_ref is not None else None
        else:
            if this_x.shape != x_axis.shape or not np.allclose(this_x, x_axis):
                raise ValueError("Cannot combine chunks with different tau axis values.")
            if this_signal.shape != weighted_signals.shape:
                raise ValueError("Cannot combine chunks with different signal shape.")
            if (this_ref is None) != (weighted_refs is None):
                raise ValueError("Cannot combine mixed presence/absence of reference signal.")
            if this_ref is not None and this_ref.shape != weighted_refs.shape:
                raise ValueError("Cannot combine chunks with different reference shape.")

        weighted_signals += weight * this_signal
        if weighted_refs is not None and this_ref is not None:
            weighted_refs += weight * this_ref
        total_weight += weight

    if total_weight <= 0 or x_axis is None or weighted_signals is None:
        print("[combine] no weighted chunk data available; skipping combined output")
        return

    signal_avg = weighted_signals / total_weight
    ref_avg = (weighted_refs / total_weight) if weighted_refs is not None else None
    contrast_avg = signal_avg / np.clip(ref_avg, 1e-12, None) if ref_avg is not None else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        dgrp = hf.create_group("data")
        dgrp.create_dataset("tau_ftns", data=x_axis)
        dgrp.create_dataset("signal1_cts_s", data=signal_avg)
        if ref_avg is not None:
            dgrp.create_dataset("signal2_cts_s", data=ref_avg)
        if contrast_avg is not None:
            dgrp.create_dataset("contrast", data=contrast_avg)

        exp = hf.create_group("experiment")
        exp.attrs["combined_from_summary_csv"] = str(summary_csv_path)
        exp.attrs["n_chunks_combined"] = len(valid_rows)
        exp.attrs["weighted_by"] = "chunk_reps"

    print(f"[combine] wrote {out_path}")
