"""Helpers for loading unified HDF5 run files."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _group_items_to_namespace(proxy: SimpleNamespace, group, *, suffix: str = "") -> None:
    for key, dataset in group.items():
        setattr(proxy, f"{key}{suffix}", np.asarray(dataset[()]))


def load_aggregated_h5_namespace(h5_path: Path) -> SimpleNamespace:
    """Load aggregated data from the unified HDF5 file into a namespace."""
    h5py = importlib.import_module("h5py")

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find HDF5 file: {h5_path.resolve()}")

    proxy = SimpleNamespace()
    with h5py.File(h5_path, "r") as hf:
        if "summary_data" in hf and "data" in hf["summary_data"]:
            data_grp = hf["summary_data/data"]
            _group_items_to_namespace(proxy, data_grp)

            if "cts_s" in hf["summary_data"]:
                cts_grp = hf["summary_data/cts_s"]
                _group_items_to_namespace(proxy, cts_grp, suffix="_cts_s")
        elif "aggregated" in hf and "data" in hf["aggregated"]:
            _group_items_to_namespace(proxy, hf["aggregated/data"])
        else:
            raise ValueError(
                f"{h5_path.name} does not contain /summary_data/data; no supported aggregation layout was found"
            )

        if "axis" in hf:
            _group_items_to_namespace(proxy, hf["axis"])

        if "metadata" in hf:
            for key, value in hf["metadata"].attrs.items():
                try:
                    setattr(proxy, key, value.item() if hasattr(value, "item") else value)
                except Exception:
                    setattr(proxy, key, value)

    return proxy
