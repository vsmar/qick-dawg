"""Reoptimization helpers for chunked runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiments.helpers.reopt_hook import DEFAULT_REOPT_CONFIG_PATH, create_reopt_callback, load_reopt_config


def resolve_reopt_settings(
    *,
    reopt_config_path: Optional[Path],
    reopt_every_n_chunks: Optional[int],
) -> Tuple[Optional[Any], int]:
    """Resolve the reoptimization callback and cadence for chunked runs."""
    if reopt_every_n_chunks is not None and int(reopt_every_n_chunks) <= 0:
        return None, 0

    config_path = Path(reopt_config_path) if reopt_config_path is not None else DEFAULT_REOPT_CONFIG_PATH
    reopt_config = load_reopt_config(config_path)
    section = reopt_config.get("reoptimization", {}) if isinstance(reopt_config.get("reoptimization", {}), dict) else {}

    file_piezo = section.get("piezo_initial_position_um")
    piezo_initial_position_um = None
    if isinstance(file_piezo, (list, tuple)) and len(file_piezo) == 3:
        piezo_initial_position_um = (float(file_piezo[0]), float(file_piezo[1]), float(file_piezo[2]))

    if piezo_initial_position_um is None:
        raise ValueError(
            f"Chunked mode requires PIEZO_INITIAL_POSITION_UM=(x, y, z) or {config_path.name}.reoptimization.piezo_initial_position_um."
        )

    cadence = reopt_every_n_chunks
    if cadence is None:
        cadence = int(section.get("every_n_chunks", 1))

    scan_ranges = section.get("scan_ranges_um") if isinstance(section.get("scan_ranges_um"), dict) else None
    scan_pixels = section.get("scan_pixels") if isinstance(section.get("scan_pixels"), dict) else None
    scan_times = section.get("scan_times_s") if isinstance(section.get("scan_times_s"), dict) else None

    reopt_callback = create_reopt_callback(
        initial_position_um=piezo_initial_position_um,
        config_path=config_path,
        scan_ranges_um=scan_ranges,
        scan_pixels=scan_pixels,
        scan_times_s=scan_times,
    )

    return reopt_callback, int(cadence)
