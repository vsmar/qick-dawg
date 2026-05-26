"""Shared helpers for experiment scripts.

This module is the stable public facade used by the run scripts. New helpers
live in dedicated modules under experiments.helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from experiments.helpers.config_builders import build_common_config
from experiments.helpers.experiment_runner import run_experiment
from experiments.helpers.experiment_specs import ChunkSpec, ExperimentSpec, PlotSpec
from experiments.helpers.h5_namespace import load_aggregated_h5_namespace
from experiments.helpers.plotting import build_plot_metadata, build_standard_title, make_chunked_plot_callback

__all__ = [
    "ChunkSpec",
    "ExperimentSpec",
    "PlotSpec",
    "build_common_config",
    "build_plot_metadata",
    "build_standard_title",
    "load_aggregated_h5_namespace",
    "make_chunked_plot_callback",
    "run_experiment",
    "run_standard_experiment",
]


def run_standard_experiment(
    *,
    run_mode: str,
    program_class: Any,
    cfg_dict: Dict[str, Any],
    config: Any,
    build_config_for_chunk: Any,
    output_dir: Any,
    experiment_name: str,
    target_total_reps: int,
    chunk_reps: int,
    acquire_progress: bool,
    plot_callback: Optional[Any] = None,
    plot_filename: str = "aggregated_plot.png",
    sweep_axis_key: Optional[str] = None,
    reopt_config_path: Optional[Path] = None,
    reopt_every_n_chunks: Optional[int] = None,
    single_plotter: Optional[Any] = None,
    single_plot_kwargs: Optional[Dict[str, Any]] = None,
    custom_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper around the new experiment runner."""
    plot = PlotSpec(
        chunked_callback=plot_callback,
        chunked_filename=plot_filename,
        sweep_axis_key=sweep_axis_key,
        single_plotter=single_plotter,
        single_plot_kwargs=dict(single_plot_kwargs or {}),
    )
    chunk = ChunkSpec(
        target_total_reps=int(target_total_reps),
        chunk_reps=int(chunk_reps),
        acquire_progress=bool(acquire_progress),
        reopt_every_n_chunks=reopt_every_n_chunks,
        reopt_config_path=reopt_config_path,
    )
    spec = ExperimentSpec(
        name=experiment_name,
        program_class=program_class,
        cfg_dict=cfg_dict,
        output_dir=output_dir,
        run_mode=run_mode,
        single_reps=int(getattr(config, "reps", chunk_reps)),
        build_config_for_reps=build_config_for_chunk,
        chunk=chunk,
        plot=plot,
        custom_attrs=custom_attrs,
        initial_config=config,
    )
    return run_experiment(spec)
