"""Standardized experiment runner."""

from __future__ import annotations

from typing import Any, Dict, Optional

from experiments.helpers.chunked_runner import run_chunked_experiment
from experiments.helpers.config import save_experiment_hdf5
from experiments.helpers.experiment_specs import ExperimentSpec
from experiments.helpers.reopt import resolve_reopt_settings


def _merge_metadata(primary: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    if primary:
        merged.update(primary)
    if extra:
        merged.update(extra)
    return merged or None


def _print_header(lines: Optional[list[str]]) -> None:
    if not lines:
        return
    for line in lines:
        print(line)


def run_experiment(spec: ExperimentSpec) -> Dict[str, Any]:
    """Run a standard experiment based on the provided specification."""
    _print_header(spec.header_lines)

    if spec.run_mode == "chunked":
        reopt_callback, reopt_every_n_chunks = resolve_reopt_settings(
            reopt_config_path=spec.chunk.reopt_config_path,
            reopt_every_n_chunks=spec.chunk.reopt_every_n_chunks,
        )

        run_summary = run_chunked_experiment(
            program_class=spec.program_class,
            cfg_dict=spec.cfg_dict,
            build_config_for_chunk=spec.build_config_for_reps,
            output_dir=spec.output_dir,
            experiment_name=spec.name,
            target_total_reps=int(spec.chunk.target_total_reps),
            chunk_reps=int(spec.chunk.chunk_reps),
            acquire_progress=bool(spec.chunk.acquire_progress),
            reopt_every_n_chunks=int(reopt_every_n_chunks),
            reopt_callback=reopt_callback,
            plot_callback=spec.plot.chunked_callback,
            plot_filename=spec.plot.chunked_filename,
            sweep_axis_key=spec.plot.sweep_axis_key,
            metadata_attrs=spec.custom_attrs,
        )
        print(f"[{spec.name}] Chunked run complete: {run_summary}")
        return {"mode": "chunked"}

    if spec.run_mode != "single":
        raise ValueError("RUN_MODE must be 'single' or 'chunked'.")

    config = spec.initial_config
    context = spec.initial_context
    if config is None:
        config, context = spec.build_config_for_reps(int(spec.single_reps))

    metadata_attrs = _merge_metadata(context, dict(spec.custom_attrs) if spec.custom_attrs else None)

    program = spec.program_class(config)
    data = program.acquire(progress=bool(spec.chunk.acquire_progress))

    out_path, timestamp = save_experiment_hdf5(
        spec.program_class,
        config,
        spec.cfg_dict,
        data,
        spec.output_dir,
        experiment_name=spec.name,
        sweep_axis_key=spec.plot.sweep_axis_key,
        custom_attrs=metadata_attrs,
    )

    if callable(spec.plot.single_plotter):
        plot_kwargs = dict(spec.plot.single_plot_kwargs or {})
        spec.plot.single_plotter(data, cfg=config, **plot_kwargs)

    return {
        "mode": "single",
        "out_path": out_path,
        "timestamp": timestamp,
        "data": data,
    }
