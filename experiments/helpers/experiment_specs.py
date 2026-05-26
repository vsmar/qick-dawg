"""Dataclasses that describe standard experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


@dataclass(frozen=True)
class PlotSpec:
    """Plot configuration for single and chunked runs."""

    chunked_callback: Optional[Callable[[Path], Any]] = None
    chunked_filename: str = "aggregated_plot.png"
    sweep_axis_key: Optional[str] = None
    single_plotter: Optional[Callable[..., Any]] = None
    single_plot_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkSpec:
    """Chunking configuration for chunked runs."""

    target_total_reps: int
    chunk_reps: int
    acquire_progress: bool = True
    reopt_every_n_chunks: Optional[int] = None
    reopt_config_path: Optional[Path] = None


@dataclass(frozen=True)
class ExperimentSpec:
    """Aggregate specification for running an experiment."""

    name: str
    program_class: Any
    cfg_dict: Dict[str, Any]
    output_dir: Path
    run_mode: str
    single_reps: int
    build_config_for_reps: Callable[[int], tuple[Any, Dict[str, Any]]]
    chunk: ChunkSpec
    plot: PlotSpec = field(default_factory=PlotSpec)
    custom_attrs: Optional[Mapping[str, Any]] = None
    header_lines: Optional[list[str]] = None
    initial_config: Any = None
    initial_context: Optional[Dict[str, Any]] = None
