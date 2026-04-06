"""
Optional class-based scaffold for experiment scripts.

Current scripts can keep function-style structure while progressively migrating to
this base class for shared setup/acquire/save/chunked orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import connect, load_config, save_experiment_hdf5
from experiment_helpers import maybe_run_chunked_mode


class BaseExperimentScript(ABC):
    """Shared orchestration for single-run and chunked-run experiment scripts."""

    program_class: Any
    experiment_name: str
    output_dir: Path

    run_mode: str = "single"
    target_total_reps: int = 0
    chunk_reps: int = 0
    acquire_progress: bool = True

    enable_reopt: bool = False
    reopt_every_n_chunks: int = 1
    reopt_hook_module: Optional[str] = None
    reopt_hook_function: str = "run_reoptimization_between_chunks"

    enable_combine_at_end: bool = True
    combine_filename: str = "combined.h5"
    combine_fn: Any = None
    piezo_initial_position_um: Optional[Tuple[float, float, float]] = None

    def __init__(self) -> None:
        self.cfg = load_config()
        connect(self.cfg)

    @abstractmethod
    def build_config(self, reps: int) -> Tuple[Any, Dict[str, Any]]:
        """Return (config, context) for the given reps count."""

    @abstractmethod
    def on_single_run_complete(self, config: Any, data: Any, out_path: Path, timestamp: str) -> None:
        """Run plotting/fit logic for single-run mode."""

    def run(self) -> None:
        if maybe_run_chunked_mode(
            run_mode=self.run_mode,
            program_class=self.program_class,
            cfg_dict=self.cfg,
            build_config_for_chunk=self.build_config,
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
            target_total_reps=int(self.target_total_reps),
            chunk_reps=int(self.chunk_reps),
            acquire_progress=bool(self.acquire_progress),
            piezo_initial_position_um=self.piezo_initial_position_um,
            combine_filename=self.combine_filename,
            combine_fn=self.combine_fn,
        ):
            return

        if self.run_mode != "single":
            raise ValueError("run_mode must be 'single' or 'chunked'.")

        config, _ = self.build_config(reps=self.default_single_reps())
        prog = self.program_class(config)
        data = prog.acquire(progress=bool(self.acquire_progress))

        out_path, timestamp = save_experiment_hdf5(
            self.program_class,
            config,
            self.cfg,
            data,
            self.output_dir,
            experiment_name=self.experiment_name,
        )
        self.on_single_run_complete(config, data, out_path, timestamp)

    @abstractmethod
    def default_single_reps(self) -> int:
        """Return reps used for single run mode."""
