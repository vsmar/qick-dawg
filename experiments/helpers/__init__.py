"""Public exports for experiments.helpers package.

Provide a small set of convenience re-exports so callers can use:

	from experiments.helpers import load_config, build_common_config

"""
from .config import (
	load_config,
	connect,
	build_nv_config,
	save_experiment_hdf5,
)

from .experiment_helpers import (
	build_common_config,
	make_chunked_plot_callback,
	run_experiment,
	run_standard_experiment,
	ChunkSpec,
	ExperimentSpec,
	PlotSpec,
)

try:
	from .reopt_hook import (
		create_reopt_callback,
		load_reopt_config,
		save_reopt_config,
		DEFAULT_REOPT_CONFIG_PATH,
	)
except ModuleNotFoundError:
	# Optional dependency (qdlutils) missing — provide safe fallbacks so
	# importing `experiments.helpers` doesn't raise. Callers intending to use
	# reoptimization should import `experiments.helpers.reopt_hook` directly
	# after installing the optional dependency.
	create_reopt_callback = None
	load_reopt_config = None
	save_reopt_config = None
	DEFAULT_REOPT_CONFIG_PATH = None

__all__ = [
	"load_config",
	"connect",
	"build_nv_config",
	"save_experiment_hdf5",
	"build_common_config",
	"make_chunked_plot_callback",
	"run_experiment",
	"run_standard_experiment",
	"ChunkSpec",
	"ExperimentSpec",
	"PlotSpec",
	"create_reopt_callback",
	"load_reopt_config",
	"save_reopt_config",
	"DEFAULT_REOPT_CONFIG_PATH",
]

