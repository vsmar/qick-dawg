"""
config.py — NV Experiment Config Loader
========================================
Loads config.yaml, validates required fields, builds a populated
NVConfiguration, and handles fine-resolution timing conversion.

Unit conversion summary
-----------------------
Standard timing  (_tus, _tns, _fMHz):
    Assigned directly via qickdawg human-unit setters → qickdawg converts
    to _treg / _freg internally.

Fine-resolution timing (_ftsamp):
    Stored in config.yaml directly as waveform sample counts.

Typical usage
-------------
    from config import load_config, build_nv_config, connect

    cfg    = load_config()
    connect(cfg)
    nv_cfg = build_nv_config(cfg)
"""

from copy import copy

import yaml
from pathlib import Path
import numpy as np
import qickdawg as qd
import h5py

# TODO: Once this is running well I'll just standardize the set-up and just have 1 path
_ROOT = Path(__file__).resolve().parents[1]
_PRIMARY_CONFIG_PATH = _ROOT / "config" / "config.yaml"
_LEGACY_CONFIG_PATH = Path(__file__).parent / "config.yaml"
CONFIG_PATH = _PRIMARY_CONFIG_PATH if _PRIMARY_CONFIG_PATH.exists() else _LEGACY_CONFIG_PATH


# =============================================================================
# Load & Validate
# =============================================================================

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load config.yaml and run basic validation. Returns the config dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict):
    """Raise early with a clear message if critical fields are missing or null."""
    errors = []

    laser_power = cfg["optics"]["excitation_laser_power_mW"]
    print(f"[config] Documented Excitation Power is {laser_power} mW, ensure this aligns with measured excitation power")

    cal = cfg.get("calibration", {})
    transition_name = cal.get("default_transition")
    if transition_name is None:
        errors.append("calibration.default_transition is missing.")
    elif transition_name not in cal:
        errors.append(
            f"calibration.default_transition='{transition_name}' does not exist under calibration."
        )
    else:
        transition = cal[transition_name]
        for key in ("mw_pi_ftsamp", "mw_pi2_ftsamp"):
            if transition.get(key) is None:
                print(f"[config] Warning: calibration.{transition_name}.{key} is not set.")

    if "timing" not in cfg:
        errors.append("timing section is missing.")

    if "other" not in cfg:
        print("[config] Warning: other section is missing; using defaults where possible.")

    if "photon_counting" not in cfg:
        print("[config] Warning: photon_counting section is missing; using defaults where possible.")

    if errors:
        raise ValueError("\n".join(errors))


# =============================================================================
# Connect to QICK board
# =============================================================================

def connect(cfg: dict = None):
    """Start the qickdawg client using the IP from config.yaml."""
    if cfg is None:
        cfg = load_config()
    ip = cfg["infrastructure"]["qick_ip"]
    print(f"[config] Connecting to QICK at {ip} ...")
    qd.start_client(ip)
    print("[config] Connected.")


# =============================================================================
# Build NVConfiguration
# =============================================================================

def build_nv_config(cfg: dict) -> qd.NVConfiguration:
    """
    Populate a qickdawg NVConfiguration from the loaded YAML dict.
    Standard timing params are assigned via human-unit setters;
    qickdawg handles _treg / _freg conversion internally.
    Fine-resolution timing (_ftsamp) is read directly from calibration and assigned.
    """
    config = qd.NVConfiguration()

    hw = cfg["hardware"]
    t = cfg["timing"]
    o = cfg.get("other", {})
    pc = cfg.get("photon_counting", {})

    # Hardware / channels
    config.adc_channel      = hw["adc_channel"]
    config.mw_channel       = hw["mw_channel"]
    config.laser_gate_pmod  = hw["laser_gate_pmod"]

    # Microwave — defaults to calibration.default_transition (lower_dip or upper_dip)
    cal = cfg["calibration"]
    transition = cal[cal["default_transition"]]
    config.mw_fMHz    = transition["mw_fMHz"]
    config.mw_nqz     = transition.get("mw_nqz", hw.get("mw_nqz", 1))
    config.mw_gain    = transition["mw_gain"]
    config.mw_pi_ftsamp = transition["mw_pi_ftsamp"]
    config.mw_pi2_ftsamp = transition["mw_pi2_ftsamp"]

    # Standard timing (qickdawg converts to _treg internally)
    config.laser_on_tus             = t.get("laser_on_tus", t.get("laser_init_tus"))
    config.readout_integration_tns  = t["readout_integration_tns"]
    config.mw_to_laser_delay_tns    = t["mw_to_laser_delay_tns"]
    config.laser_readout_offset_tus = t["laser_readout_offset_tus"]
    config.readout_reference_start_tus = t.get("readout_reference_start_tus", 3.5)
    config.relax_delay_tus          = t["relax_delay_tus"]

    # Shared run controls
    config.reps = o.get("reps", 100)
    config.get_reference = o.get("get_reference", True)
    config.pre_init = o.get("pre_init", t.get("pre_init", True))

    # Photon counting controls
    config.edge_counting = pc.get("edge_counting", True)
    config.high_threshold = pc.get("high_threshold", 8000)
    config.low_threshold = pc.get("low_threshold", 500)

    # Temporary laser off safegaurd
    qd.laser_off(copy(config))

    return config


def set_hdf5_attr(group, key: str, value):
    """Set HDF5 attr while skipping None values."""
    if value is not None:
        group.attrs[key] = value


def write_hdf5_attrs(group, attrs: dict):
    """Write dict entries as HDF5 attributes, skipping None values."""
    for key, value in attrs.items():
        set_hdf5_attr(group, key, value)


def _is_axis_dataset_name(name: str) -> bool:
    """Return True if dataset name looks like a sweep axis (using suffix patterns)."""
    if not name:
        return False
    s = str(name).lower()
    if s == "sweep_pts":
        return True
    suffixes = (
        "_ftns", "_ftsamp", "_ftus", "_tns", "_tsamp", "_tus",
        "_fMHz", "_fGHz", "_freg", "_pdeg", "_preg",
    )
    return any(s.endswith(suf) for suf in suffixes)


_AXIS_SUFFIX_GROUPS = {
    "frequency": ("_fMHz", "_fGHz", "_freg"),
    "fine_time": ("_ftsamp", "_ftns", "_ftus"),
    "time": ("_tsamp", "_tns", "_tus"),
    "phase": ("_pdeg", "_preg"),
}


def _axis_variant_names(axis_name: str) -> list[str]:
    if not axis_name:
        return []
    if axis_name == "sweep_pts":
        return [axis_name]
    for suffixes in _AXIS_SUFFIX_GROUPS.values():
        for suffix in suffixes:
            if axis_name.endswith(suffix):
                stem = axis_name[: -len(suffix)]
                return [f"{stem}{alt}" for alt in suffixes]
    return [axis_name]


def _is_cts_s_dataset(name: str) -> bool:
    """Return True if dataset name indicates count-rate (cts/s) data."""
    return "_cts_s" in str(name).lower()


def _normalize_data_for_hdf5(data_dict: dict) -> tuple:
    """
    Organize raw experiment data into structured groups.
    
    Returns
    -------
    (data_arrays, cts_s_arrays, axis_name, axis_data)
        - data_arrays: signal/reference counts (excluding cts_s and contrast)
        - cts_s_arrays: count-rate versions
        - axis_name: name of sweep axis or None
        - axis_data: the sweep axis array or None
    """
    data_arrays = {}
    cts_s_arrays = {}
    axis_name = None
    axis_data = None
    
    for key, arr in data_dict.items():
        if key == "contrast":
            continue
        if _is_axis_dataset_name(key):
            axis_name = key
            axis_data = np.asarray(arr)
            continue
        if _is_cts_s_dataset(key):
            # Normalize count-rate key by removing the suffix `_cts_s` (case-insensitive)
            if key.lower().endswith("_cts_s"):
                base = key[: len(key) - 6]
            else:
                base = key
            cts_s_arrays[base] = np.asarray(arr)
        else:
            data_arrays[key] = np.asarray(arr)
    
    return data_arrays, cts_s_arrays, axis_name, axis_data


def _coerce_acquired_data_to_dict(data: object) -> dict:
    """Convert acquired experiment output into a plain dict."""
    if hasattr(data, "keys"):
        return dict(data)
    if hasattr(data, "__dict__"):
        return dict(vars(data))
    return {"data": np.asarray(data)}


def _numeric_only_arrays(arrays: dict) -> dict:
    """Keep only numeric arrays; drop strings/objects that cannot be aggregated."""
    numeric = {}
    for key, arr in arrays.items():
        arr_np = np.asarray(arr)
        if np.issubdtype(arr_np.dtype, np.number):
            numeric[key] = arr_np
    return numeric


def normalize_acquired_data(data: object, sweep_axis_key: str = None) -> tuple:
    """Normalize acquired data into numeric summary arrays and optional sweep axis.

    Returns
    -------
    (data_arrays, cts_s_arrays, axis_name, axis_data, axis_variants)
        data_arrays and cts_s_arrays contain only numeric arrays.
    """
    raw_data = _coerce_acquired_data_to_dict(data)
    data_arrays, cts_s_arrays, detected_axis_name, detected_axis_data = _normalize_data_for_hdf5(raw_data)

    data_arrays = _numeric_only_arrays(data_arrays)
    cts_s_arrays = _numeric_only_arrays(cts_s_arrays)

    axis_name = sweep_axis_key or detected_axis_name
    axis_data = None
    axis_variants: dict[str, np.ndarray] = {}

    axis_candidates = []
    if axis_name:
        axis_candidates.extend(_axis_variant_names(axis_name))
    if detected_axis_name and detected_axis_name not in axis_candidates:
        axis_candidates.extend(_axis_variant_names(detected_axis_name))

    for name in axis_candidates:
        if name in raw_data:
            candidate = np.asarray(raw_data[name])
            if np.issubdtype(candidate.dtype, np.number):
                axis_variants[name] = candidate

    if axis_name and axis_name in axis_variants:
        axis_data = axis_variants[axis_name]
    elif detected_axis_name and detected_axis_name in axis_variants:
        axis_data = axis_variants[detected_axis_name]
    elif detected_axis_data is not None:
        candidate = np.asarray(detected_axis_data)
        if np.issubdtype(candidate.dtype, np.number):
            axis_data = candidate

    if axis_name and axis_data is not None and axis_name not in axis_variants:
        axis_variants[axis_name] = axis_data

    for axis_key in axis_variants:
        data_arrays.pop(axis_key, None)

    return data_arrays, cts_s_arrays, axis_name, axis_data, axis_variants



def collect_required_cfg_attrs(cfg_obj, required_keys) -> dict:
    """Collect required_cfg values from NVConfiguration when present."""
    attrs = {}
    for key in required_keys:
        if hasattr(cfg_obj, key):
            attrs[key] = getattr(cfg_obj, key)
    return attrs


def add_unit_pair_expansions(attrs: dict, cfg_obj) -> dict:
    """Add paired unit forms (treg<->tns/tus, freg<->fMHz, ftsamp<->ftns/ftus)."""
    expanded = dict(attrs)

    for key, value in list(attrs.items()):
        if key.endswith("_treg"):
            stem = key[:-5]
            for alt in (f"{stem}_tns", f"{stem}_tus"):
                if alt not in expanded and hasattr(cfg_obj, alt):
                    expanded[alt] = getattr(cfg_obj, alt)

        if key.endswith("_freg"):
            stem = key[:-5]
            for alt in (f"{stem}_fMHz"):
                if alt not in expanded and hasattr(cfg_obj, alt):
                    expanded[alt] = getattr(cfg_obj, alt)

        if key.endswith("_fMHz"):
            stem = key[:-5]
            for alt in (f"{stem}_freg", f"{stem}_fGHz"):
                if alt not in expanded and hasattr(cfg_obj, alt):
                    expanded[alt] = getattr(cfg_obj, alt)

        if key.endswith("_ftsamp"):
            stem = key[:-7]
            for alt in (f"{stem}_ftns", f"{stem}_ftus"):
                if alt not in expanded and hasattr(cfg_obj, alt):
                    expanded[alt] = getattr(cfg_obj, alt)

        if key.endswith("_ftns"):
            stem = key[:-5]
            for alt in (f"{stem}_ftsamp", f"{stem}_ftus"):
                if alt not in expanded and hasattr(cfg_obj, alt):
                    expanded[alt] = getattr(cfg_obj, alt)

        if key.endswith("_ftus"):
            stem = key[:-5]
            for alt in (f"{stem}_ftsamp", f"{stem}_ftns"):
                if alt not in expanded and hasattr(cfg_obj, alt):
                    expanded[alt] = getattr(cfg_obj, alt)

    return expanded


def save_experiment_hdf5(
    program_class,
    config_obj,
    cfg_dict: dict,
    data,
    output_dir: Path,
    experiment_name: str,
    sweep_axis_key: str = None,
    custom_attrs: dict = None,
) -> tuple[Path, str]:
    """
    Save experiment data and metadata using the standard project HDF5 layout.
    
    Layout:
      /metadata/          — attributes with experiment config
      /axis/{axis_name}   — sweep axis data (stored once, easy to extract)
      /summary_data/data/ — raw signal/reference traces
      /summary_data/cts_s/ — count-rate versions

    Parameters
    ----------
    sweep_axis_key : str, optional
        Name of the sweep axis (e.g., "mw_duration_ftns", "mw_fMHz", "tau_ftns").
        Stored as metadata; axis data will be saved under /axis/ if found.

    Returns
    -------
    (Path, str)
        Saved file path and timestamp string used in the filename/metadata.
    """
    from datetime import UTC, datetime
    from experiments.helpers.data_manager import DataManager

    output_dir.mkdir(parents=True, exist_ok=True)

    local_dt = datetime.now().astimezone()
    timestamp_local = local_dt.strftime("%Y%m%d_%H%M%S")
    run_id = timestamp_local
    out_path = output_dir / f"{experiment_name}_{timestamp_local}.h5"

    data_manager = DataManager(out_path, experiment_name, run_id)
    
    data_manager.write_initial_metadata(cfg_dict, program_class, config_obj, sweep_axis_key, custom_attrs)

    data_arrays, cts_s_arrays, detected_axis_name, axis_data, axis_variants = normalize_acquired_data(
        data,
        sweep_axis_key=sweep_axis_key,
    )

    if sweep_axis_key is None:
        sweep_axis_key = detected_axis_name
    
    if axis_variants or (sweep_axis_key and axis_data is not None):
        data_manager.write_sweep_axis(sweep_axis_key, axis_data, axis_variants)
    
    with h5py.File(out_path, "a") as f:
        summary_grp = f.require_group("summary_data")
        data_grp = summary_grp.require_group("data")
        cts_s_grp = summary_grp.require_group("cts_s")
        
        for key, arr in data_arrays.items():
            try:
                data_grp.create_dataset(key, data=arr)
            except TypeError:
                data_grp.attrs[key] = str(arr)
        
        for key, arr in cts_s_arrays.items():
            try:
                cts_s_grp.create_dataset(key, data=arr)
            except TypeError:
                cts_s_grp.attrs[key] = str(arr)

    return out_path, timestamp_local