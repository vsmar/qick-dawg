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

Fine-resolution timing (_tdds / tsample):
    Stored in config.yaml directly as waveform sample counts.

Typical usage
-------------
    from config import load_config, build_nv_config, connect

    cfg    = load_config()
    connect(cfg)
    nv_cfg = build_nv_config(cfg)
"""

import yaml
from pathlib import Path
from copy import copy
import numpy as np
import qickdawg as qd

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
        for key in ("mw_pi_tdds", "mw_pi2_tdds"):
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
    Fine-resolution timing (_tdds) is read directly from calibration and assigned.
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
    config.mw_pi_tdds = transition["mw_pi_tdds"]
    config.mw_pi2_tdds = transition["mw_pi2_tdds"]

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

    return config


# =============================================================================
# Helper functions
# =============================================================================

def ns_to_samples(duration_ns: float, soccfg, mw_channel: int) -> int:
    """Convert a single duration in ns to waveform samples (_tdds units).
    
    Used for pulse durations and fine-resolution timing that need to be in
    waveform sample units (not register units). NVConfiguration handles
    automatic conversion for human units like fMHz, fGHz, tus, tns.
    
    Parameters
    ----------
    duration_ns    : float, duration in nanoseconds
    soccfg         : live soccfg object from qickdawg / QICK after connection
    mw_channel     : MW generator channel index
    
    Returns
    -------
    int, duration in waveform samples (tdds)
    """
    samps_per_clk = soccfg["gens"][mw_channel]["samps_per_clk"]
    ns_per_sample = (soccfg.cycles2us(1) * 1000) / samps_per_clk
    return int(round(duration_ns / ns_per_sample))


def get_ns_per_sample(soccfg, mw_channel: int) -> float:
    """Return waveform sample duration (ns) for the selected generator channel."""
    samps_per_clk = soccfg["gens"][mw_channel]["samps_per_clk"]
    return (soccfg.cycles2us(1) * 1000) / samps_per_clk


def set_hdf5_attr(group, key: str, value):
    """Set HDF5 attr while skipping None values."""
    if value is not None:
        group.attrs[key] = value


def write_hdf5_attrs(group, attrs: dict):
    """Write dict entries as HDF5 attributes, skipping None values."""
    for key, value in attrs.items():
        set_hdf5_attr(group, key, value)


def collect_required_cfg_attrs(cfg_obj, required_keys) -> dict:
    """Collect required_cfg values from NVConfiguration when present."""
    attrs = {}
    for key in required_keys:
        if hasattr(cfg_obj, key):
            attrs[key] = getattr(cfg_obj, key)
    return attrs


def add_unit_pair_expansions(attrs: dict, cfg_obj, ns_per_sample: float) -> dict:
    """Add paired unit forms (treg<->tns/tus, freg<->fMHz, tdds->ns)."""
    expanded = dict(attrs)

    for key, value in list(attrs.items()):
        if key.endswith("_treg"):
            stem = key[:-5]
            for alt in (f"{stem}_tns", f"{stem}_tus"):
                if alt not in expanded and hasattr(cfg_obj, alt):
                    expanded[alt] = getattr(cfg_obj, alt)

        if key.endswith("_freg"):
            alt = f"{key[:-5]}_fMHz"
            if alt not in expanded and hasattr(cfg_obj, alt):
                expanded[alt] = getattr(cfg_obj, alt)

        if key.endswith("_fMHz"):
            alt = f"{key[:-5]}_freg"
            if alt not in expanded and hasattr(cfg_obj, alt):
                expanded[alt] = getattr(cfg_obj, alt)

        if "_tdds" in key and np.isscalar(value):
            ns_key = key.replace("_tdds", "_ns")
            if ns_key not in expanded:
                expanded[ns_key] = float(value) * ns_per_sample

    return expanded


def save_experiment_hdf5(
    program_class,
    config_obj,
    cfg_dict: dict,
    data,
    output_dir: Path,
    experiment_name: str,
    ns_per_sample: float,
    custom_attrs: dict = None,
) -> tuple[Path, str]:
    """
    Save experiment data and metadata using the standard project HDF5 layout.

    Returns
    -------
    (Path, str)
        Saved file path and timestamp string used in the filename/metadata.
    """
    from datetime import datetime
    import h5py

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{experiment_name}_{timestamp}.h5"

    with h5py.File(out_path, "w") as f:
        if hasattr(data, "keys"):
            data_group = f.create_group("data")
            for key in data.keys():
                data_group.create_dataset(key, data=np.asarray(data[key]))
        else:
            f.create_dataset("data", data=np.asarray(data))

        config_yaml = yaml.dump(cfg_dict, sort_keys=False)
        f.attrs["config_yaml_full"] = config_yaml
        f.attrs["config_yaml"] = config_yaml

        required_attrs = collect_required_cfg_attrs(config_obj, program_class.required_cfg)
        required_attrs = add_unit_pair_expansions(required_attrs, config_obj, ns_per_sample)

        experiment_attrs = {"timestamp": timestamp}
        experiment_attrs.update(required_attrs)
        if custom_attrs:
            experiment_attrs.update(custom_attrs)

        exp = f.create_group("experiment")
        write_hdf5_attrs(exp, experiment_attrs)

        resolved_attrs = dict(experiment_attrs)
        resolved_attrs["experiment_name"] = experiment_name
        resolved_attrs["excitation_laser_power_mW"] = cfg_dict.get("optics", {}).get("excitation_laser_power_mW")

        sample_cfg = cfg_dict.get("sample", {})
        resolved_attrs["sample_id"] = sample_cfg.get("sample_id")
        resolved_attrs["sil_id"] = sample_cfg.get("sil_id")
        resolved_attrs["sample_notes"] = sample_cfg.get("notes")

        resolved = f.create_group("resolved_config")
        write_hdf5_attrs(resolved, resolved_attrs)

    return out_path, timestamp