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
    Stored in config.yaml as nanoseconds (human-readable).
    convert_fine_timing() converts to samples at runtime using:

        duration_samples = duration_ns / (cycles2ns(1) / samps_per_clk)

    where samps_per_clk comes from the live soccfg after board connection.
    Call this after connect().

Typical usage
-------------
    from config import load_config, build_nv_config, connect, convert_fine_timing

    cfg    = load_config()
    soc, soccfg = connect(cfg)          # returns soc objects if qickdawg exposes them
    nv_cfg = build_nv_config(cfg)
    fine   = convert_fine_timing(cfg, soccfg, nv_cfg.mw_channel)

    # fine["pi_pulse"]  → duration in samples (tdds)
    # fine["pi2_pulse"] → duration in samples (tdds)
"""

import yaml
from pathlib import Path
from copy import copy
import qickdawg as qd

CONFIG_PATH = Path(__file__).parent / "config.yaml"


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

    if cfg["optics"]["excitation_laser_power_mW"] is None:
        errors.append(
            "optics.excitation_laser_power_mW is null — "
            "measure laser power and update config.yaml before running."
        )

    # Calibration values warn but don't block — Rabi establishes them.
    cal = cfg.get("calibration", {})
    for key in ("pi_pulse_tns", "pi2_pulse_tns"):
        if cal.get(key) is None:
            print(f"[config] Warning: calibration.{key} is not set.")

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
    Fine-resolution timing (_tdds) is NOT set here — call convert_fine_timing()
    separately after connect(), as it requires the live soccfg.
    """
    config = qd.NVConfiguration()

    hw = cfg["hardware"]
    t  = cfg["timing"]

    # Hardware
    config.adc_channel      = hw["adc_channel"]
    config.mw_channel       = hw["mw_channel"]
    config.mw_nqz           = hw["mw_nqz"]
    config.laser_gate_pmod  = hw["laser_gate_pmod"]

    # Microwave — defaults to calibration.default_transition (lower_dip or upper_dip)
    cal = cfg["calibration"]
    transition = cal[cal["default_transition"]]
    config.freq_fMHz = transition["freq_fMHz"]
    config.mw_gain   = transition["mw_gain"]

    # Standard timing (qickdawg converts to _treg internally)
    config.laser_init_tus           = t["laser_init_tus"]
    config.readout_integration_tns  = t["readout_integration_tns"]
    config.mw_to_laser_delay_tns    = t["mw_to_laser_delay_tns"]
    config.laser_readout_offset_tus = t["laser_readout_offset_tus"]
    config.pulse_seq_delay_tus      = t["pulse_seq_delay_tus"]
    config.pre_init                 = t["pre_init"]

    return config


# =============================================================================
# Fine-Resolution Timing Conversion  (ns → _tdds samples)
# =============================================================================

def convert_fine_timing(
    cfg: dict,
    soccfg,
    mw_channel: int,
) -> dict:
    """
    Convert pi_pulse_tns / pi2_pulse_tns from the active calibration transition
    to waveform samples (_tdds).

    Must be called after connect(), because samps_per_clk is read from the
    live soccfg returned by the board.

    Parameters
    ----------
    cfg        : dict loaded by load_config()
    soccfg     : live soccfg object from qickdawg / QICK after connection
    mw_channel : MW generator channel index (used to look up samps_per_clk)

    Returns
    -------
    dict mapping parameter name → duration in samples (int)
    e.g. {"pi_pulse": 400, "pi2_pulse": 200}

    Reads from calibration[default_transition] by default. Pass transition="upper_dip"
    to override.

    Conversion
    ----------
    One clock cycle = cycles2ns(1) nanoseconds.
    One clock cycle contains samps_per_clk waveform samples.
    Therefore one sample = cycles2ns(1) / samps_per_clk  nanoseconds.

        samples = duration_ns / (cycles2ns(1) / samps_per_clk)
    """
    from qick.helpers import cycles2ns  # import here to avoid hard dep at module level

    # Read pi/pi2 pulse times from the active transition in calibration
    cal = cfg["calibration"]
    transition = cal[cal["default_transition"]]
    fine_ns = {
        "pi_pulse":  transition["pi_pulse_tns"],
        "pi2_pulse": transition["pi2_pulse_tns"],
    }

    samps_per_clk = soccfg["gens"][mw_channel]["samps_per_clk"]
    ns_per_sample = cycles2ns(1) / samps_per_clk

    result = {}
    for name, duration_ns in fine_ns.items():
        if duration_ns is None:
            print(f"[config] Warning: fine_timing_ns.{name} is not set (null).")
            result[name] = None
            continue

        samples = duration_ns / ns_per_sample
        samples_int = int(round(samples))

        if abs(samples - samples_int) > 0.01:
            print(
                f"[config] Warning: fine_timing_ns.{name} = {duration_ns} ns "
                f"does not land on an exact sample boundary "
                f"({samples:.3f} samples). Rounded to {samples_int}."
            )

        result[name] = samples_int
        print(f"[config] fine_timing: {name} = {duration_ns} ns → {samples_int} samples")

    return result


# =============================================================================
# Convenience: ns_to_samples  (for one-off conversions in experiment scripts)
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
    from qick.helpers import cycles2ns
    samps_per_clk = soccfg["gens"][mw_channel]["samps_per_clk"]
    ns_per_sample = cycles2ns(1) / samps_per_clk
    return int(round(duration_ns / ns_per_sample))