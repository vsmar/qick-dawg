"""Shared configuration builders for experiment scripts."""

from __future__ import annotations

from typing import Dict, Optional

from experiments.helpers.config import build_nv_config


def build_common_config(
    cfg: Dict,
    reps: int,
    *,
    transition: Optional[str] = None,
    override_freq_fMHz: Optional[float] = None,
    override_mw_gain: Optional[float] = None,
    override_mw_pi_ftsamp: Optional[int] = None,
    override_mw_pi_ftns: Optional[float] = None,
    get_reference: bool = True,
) -> tuple:
    """Build a common NVConfiguration with shared override handling."""
    cfg = dict(cfg)
    config = build_nv_config(cfg)

    active_transition = transition or cfg["calibration"]["default_transition"]
    transition_cfg = cfg["calibration"][active_transition]

    config.mw_fMHz = override_freq_fMHz if override_freq_fMHz is not None else transition_cfg["mw_fMHz"]
    config.mw_gain = override_mw_gain if override_mw_gain is not None else transition_cfg["mw_gain"]

    if override_mw_pi_ftsamp is not None and override_mw_pi_ftns is not None:
        raise ValueError("Set only one of override_mw_pi_ftsamp or override_mw_pi_ftns.")

    if override_mw_pi_ftsamp is not None:
        config.mw_pi_ftsamp = int(override_mw_pi_ftsamp)
        pi_source = f"override_mw_pi_ftsamp={config.mw_pi_ftsamp}"
    elif override_mw_pi_ftns is not None:
        config.mw_pi_ftns = float(override_mw_pi_ftns)
        pi_source = f"override_mw_pi_ftns={config.mw_pi_ftns}"
    else:
        if transition_cfg.get("mw_pi_ftsamp") is None:
            pi_source = "calibration:missing_pi"
        else:
            config.mw_pi_ftsamp = int(transition_cfg["mw_pi_ftsamp"])
            pi_source = f"calibration.{active_transition}.mw_pi_ftsamp={config.mw_pi_ftsamp}"

    config.reps = int(reps)
    config.get_reference = bool(get_reference)

    return config, active_transition, pi_source
