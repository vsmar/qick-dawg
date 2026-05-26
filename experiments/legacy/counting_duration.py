"""
counting_duration.py
====================
Run a counting-duration fine-resolution sweep, save data, and generate
SNR vs integration time plots with optimal window annotations.

Edit the EXPERIMENT PARAMETERS block before each run.
"""

from pathlib import Path
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from qickdawg.finetimingsuite import CountingDurationFineRes

from experiments.helpers.config import (
    load_config,
    connect,
    save_experiment_hdf5,
)
from experiments.helpers.config_builders import build_common_config

# =============================================================================
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

READOUT_OFFSET_START_TNS = 1000
READOUT_OFFSET_STOP_TNS = 2500
READOUT_OFFSET_STEP_TNS = 10

REPS = 300_000
READOUT_INTEGRATION_TNS = 10
GET_REFERENCE = True

T_MIN_TUS = 0.0
THRESHOLD_PCT = 10.0
ACQUIRE_PROGRESS = True

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "counting_duration"


# =============================================================================
# Acquisition helpers
# =============================================================================

def _as_scalar(value):
    """Extract scalar from array-like value."""
    arr = np.asarray(value).squeeze()
    if arr.size == 1:
        return float(arr)
    return float(np.mean(arr))


def _acquire_full_res(config, delays_tns, show_progress=True):
    """Acquire counting duration sweep over readout offsets."""
    results = None

    total_steps = len(delays_tns)
    sweep_iter = tqdm(
        delays_tns,
        total=total_steps,
        desc="Counting-duration sweep",
        unit="step",
        disable=not show_progress,
    )

    for idx, delay_tns in enumerate(sweep_iter, start=1):
        config.laser_readout_offset_tns = int(delay_tns)

        prog = CountingDurationFineRes(config)
        d = prog.acquire()

        if results is None:
            results = copy(d)
            for key in d.keys():
                results[key] = np.empty(0, dtype=float)
            results.delay_tus = np.empty(0, dtype=float)
            results.delay_tns = np.empty(0, dtype=float)

        for key, value in d.items():
            results[key] = np.append(results[key], _as_scalar(value))

        results.delay_tus = np.append(results.delay_tus, config.laser_readout_offset_tus)
        results.delay_tns = np.append(results.delay_tns, delay_tns)

        sweep_iter.set_postfix({
            "offset_ns": int(delay_tns),
            "done": f"{idx}/{total_steps}",
        }, refresh=False)

    return results


def _compute_optimal_window(data, *, t_min_tus=0.0, threshold_pct=10.0):
    """Compute optimal readout window based on cumulative SNR."""
    delay_tus = np.asarray(getattr(data, "delay_tus", []), dtype=float)
    signal_on = np.asarray(
        getattr(data, "signal1_cts_s", getattr(data, "signal1", [])),
        dtype=float,
    )
    signal_off = np.asarray(
        getattr(data, "signal2_cts_s", getattr(data, "signal2", [])),
        dtype=float,
    )

    n = min(len(delay_tus), len(signal_on), len(signal_off))
    if n == 0:
        raise ValueError("No data found for counting duration analysis.")

    delay_s = delay_tus[:n] * 1e-6
    signal_on = signal_on[:n]
    signal_off = signal_off[:n]
    subtracted = signal_off - signal_on

    t_min_s = float(t_min_tus) * 1e-6
    mask = delay_s >= t_min_s
    if not np.any(mask):
        raise ValueError("No points remain after applying t_min_tus.")

    delay_s = delay_s[mask] - t_min_s
    signal_on = signal_on[mask]
    signal_off = signal_off[mask]
    subtracted = subtracted[mask]

    snr_values = np.zeros_like(delay_s, dtype=float)
    for idx, t in enumerate(delay_s):
        mask_t = delay_s <= t
        signal = np.trapz(subtracted[mask_t], delay_s[mask_t])
        noise = np.trapz(signal_on[mask_t] + signal_off[mask_t], delay_s[mask_t])
        snr_values[idx] = signal / np.sqrt(noise) if noise > 0 else 0.0

    if snr_values.size > 1:
        max_idx = int(np.argmax(snr_values[1:]) + 1)
    else:
        max_idx = 0
    max_time = float(delay_s[max_idx])
    max_snr = float(snr_values[max_idx])

    max_signal = float(np.max(subtracted)) if subtracted.size else 0.0
    threshold = (float(threshold_pct) / 100.0) * max_signal
    above_threshold = subtracted >= threshold if subtracted.size else np.array([])

    if above_threshold.size and np.any(above_threshold):
        first_above_idx = int(np.where(above_threshold)[0][0])
        switch_idx = max(first_above_idx - 1, 0)
    else:
        switch_idx = 0

    switch_on_time = float(delay_s[switch_idx])

    readout_mask = (delay_s >= switch_on_time) & (delay_s <= max_time)
    if np.count_nonzero(readout_mask) < 2:
        lo = min(switch_idx, max_idx)
        hi = max(switch_idx, max_idx)
        readout_mask = np.zeros_like(delay_s, dtype=bool)
        readout_mask[lo:hi + 1] = True

    signal_window = np.trapz(subtracted[readout_mask], delay_s[readout_mask])
    noise_window = np.trapz(signal_on[readout_mask] + signal_off[readout_mask], delay_s[readout_mask])
    readout_window_snr = signal_window / np.sqrt(noise_window) if noise_window > 0 else 0.0

    return {
        "delay_s": delay_s,
        "signal_on": signal_on,
        "signal_off": signal_off,
        "subtracted": subtracted,
        "snr_values": snr_values,
        "switch_on_time_s": switch_on_time,
        "max_time_s": max_time,
        "max_snr": max_snr,
        "readout_window_snr": float(readout_window_snr),
        "threshold": float(threshold),
        "threshold_pct": float(threshold_pct),
        "t_min_s": float(t_min_s),
    }


def _plot_counting_duration(summary, *, show=True):
    """Plot signal traces and SNR with optimal window annotations."""
    delay_us = summary["delay_s"] * 1e6
    signal_on = summary["signal_on"]
    signal_off = summary["signal_off"]
    subtracted = summary["subtracted"]
    snr_values = summary["snr_values"]

    switch_on_us = summary["switch_on_time_s"] * 1e6
    max_time_us = summary["max_time_s"] * 1e6
    window_us = max_time_us - switch_on_us

    fig_data, ax_data = plt.subplots(1, 2, figsize=(9.2, 4.6))
    ax_data[0].plot(delay_us, signal_on, label="signal1 (mw on)")
    ax_data[0].plot(delay_us, signal_off, label="signal2 (mw off)")
    ax_data[0].set_title("Photon Rate vs Time")
    ax_data[0].set_xlabel("Time (us)")
    ax_data[0].set_ylabel("Counts/s")
    ax_data[0].legend()
    ax_data[0].grid(True, alpha=0.3)

    ax_data[1].plot(delay_us, subtracted, label="signal1 - signal2")
    ax_data[1].axvline(switch_on_us, linestyle=":", color="orange", linewidth=2, label="RO start")
    ax_data[1].axvline(max_time_us, linestyle="--", color="red", linewidth=2, label="RO end")
    ax_data[1].set_title("Subtracted Signal")
    ax_data[1].set_xlabel("Time (us)")
    ax_data[1].set_ylabel("Counts/s")
    ax_data[1].legend()
    ax_data[1].grid(True, alpha=0.3)

    fig_snr, ax_snr = plt.subplots(figsize=(7.6, 4.6))
    ax_snr.plot(delay_us, snr_values, label="SNR")
    ax_snr.axvline(switch_on_us, linestyle=":", color="orange", linewidth=2, label="RO start")
    ax_snr.axvline(max_time_us, linestyle="--", color="red", linewidth=2, label="RO end")
    ax_snr.set_xlabel("Integration Time (us)")
    ax_snr.set_ylabel("SNR")
    ax_snr.set_title("SNR vs Integration Window")
    ax_snr.grid(True, alpha=0.3)
    ax_snr.legend(loc="lower right")

    table_rows = [
        ("Readout Start", f"{switch_on_us:.3f} us"),
        ("Readout End", f"{max_time_us:.3f} us"),
        ("Readout Duration", f"{window_us:.3f} us"),
        ("Readout Window SNR", f"{summary['readout_window_snr']:.3f}"),
    ]
    left_width = max(len(k) for k, _ in table_rows)
    right_width = max(len(v) for _, v in table_rows)
    table_lines = [
        f"{'Optimal Parameters'.ljust(left_width)} | {'Value'.ljust(right_width)}",
        "-" * (left_width + right_width + 3),
    ]
    for k, v in table_rows:
        table_lines.append(f"{k.ljust(left_width)} | {v.rjust(right_width)}")

    ax_snr.text(
        0.02,
        0.98,
        "\n".join(table_lines),
        transform=ax_snr.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.3"},
    )

    fig_data.tight_layout()
    fig_snr.tight_layout()
    if show:
        plt.show()

    return fig_data, fig_snr

# =============================================================================
# Setup
# =============================================================================

cfg = load_config()
connect(cfg)

config, active_transition, _ = build_common_config(cfg, REPS, get_reference=GET_REFERENCE)
config.readout_integration_tns = int(READOUT_INTEGRATION_TNS)

# Create sweep array
delays_tns = np.arange(
    READOUT_OFFSET_START_TNS,
    READOUT_OFFSET_STOP_TNS + 1,
    READOUT_OFFSET_STEP_TNS,
)
print(
    f"[counting_duration] Sweep: {len(delays_tns)} points from "
    f"{READOUT_OFFSET_START_TNS} to {READOUT_OFFSET_STOP_TNS} ns"
)
print(f"[counting_duration] Reps: {REPS}, Integration: {READOUT_INTEGRATION_TNS} ns")
print(f"[counting_duration] Transition: {active_transition}")

# =============================================================================
# Acquire
# =============================================================================

data = _acquire_full_res(config, delays_tns, show_progress=ACQUIRE_PROGRESS)

summary = _compute_optimal_window(
    data,
    t_min_tus=T_MIN_TUS,
    threshold_pct=THRESHOLD_PCT,
)

switch_on_tus = summary["switch_on_time_s"] * 1e6
max_time_tus = summary["max_time_s"] * 1e6
window_tus = max_time_tus - switch_on_tus

print(
    f"[counting_duration] Optimal window: start={switch_on_tus:.3f} us, "
    f"duration={window_tus:.3f} us"
)

# =============================================================================
# Save to HDF5
# =============================================================================

custom_attrs = {
    "analysis": {
        "t_min_tus": float(T_MIN_TUS),
        "threshold_pct": float(THRESHOLD_PCT),
        "readout_start_tus": float(switch_on_tus),
        "readout_end_tus": float(max_time_tus),
        "readout_duration_tus": float(window_tus),
        "max_cumulative_snr": float(summary["max_snr"]),
        "readout_window_snr": float(summary["readout_window_snr"]),
    }
}

out_path, timestamp = save_experiment_hdf5(
    CountingDurationFineRes,
    config,
    cfg,
    data,
    OUTPUT_DIR,
    experiment_name="counting_duration_fine_res",
    sweep_axis_key="delay_tus",
    custom_attrs=custom_attrs,
)
run_id = out_path.stem

print(f"[counting_duration] Saved -> {out_path}")

# =============================================================================
# Plot
# =============================================================================

_plot_counting_duration(summary, show=True)
