"""
counting_duration_full_res.py
=============================
Run a counting-duration full-resolution sweep, save data, and generate:
1) Raw counts + subtracted signal
2) SNR vs integration time with readout-window annotations

Edit the EXPERIMENT PARAMETERS block before each run.
"""

from pathlib import Path
from copy import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from qickdawg.nvtestsuite.counting_duration_fine_res import CountingDurationFineRes

from config import (
    load_config,
    build_nv_config,
    connect,
    save_experiment_hdf5,
)

# =============================================================================
# EXPERIMENT PARAMETERS - edit these before each run
# =============================================================================

READOUT_OFFSET_START_TNS = 1000
READOUT_OFFSET_STOP_TNS = 2500
READOUT_OFFSET_STEP_TNS = 10

REPS = 200_000
READOUT_INTEGRATION_TNS = 10
GET_REFERENCE = True

THRESHOLD_PCT = 10.0
T_MIN_S = 0.0

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "counting_duration"


# =============================================================================
# Acquisition helpers
# =============================================================================

def _as_scalar(value):
    arr = np.asarray(value).squeeze()
    if arr.size == 1:
        return float(arr)
    return float(np.mean(arr))


def _acquire_full_res(config, delays_tns, show_progress=True):
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
            results.delay = np.empty(0, dtype=float)

        for key, value in d.items():
            results[key] = np.append(results[key], _as_scalar(value))

        results.delay = np.append(results.delay, config.laser_readout_offset_tus)

        sweep_iter.set_postfix({
            "offset_ns": int(delay_tns),
            "done": f"{idx}/{total_steps}",
        }, refresh=False)

    return results


def _results_to_dataframe(results):
    delay_us = np.asarray(results.delay, dtype=float)

    signal1_cts_s = np.asarray(
        getattr(results, "signal1_cts_s", getattr(results, "signal1")),
        dtype=float,
    )
    signal2_cts_s = np.asarray(
        getattr(results, "signal2_cts_s", getattr(results, "signal2")),
        dtype=float,
    )

    n = min(len(delay_us), len(signal1_cts_s), len(signal2_cts_s))
    return pd.DataFrame(
        {
            "delay_us": delay_us[:n],
            "signal1_cts_s": signal1_cts_s[:n],
            "signal2_cts_s": signal2_cts_s[:n],
        }
    )


# =============================================================================
# Analysis + plotting
# =============================================================================

def _extract_from_df(df):
    cd_times = df["delay_us"].to_numpy(dtype=float) * 1e-6  # us -> s
    signal_off = df["signal1_cts_s"].to_numpy(dtype=float)   # was ON before
    signal_on = df["signal2_cts_s"].to_numpy(dtype=float)    # was OFF before
    return signal_on, signal_off, cd_times


def optimize_counting_duration(df, t_min=0.0, threshold_pct=10.0, run_id=None):
    signal_on, signal_off, cd_times = _extract_from_df(df)

    mask = cd_times >= t_min
    cd_times = cd_times[mask] - t_min
    signal_on = signal_on[mask]
    signal_off = signal_off[mask]

    subtracted = signal_on - signal_off

    snr_values = []
    for t in cd_times:
        mask_t = cd_times <= t
        signal = np.trapezoid(subtracted[mask_t], cd_times[mask_t])
        noise = np.trapezoid(signal_on[mask_t] + signal_off[mask_t], cd_times[mask_t])
        snr_values.append(signal / np.sqrt(noise) if noise > 0 else 0.0)

    snr_values = np.asarray(snr_values, dtype=float)

    if len(snr_values) > 1:
        max_idx = int(np.argmax(snr_values[1:]) + 1)
    else:
        max_idx = 0

    max_time = float(cd_times[max_idx])
    max_snr = float(snr_values[max_idx])

    max_signal = float(np.max(subtracted))
    threshold = (threshold_pct / 100.0) * max_signal

    above_threshold = subtracted >= threshold
    if np.any(above_threshold):
        first_above_idx = int(np.where(above_threshold)[0][0])
        switch_idx = max(first_above_idx - 1, 0)
    else:
        switch_idx = 0

    switch_on_time = float(cd_times[switch_idx])
    switch_on_signal = float(subtracted[switch_idx])

    readout_mask = (cd_times >= switch_on_time) & (cd_times <= max_time)
    if np.count_nonzero(readout_mask) < 2:
        lo = min(switch_idx, max_idx)
        hi = max(switch_idx, max_idx)
        readout_mask = np.zeros_like(cd_times, dtype=bool)
        readout_mask[lo : hi + 1] = True

    signal_window = np.trapezoid(subtracted[readout_mask], cd_times[readout_mask])
    noise_window = np.trapezoid(
        signal_on[readout_mask] + signal_off[readout_mask],
        cd_times[readout_mask],
    )
    readout_window_snr = float(signal_window / np.sqrt(noise_window)) if noise_window > 0 else 0.0

    run_suffix = f" | run_id={run_id}" if run_id else ""

    fig_data, ax_data = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)
    if run_id:
        fig_data.suptitle(f"run_id={run_id}")

    ax_data[0].plot(cd_times, signal_on, label="signal 1 (on)")
    ax_data[0].plot(cd_times, signal_off, label="signal 2 (off)")
    ax_data[0].set_title("PL vs Time")
    ax_data[0].set_xlabel("Time (s)")
    ax_data[0].set_ylabel("Counts/s")
    ax_data[0].grid(True)
    ax_data[0].legend()

    ax_data[1].plot(cd_times, subtracted, label="on - off")
    ax_data[1].set_title("Subtracted Signal")
    ax_data[1].set_xlabel("Time (s)")
    ax_data[1].grid(True)
    ax_data[1].legend()

    fig_snr, ax_snr = plt.subplots(figsize=(7.2, 5.0))
    ax_snr.plot(cd_times, snr_values, label="SNR")
    ax_snr.axvline(switch_on_time, linestyle=":", color="orange", linewidth=2, label="RO start")
    ax_snr.axvline(max_time, linestyle="--", color="red", linewidth=2, label="RO end")

    table_rows = [
        ("Run ID", run_id if run_id else "N/A"),
        ("Readout Start", f"{switch_on_time * 1e6:.3f} us"),
        ("Readout End", f"{max_time * 1e6:.3f} us"),
        ("Readout Duration", f"{(max_time - switch_on_time) * 1e6:.3f} us"),
        ("Readout Window SNR", f"{readout_window_snr:.3f}"),
    ]
    left_width = max(len(k) for k, _ in table_rows)
    right_width = max(len(v) for _, v in table_rows)

    table_lines = [
        f"{'Optimal Parameters'.ljust(left_width)} | {'Value'.ljust(right_width)}",
        "-" * (left_width + right_width + 3),
    ]
    for k, v in table_rows:
        table_lines.append(f"{k.ljust(left_width)} | {v.rjust(right_width)}")
    table_text = "\n".join(table_lines)

    ax_snr.text(
        0.02,
        0.98,
        table_text,
        transform=ax_snr.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.3"},
    )

    ax_snr.set_xlabel("Integration Time (s)")
    ax_snr.set_ylabel("SNR")
    ax_snr.set_title(f"SNR vs Readout Duration{run_suffix}")
    ax_snr.legend(loc="lower right")
    ax_snr.grid(True)

    return {
        "switch_on_time": switch_on_time,
        "switch_on_signal": switch_on_signal,
        "max_time": max_time,
        "max_snr": max_snr,
        "readout_window_snr": readout_window_snr,
        "threshold": threshold,
        "fig_data": fig_data,
        "ax_data": ax_data,
        "fig_snr": fig_snr,
        "ax_snr": ax_snr,
    }


def _save_analysis_figures(analysis, out_path):
    out_path = Path(out_path)
    data_fig_path = out_path.with_name(f"{out_path.stem}_raw_and_subtracted.png")
    snr_fig_path = out_path.with_name(f"{out_path.stem}_snr.png")

    analysis["fig_data"].savefig(data_fig_path, dpi=300, bbox_inches="tight")
    analysis["fig_snr"].savefig(snr_fig_path, dpi=300, bbox_inches="tight")

    print(f"Saved figure: {data_fig_path}")
    print(f"Saved figure: {snr_fig_path}")


# =============================================================================
# Main execution
# =============================================================================

def main():
    cfg = load_config()
    connect(cfg)

    config = build_nv_config(cfg)
    config.readout_integration_tns = int(READOUT_INTEGRATION_TNS)
    config.reps = int(REPS)
    config.get_reference = bool(GET_REFERENCE)

    delays_tns = np.arange(
        READOUT_OFFSET_START_TNS,
        READOUT_OFFSET_STOP_TNS,
        READOUT_OFFSET_STEP_TNS,
    )

    results = _acquire_full_res(config, delays_tns)

    out_path, _ = save_experiment_hdf5(
        CountingDurationFineRes,
        config,
        cfg,
        results,
        OUTPUT_DIR,
        experiment_name="counting_duration_full_res",
    )
    run_id = out_path.stem

    df = _results_to_dataframe(results)
    analysis = optimize_counting_duration(
        df,
        t_min=float(T_MIN_S),
        threshold_pct=float(THRESHOLD_PCT),
        run_id=run_id,
    )
    _save_analysis_figures(analysis, out_path)

    plt.show()


if __name__ == "__main__":
    main()
