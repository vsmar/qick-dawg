from copy import copy
import qickdawg as qd

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import time

from scipy.optimize import curve_fit


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def initialize():
    qd.start_client('192.168.3.1')

    default_config = qd.NVConfiguration()
    default_config.adc_channel = 0
    default_config.edge_counting = True
    default_config.high_threshold = 8000
    default_config.low_threshold = 500
    default_config.mw_channel = 0
    default_config.mw_nqz = 1
    default_config.mw_gain = 12000
    default_config.laser_gate_pmod = 0
    default_config.relax_delay_tns = 50

    # PL program
    config_PL = copy(default_config)
    config_PL.readout_integration_treg = 2**16 - 1
    config_PL.reps = 10000
    prog_PL = qd.PLIntensity(config_PL)

    # ODMR program
    config_ODMR = copy(default_config)
    config_ODMR.readout_integration_tus = qd.max_int_time_tus
    config_ODMR.mw_gain = 12000
    config_ODMR.pre_init = True 
    config_ODMR.reps = 5000
    config_ODMR.relax_delay_treg = 300
    config_ODMR.add_linear_sweep('mw', 'fMHz', start=2200, stop=2900, delta=2) #########################################################################
    prog_ODMR = qd.LockinODMR(config_ODMR)

    return default_config, prog_PL, prog_ODMR

# ------------------------------------------------------------
# Lorentzian + fitting
# ------------------------------------------------------------
def lorentzian(f, baseline, log_amp, f0, gamma):
    A = np.exp(log_amp)
    return baseline - A / (1 + ((f - f0)/gamma)**2)


def fit_lorentzian(freqs, contrast):
    baseline0 = np.median(contrast)
    amp0 = np.log(np.max(contrast) - np.min(contrast) + 1e-6)
    f0_0 = freqs[np.argmin(contrast)]
    gamma0 = (freqs[-1] - freqs[0]) / 20

    p0 = [baseline0, amp0, f0_0, gamma0]

    try:
        params, cov = curve_fit(
            lorentzian,
            freqs,
            contrast,
            p0=p0,
            maxfev=5000
        )
        f0 = params[2]
        peak_val = lorentzian(f0, *params)

        return {"success": True, "f0": f0, "peak_val": peak_val, "params": params}

    except Exception:
        idx = np.argmin(contrast)
        return {"success": False, "f0": freqs[idx], "peak_val": contrast[idx]}


# ------------------------------------------------------------
# Saving + plotting
# ------------------------------------------------------------
def save_odmr_csv(run_dir, index, freqs, signal, reference, cps, timestamp):
    df = pd.DataFrame({
        "frequency_MHz": freqs,
        "signal": signal,
        "reference": reference,
        "ratio": signal / reference,
        "cps_prior": cps,
        "timestamp": timestamp
    })
    df.to_csv(os.path.join(run_dir, f"odmr_{index:03d}.csv"), index=False)


def plot_single_odmr(run_dir, index, freqs, signal, reference, cps, timestamp):
    contrast = signal / reference
    fit = fit_lorentzian(freqs, contrast)

    peak_freq = fit["f0"]
    peak_val = fit["peak_val"]

    plt.figure(figsize=(7,5))
    plt.plot(freqs, contrast, 'k.', label="Data")

    if fit["success"]:
        f_fit = np.linspace(freqs.min(), freqs.max(), 1000)
        y_fit = lorentzian(f_fit, *fit["params"])
        plt.plot(f_fit, y_fit, 'r-', label="Lorentzian fit")

    plt.annotate(
        f"{peak_freq:.2f} MHz",
        xy=(peak_freq, peak_val),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="red"
    )

    plt.title(f"ODMR #{index:03d}")
    plt.suptitle(f"Timestamp: {timestamp} | CPS prior: {cps:.2f}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Contrast (arb.)")
    plt.legend()

    plt.savefig(os.path.join(run_dir, f"odmr_{index:03d}.png"), dpi=200)
    plt.close()

    return peak_freq


def plot_all_odmrs(run_dir, all_data):
    plt.figure(figsize=(8,6))

    for entry in all_data:
        freqs = entry["freqs"]
        ratio = entry["signal"] / entry["reference"]
        label = f"#{entry['index']:03d} ({entry['cps']:.1f} cps)"
        plt.plot(freqs, ratio, label=label)

    plt.title("All ODMRs")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("MW / Reference")
    plt.legend(fontsize=8)

    plt.savefig(os.path.join(run_dir, "all_odmrs.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main():
    default_config, prog_PL, prog_ODMR = initialize()

    run_dir = f"overnight_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(run_dir, exist_ok=True)

    all_odmr_data = []
    num_scans = 12   # 6 hours, 20 min interval

    for idx in range(num_scans):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        qd.laser_on(default_config)
        time.sleep(3)

        counts = prog_PL.acquire() / 10000
        cps = counts / qd.max_int_time_treg / qd.min_time_tns * 1e9

        qd.laser_off(default_config)

        d = prog_ODMR.acquire()
        freqs = d.frequencies
        signal = d.signal
        reference = d.reference

        save_odmr_csv(run_dir, idx, freqs, signal, reference, cps, timestamp)
        plot_single_odmr(run_dir, idx, freqs, signal, reference, cps, timestamp)

        all_odmr_data.append({
            "index": idx,
            "freqs": freqs,
            "signal": signal,
            "reference": reference,
            "cps": cps,
            "timestamp": timestamp
        })

        print(f"Completed ODMR {idx+1}/{num_scans} at {timestamp}")
        time.sleep(5 * 60) #########################################################################

    # Save combined CSV
    combined = []
    for entry in all_odmr_data:
        for f, s, r in zip(entry["freqs"], entry["signal"], entry["reference"]):
            combined.append({
                "index": entry["index"],
                "timestamp": entry["timestamp"],
                "cps_prior": entry["cps"],
                "frequency_MHz": f,
                "signal": s,
                "reference": r,
                "ratio": s/r
            })

    pd.DataFrame(combined).to_csv(os.path.join(run_dir, "all_odmrs.csv"), index=False)
    plot_all_odmrs(run_dir, all_odmr_data)


if __name__ == "__main__":
    main()