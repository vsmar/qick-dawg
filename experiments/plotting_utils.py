"""
Shared plotting utilities for NV experiment scripts.

This module standardizes two common data views:
1) Debug raw view with all available channels.
2) Twin-axis view with contrast on the left axis and signal/reference on the right.

It also provides a unified extractor that can switch between normalized
(cts/s) channels and raw channels.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt


# Shared palette across experiment plots.
SIGNAL_COLOR = "tab:blue"
REFERENCE_COLOR = "tab:orange"
CONTRAST_COLOR = "tab:green"
STEADY_SIGNAL_COLOR = "tab:purple"
STEADY_REFERENCE_COLOR = "tab:red"


def _get_channel_array(data: Any, base_name: str, use_counts_s: bool) -> Optional[np.ndarray]:
    """Extract one channel as a 1D float array if present, else None."""
    preferred = f"{base_name}_cts_s" if use_counts_s else base_name
    fallbacks = [base_name] if use_counts_s else [f"{base_name}_cts_s"]

    raw = getattr(data, preferred, None)
    if raw is None:
        for key in fallbacks:
            raw = getattr(data, key, None)
            if raw is not None:
                break

    if raw is None:
        return None

    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array for {preferred}, got shape {arr.shape}.")
    return arr


def extract_standard_traces(
    data: Any,
    x_axis: np.ndarray,
    use_counts_s: bool = True,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract canonical experiment traces and compute contrast when needed.

    Returned keys:
        signal1, signal2, reference1, reference2, contrast
    """
    x_axis = np.asarray(x_axis, dtype=float)
    if x_axis.ndim != 1:
        raise ValueError(f"Expected 1D x-axis, got shape {x_axis.shape}.")

    traces: Dict[str, Optional[np.ndarray]] = {
        "signal1": _get_channel_array(data, "signal1", use_counts_s),
        "signal2": _get_channel_array(data, "signal2", use_counts_s),
        "reference1": _get_channel_array(data, "reference1", use_counts_s),
        "reference2": _get_channel_array(data, "reference2", use_counts_s),
        "contrast": None,
    }

    contrast_raw = getattr(data, "contrast", None)
    if contrast_raw is not None:
        contrast = np.asarray(contrast_raw, dtype=float)
        if contrast.ndim != 1:
            raise ValueError(f"Expected 1D contrast data, got shape {contrast.shape}.")
        traces["contrast"] = contrast
    elif traces["signal1"] is not None and traces["signal2"] is not None:
        traces["contrast"] = traces["signal1"] / np.clip(traces["signal2"], 1e-12, None)

    for key, arr in traces.items():
        if arr is not None and len(arr) != len(x_axis):
            raise ValueError(
                f"Length mismatch for {key}: x has {len(x_axis)} points but {key} has {len(arr)}."
            )

    return traces


def format_metadata_lines(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Format metadata key-value pairs into a compact single-line annotation string."""
    if not metadata:
        return None

    hidden_keys = {"sequence", "xy8_phase_order"}

    key_label_map = {
        "pi2_ftns": r"$\pi/2$",
        "pi2_ftsamp": r"$\pi/2$ (samples)",
        "mw_pi2_ftns": r"mw $\pi/2$",
        "mw_pi2_ftsamp": r"mw $\pi/2$ (samples)",
        "pi_ftns": r"$\pi$",
        "pi_ftsamp": r"$\pi$ (samples)",
    }

    def pretty_key(key: str) -> str:
        if key in key_label_map:
            return key_label_map[key]

        # Drop ft* nomenclature in display labels for readability.
        label = key
        label = label.replace("_ftns", "_ns")
        label = label.replace("_ftus", "_us")
        label = label.replace("_ftsamp", "_samples")
        return label

    def pretty_pair(key: str, value: Any) -> str:
        label = pretty_key(key)
        # For pi pulse widths, prefer explicit physical units in the rendered text.
        if key in {"pi2_ftns", "mw_pi2_ftns", "pi_ftns"}:
            return f"{label}={value} ns"
        return f"{label}={value}"

    parts = []
    for key, value in metadata.items():
        if key in hidden_keys:
            continue
        if value is None:
            continue
        parts.append(pretty_pair(str(key), value))

    if not parts:
        return None
    return " | ".join(parts)


def _add_metadata_text(fig: plt.Figure, metadata_text: Optional[str], position: str) -> None:
    """Place metadata text at top or bottom margin."""
    if not metadata_text:
        return

    if position == "top":
        fig.text(0.5, 0.99, metadata_text, ha="center", va="top", fontsize=8.5)
    else:
        fig.text(0.5, 0.01, metadata_text, ha="center", va="bottom", fontsize=8.5)


def _add_sequence_text(ax: plt.Axes, sequence_text: Optional[str]) -> None:
    """Render a pulse-sequence annotation inside the plot area."""
    if not sequence_text:
        return

    ax.text(
        0.01,
        0.98,
        f"Sequence: {sequence_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )


def plot_debug_traces(
    x_axis: np.ndarray,
    traces: Mapping[str, Optional[np.ndarray]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    metadata: Optional[Mapping[str, Any]] = None,
    sequence_text: Optional[str] = None,
    metadata_position: str = "bottom",
    figsize: tuple[float, float] = (10.0, 4.8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot all available raw channels for debugging."""
    x_axis = np.asarray(x_axis, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    style = {
        "signal1": {"color": SIGNAL_COLOR, "label": "signal", "linestyle": "-"},
        "signal2": {"color": REFERENCE_COLOR, "label": "reference", "linestyle": "-"},
        "reference1": {
            "color": STEADY_SIGNAL_COLOR,
            "label": "steady state (signal)",
            "linestyle": "--",
        },
        "reference2": {
            "color": STEADY_REFERENCE_COLOR,
            "label": "steady state (reference)",
            "linestyle": "--",
        },
    }

    for key in ("signal1", "signal2", "reference1", "reference2"):
        arr = traces.get(key)
        if arr is None:
            continue
        ax.plot(
            x_axis,
            arr,
            "o",
            markersize=4.0,
            linewidth=1.1,
            alpha=0.9,
            color=style[key]["color"],
            linestyle=style[key]["linestyle"],
            label=style[key]["label"],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)
    _add_sequence_text(ax, sequence_text)

    metadata_text = format_metadata_lines(metadata)
    _add_metadata_text(fig, metadata_text, metadata_position)

    if metadata_text is not None:
        if metadata_position == "top":
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        else:
            fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

    return fig, ax


def plot_contrast_twin(
    x_axis: np.ndarray,
    traces: Mapping[str, Optional[np.ndarray]],
    *,
    x_label: str,
    title: str,
    metadata: Optional[Mapping[str, Any]] = None,
    sequence_text: Optional[str] = None,
    metadata_position: str = "bottom",
    figsize: tuple[float, float] = (10.0, 4.8),
    raw_alpha: float = 0.35,
    fit_x: Optional[np.ndarray] = None,
    fit_y: Optional[np.ndarray] = None,
    fit_label: str = "fit",
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes, Optional[plt.Axes], list[Any]]:
    """
    Plot contrast on left y-axis and signal/reference on right y-axis.

    Returns (fig, ax_left, ax_right, legend_lines).
    """
    x_axis = np.asarray(x_axis, dtype=float)
    contrast = traces.get("contrast")
    signal1 = traces.get("signal1")
    signal2 = traces.get("signal2")

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx() if (signal1 is not None or signal2 is not None) else None

    lines = []

    if contrast is not None:
        lines += ax_left.plot(
            x_axis,
            contrast,
            "s--",
            markersize=5.0,
            linewidth=1.2,
            alpha=0.9,
            color=CONTRAST_COLOR,
            label="contrast ratio",
        )

    if fit_x is not None and fit_y is not None:
        merged_fit_kwargs: Dict[str, Any] = {
            "color": "black",
            "linewidth": 2.0,
            "alpha": 0.95,
            "label": fit_label,
        }
        if fit_kwargs:
            merged_fit_kwargs.update(dict(fit_kwargs))
        lines += ax_left.plot(fit_x, fit_y, "-", **merged_fit_kwargs)

    if ax_right is not None:
        if signal1 is not None:
            lines += ax_right.plot(
                x_axis,
                signal1,
                "o-",
                markersize=4.0,
                linewidth=1.1,
                alpha=raw_alpha,
                color=SIGNAL_COLOR,
                label="signal",
            )

        if signal2 is not None:
            lines += ax_right.plot(
                x_axis,
                signal2,
                "o-",
                markersize=4.0,
                linewidth=1.1,
                alpha=raw_alpha,
                color=REFERENCE_COLOR,
                label="reference",
            )

    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel("Contrast ratio")
    if ax_right is not None:
        ax_right.set_ylabel("Counts/s")

    ax_left.set_title(title)
    ax_left.grid(alpha=0.25)
    _add_sequence_text(ax_left, sequence_text)

    if lines:
        labels = [line.get_label() for line in lines]
        ax_left.legend(lines, labels, loc="best", framealpha=0.95)

    metadata_text = format_metadata_lines(metadata)
    _add_metadata_text(fig, metadata_text, metadata_position)

    if metadata_text is not None:
        if metadata_position == "top":
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        else:
            fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

    return fig, ax_left, ax_right, lines
