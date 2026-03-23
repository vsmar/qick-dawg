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

    parts = []
    for key, value in metadata.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")

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


def plot_debug_traces(
    x_axis: np.ndarray,
    traces: Mapping[str, Optional[np.ndarray]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    metadata: Optional[Mapping[str, Any]] = None,
    metadata_position: str = "bottom",
    figsize: tuple[float, float] = (10.0, 4.8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot all available raw channels for debugging."""
    x_axis = np.asarray(x_axis, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    style = {
        "signal1": {"color": "tab:blue", "label": "signal1"},
        "signal2": {"color": "tab:orange", "label": "signal2"},
        "reference1": {"color": "tab:green", "label": "reference1"},
        "reference2": {"color": "tab:red", "label": "reference2"},
    }

    for key in ("signal1", "signal2", "reference1", "reference2"):
        arr = traces.get(key)
        if arr is None:
            continue
        ax.plot(
            x_axis,
            arr,
            "o-",
            markersize=4.0,
            linewidth=1.1,
            alpha=0.9,
            color=style[key]["color"],
            label=style[key]["label"],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)

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
            color="tab:green",
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
                color="tab:blue",
                label="signal1",
            )

        if signal2 is not None:
            lines += ax_right.plot(
                x_axis,
                signal2,
                "o-",
                markersize=4.0,
                linewidth=1.1,
                alpha=raw_alpha,
                color="tab:orange",
                label="signal2",
            )

    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel("Contrast ratio")
    if ax_right is not None:
        ax_right.set_ylabel("Counts/s")

    ax_left.set_title(title)
    ax_left.grid(alpha=0.25)

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
