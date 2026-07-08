"""
Instruction-level IR and user-facing scheduling API for QICK-DAWG.

Instruction types
-----------------
PlayIR      — play a pulse with optional per-call overrides
DelayIR     — insert a timed gap on a channel
TriggerIR   — fire a trigger / start ADC readout

Context managers
----------------
Sequence    — top-level container; all instructions live inside a Sequence
sweep()     — one tProc loop level over one or more co-varying SweepAxes
repeat()    — counted loop driven by a RepeatAxis (fixed or variable bound)

Parameter coercion
------------------
Every field that can be swept accepts:
    int | float       → constant Parameter in SI base units (s, Hz, deg)
    SweepAxis         → Parameter.from_axis()
    RepeatAxis        → Parameter.from_axis()
    Pattern           → Parameter.from_pattern()
    Parameter         → passed through unchanged
    None              → inherited from DefinePulse or left unset

Coercion happens at the IR boundary so all downstream layers only
ever handle Parameter — never bare numbers, axes, or raw Patterns.

Structural vs. value-handle split
----------------------------------
SweepAxis / RepeatAxis / Pattern describe the *structure* of register
updates across loop iterations (start/stop/step, a repeat bound, a
cyclic lookup table). They are not "current value" handles.

Parameter is the *value handle* an instruction actually reads: either
a fixed constant, or a reference to whatever a sweep/repeat/pattern is
currently driving into a register.

Accordingly:
    - SweepBlock / RepeatBlock (structural loop containers) hold the
      raw axis objects directly.
    - PlayIR / DelayIR (instructions) hold Parameter only, so every
      downstream consumer has one uniform type to deal with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from parameters import Parameter
from pulses import DefinePulse
from sweep_axis import Pattern, RepeatAxis, SweepAxis
from units import SI_DEFAULT_UNIT

_param_autoname = count()


# ---------------------------------------------------------------------------
# Parameter coercion
# ---------------------------------------------------------------------------

def _coerce_param(
    value: int | float | Parameter | SweepAxis | RepeatAxis | Pattern | None,
    kind: str,
    field_name: str,
) -> Parameter | None:
    """Normalize a user-supplied value to a Parameter, or None.

    Allows user to use the swept axis or pattern as a handle,
    by executing the appropriate conversion to a Parameter.

    - None          → None (caller interprets as "inherit" or "unset")
    - Parameter     → unchanged
    - Pattern       → Parameter.from_pattern()
    - SweepAxis     → Parameter.from_axis()
    - RepeatAxis    → Parameter.from_axis()
    - int | float   → Parameter.constant() in SI default unit for kind
    """
    if value is None:
        return None
    if isinstance(value, Parameter):
        return value
    if isinstance(value, Pattern):
        return Parameter.from_pattern(value)
    if isinstance(value, (SweepAxis, RepeatAxis)):
        return Parameter.from_axis(value)

    # bare number — wrap as SI constant
    unit = SI_DEFAULT_UNIT[kind]
    return Parameter.constant(
        name=f"_{field_name}_{next(_param_autoname)}",
        value=value,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# IR instruction types
# ---------------------------------------------------------------------------

@dataclass
class PlayIR:
    """Play a pulse on a DAC channel.

    amp: fraction of pulse amplitude ceiling [0, 2]
      - 1.0  universal ceiling for arbitrary waveforms
      - √2   maximum ceiling for arbitrary waveform (certain cases only 90 degree phases)
      - 2.0  ceiling for constant (non-waveform) pulses

    All of phase / amplitude / frequency are stored as Parameter after
    __post_init__ — including phase, even when the user supplied a
    Pattern (it gets wrapped via Parameter.from_pattern()).
    """
    pulse:       DefinePulse
    phase:       Parameter | None = None
    amplitude:   Parameter | None = None
    frequency:   Parameter | None = None
    channel:     int | None = None

    def __post_init__(self):
        if not isinstance(self.pulse, DefinePulse):
            raise TypeError(
                f"PlayIR: pulse is defined with the wrong type."
            )

        for name in ["phase", "amplitude", "frequency"]:
            val = getattr(self, name)
            if val is None:
                val = getattr(self.pulse, name)
            setattr(self, name, _coerce_param(val, name, name))

        if self.channel is None:
            self.channel = self.pulse.channel
        if not isinstance(self.channel, int) or self.channel < 0:
            raise TypeError(
                f"PlayIR: channel must be a non-negative int, got {self.channel!r}."
            )


@dataclass
class DelayIR:
    """Insert a timed gap on a channel.

    duration: time in seconds (bare number), or a Parameter / SweepAxis.
    """
    duration: Parameter
    channel:  int

    def __post_init__(self):
        self.duration = _coerce_param(self.duration, "time", "duration")
        if not isinstance(self.channel, int) or self.channel < 0:
            raise ValueError(
                f"DelayIR: channel must be a non-negative int, got {self.channel!r}."
            )


@dataclass
class TriggerIR:
    # TODO: Implement post init / constructor
    """Fire a digital trigger and optionally start an ADC readout."""
    duration:    int   # treg units
    pin:         int
    adc_channel: int
    readout:     bool = False


# ---------------------------------------------------------------------------
# IR block types
# ---------------------------------------------------------------------------

@dataclass
class SweepBlock:
    """One tProc loop level; all axes co-vary on the same loop counter.

    All supplied SweepAxes must have the same num_steps.
     # NOTE: Only considering the case of a single axis so far

    Holds the raw SweepAxis objects (structural), not Parameters —
    this block defines the loop, it isn't a value reference.
    """
    axes: tuple[SweepAxis, ...]
    body: list = field(default_factory=list)

    def __post_init__(self):
        if not self.axes:
            raise ValueError("sweep() requires at least one SweepAxis.")
        steps = {ax.num_steps for ax in self.axes}
        if len(steps) > 1:
            names = ", ".join(ax.name for ax in self.axes)
            raise ValueError(
                f"Co-varying SweepAxes must have the same num_steps; "
                f"[{names}] have steps {steps}."
            )

    @property  # TODO: requires SweepAxis.num_steps to be implemented in sweep_axis.py
    def num_steps(self) -> int:
        return self.axes[0].num_steps


@dataclass
class RepeatBlock:
    """Counted tProc loop driven by a RepeatAxis.

    The RepeatAxis carries its own bound (int or SweepAxis), so this block
    does not need a separate n argument. Holds the raw RepeatAxis
    (structural), not a Parameter.
    """
    axis: RepeatAxis
    body: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Active context state
# ---------------------------------------------------------------------------

_active_sequence: Sequence | None = None
_active_block_stack: list[SweepBlock | RepeatBlock] = []


# ---------------------------------------------------------------------------
# Sequence context manager
# ---------------------------------------------------------------------------

class Sequence:
    """Top-level container for all IR nodes.

    Usage:

        with Sequence() as seq:
            play(pi_pulse)
            delay(100e-9, channel=0)

            t = LinearSweepAxis.create("delay", 0, 100e-6, 1e-6, unit="s")
            with sweep(t):
                play(pi_pulse)
                delay(t, channel=0)
    """

    def __init__(self):
        self.instructions: list = []

    def __enter__(self):
        global _active_sequence
        if _active_sequence is not None:
            raise RuntimeError("Nested Sequence contexts are not supported.")
        _active_sequence = self
        return self

    def __exit__(self, *args):
        global _active_sequence
        _active_sequence = None

    def __repr__(self):
        lines = [f"Sequence ({len(self.instructions)} top-level nodes)"]
        for node in self.instructions:
            lines.append(f"  {node}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweep() context manager
# ---------------------------------------------------------------------------

class sweep:
    """Open one tProc loop level over one or more co-varying SweepAxes.

    Usage:

        t = LinearSweepAxis.create("delay", 0, 100e-6, 1e-6, unit="s")
        f = LinearSweepAxis.create("freq", 5e9, 5.1e9, 1e6, unit="Hz")

        # co-varying (one loop, two register updates per step)
        with sweep(t, f):
            play(pi_pulse, freq=f)
            delay(t, channel=0)

        # nested (two independent loops)
        with sweep(t):
            with sweep(f):
                play(pi_pulse, freq=f)
                delay(t, channel=0)
    """

    def __init__(self, *axes: SweepAxis):
        if not all(isinstance(ax, SweepAxis) for ax in axes):
            bad = [type(ax).__name__ for ax in axes if not isinstance(ax, SweepAxis)]
            raise TypeError(
                f"sweep() only accepts SweepAxis objects; got {bad}. "
                f"For structural loops use repeat() with a RepeatAxis."
            )
        self._block = SweepBlock(axes=tuple(axes))

    def __enter__(self):
        if _active_sequence is None:
            raise RuntimeError("sweep() must be used inside a Sequence.")
        _active_block_stack.append(self._block)
        return self

    def __exit__(self, *args):
        block = _active_block_stack.pop()
        _record(block)


# ---------------------------------------------------------------------------
# repeat() context manager
# ---------------------------------------------------------------------------

class repeat:
    """Open a counted tProc loop driven by a RepeatAxis.

    The RepeatAxis carries its own bound (fixed int or SweepAxis), so no
    separate n argument is needed here.

    Usage:

        # fixed repeat — always 8 iterations
        i = RepeatAxis("echo", bound=8)
        with repeat(i):
            delay(tau, channel=0)
            play(pi_pulse)
            delay(tau, channel=0)

        # variable repeat — N iterations where N is a swept SweepAxis
        N = LinearSweepAxis.create("N", 1, 32, 1, unit="count")
        i = RepeatAxis("pulse_index", bound=N)
        xy8 = Pattern.create([0, 90, 0, 90, 90, 0, 90, 0], unit="deg", axis=i)

        with sweep(N):
            with repeat(i):
                # xy8 is a Pattern (structural); play() coerces it into a
                # Parameter via Parameter.from_pattern() at the IR boundary.
                play(pi_pulse, phase=xy8)
    """

    def __init__(self, axis: RepeatAxis):
        if not isinstance(axis, RepeatAxis):
            raise TypeError(
                f"repeat() expects a RepeatAxis, got {type(axis).__name__}. "
                f"For parameter sweeps use sweep() with a SweepAxis."
            )
        # If bound is a SweepAxis, verify an enclosing sweep() owns it
        if isinstance(axis.bound, SweepAxis):
            enclosing = {
                ax
                for blk in _active_block_stack
                if isinstance(blk, SweepBlock)
                for ax in blk.axes
            }
            if axis.bound not in enclosing:
                raise RuntimeError(
                    f"RepeatAxis '{axis.name}' is bounded by SweepAxis "
                    f"'{axis.bound.name}', but no enclosing sweep() block "
                    f"owns that axis."
                )
        self._block = RepeatBlock(axis=axis)

    def __enter__(self):
        if _active_sequence is None:
            raise RuntimeError("repeat() must be used inside a Sequence.")
        _active_block_stack.append(self._block)
        return self

    def __exit__(self, *args):
        block = _active_block_stack.pop()
        _record(block)


# ---------------------------------------------------------------------------
# Internal record helper
# ---------------------------------------------------------------------------

def _record(node):
    """Append node to the innermost active block or the Sequence."""
    if _active_block_stack:
        _active_block_stack[-1].body.append(node)
    elif _active_sequence is not None:
        _active_sequence.instructions.append(node)
    else:
        raise RuntimeError(
            "Instructions must be created inside a Sequence context."
        )
    return node


# ---------------------------------------------------------------------------
# User-facing scheduling functions
# ---------------------------------------------------------------------------

def play(
    pulse: DefinePulse,
    *,
    phase:   int | float | Parameter | SweepAxis | Pattern | None = None,
    amplitude:     int | float | Parameter | SweepAxis | None = None,
    frequency:    int | float | Parameter | SweepAxis | None = None,
    channel: int | None = None,
) -> PlayIR:
    """Schedule a pulse.

    All keyword arguments accept plain numbers (SI units), a Parameter,
    a SweepAxis, a RepeatAxis, or (for phase) a Pattern.
    Omitted fields are inherited from the DefinePulse.

    NOTE: an earlier draft of this function also accepted an `offset`
    (relative start-time) argument, but PlayIR has no field to hold it
    yet — it's been dropped here rather than silently swallowed. Add a
    `start_offset` field to PlayIR (and thread it through here) once
    that timing semantic is defined.
    """
    return _record(PlayIR(
        pulse=pulse,
        phase=phase,
        amplitude=amplitude,
        frequency=frequency,
        channel=channel,
    ))


def delay(
    duration: int | float | Parameter | SweepAxis,
    channel:  int,
) -> DelayIR:
    """Insert a timed gap. duration accepts seconds, a Parameter, or a SweepAxis."""
    return _record(DelayIR(duration=duration, channel=channel))