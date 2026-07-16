"""
Simulation backend for QICK-DAWG.

Renders the affine timing IR (timing.TimedInstruction) into concrete
per-channel DAC output for ONE selected point of the sweep space, using the
NCO/mixing model from pulseview. No hardware creep: registers, waveform
memory layout, and instruction emission are deliberately out of scope.

Pipeline
--------
    Sequence
      -> timing._schedule            (affine times, program order preserved)
      -> concretize()                pick sweep iterations, EXPAND repeats,
                                     resolve Parameters -> pulseview.PulseIR
      -> stack shots                 sweep bodies are independent shots; when
                                     the cursor reset makes later times regress,
                                     lay the next shot after the previous one
      -> render()                    per-channel NCO evolution + mixing
      -> SimResult.plot()

Semantics of "one selected point"
---------------------------------
  * SweepAxis   — you choose ONE iteration index k in [0, num_steps]
                  (num_steps excludes the initial value, so there are
                  num_steps + 1 points). The rendered timeline is the single
                  shot the hardware would run at that k.
  * RepeatAxis  — repeats happen *within* a shot, so they are fully expanded:
                  every TimedInstruction carrying a repeat counter term is
                  instantiated once per iteration. A Pattern indexed by that
                  axis resolves per iteration (values[i % len]).
  * Pattern     — resolved through its driving axis's counter value.

Shot stacking heuristic
-----------------------
timing_pass resets the per-channel cursor to 0 after a SweepBlock (independent
shots), so instructions after a sweep can have base times that numerically
overlap the sweep body. We detect a shot boundary when an instruction on some
channel starts BEFORE that channel's previous instruction ended (in raw
evaluated time) and restart the global timeline at the aligned end of
everything rendered so far. A robust fix would be an explicit reset marker on
TimedInstruction / a shot id from the timing pass; this heuristic covers the
common prelude / sweep / readout structure for now.

Unit inversion
--------------
Parameters store canonical hardware units (freg / preg / ftsamp). The
simulator needs Hz / deg for pulseview, so UnitModel provides the inverses.
The defaults match sim_compat's mock soccfg (32-bit words over FREQ_DAC);
when running against real hardware, construct UnitModel from qd.soccfg's
reg2freq / reg2deg so the round trip is exact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import count as _count, product

import numpy as np
import matplotlib.pyplot as plt

from .parameters import Parameter
from .pulseview import (
    FREQ_DAC,
    EnvelopeIR,
    NCOState,
    PulseIR,
    calculate_fcw,
    evolve_nco,
    mix_pulse,
)
from .sweep_axis import (
    ExponentialSweepAxis,
    LinearSweepAxis,
    Pattern,
    RepeatAxis,
    SweepAxis,
)
from .timing import TimedInstruction, _schedule
from .units import get_soccfg

FTSAMP_PER_TREG = 16


# ---------------------------------------------------------------------------
# Unit inversion (canonical hardware units -> physical, for the NCO model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitModel:
    """Inverse conversions canonical -> physical.

    By default the inverses come from the ACTIVE soccfg (units.get_soccfg):
    the board config's reg2freq / reg2deg when qickdawg is connected, or
    SoftSocCfg's software formulas otherwise -- so forward conversion and
    simulation round-trip through the same arithmetic either way. Explicit
    freg_to_hz / preg_to_deg callables override; the bit-width fields are a
    last-resort fallback for soccfgs lacking the inverse methods.
    """
    freq_dac: float = FREQ_DAC
    freq_bits: int = 32
    phase_bits: int = 32
    freg_to_hz: callable = None
    preg_to_deg: callable = None

    def hz(self, freg: int) -> float:
        if self.freg_to_hz is not None:
            return float(self.freg_to_hz(freg))
        cfg = get_soccfg()
        if hasattr(cfg, "reg2freq"):
            return float(cfg.reg2freq(freg)) * 1e6     # qick reg2freq is MHz
        return freg / 2**self.freq_bits * self.freq_dac

    def deg(self, preg: int) -> float:
        if self.preg_to_deg is not None:
            return float(self.preg_to_deg(preg))
        cfg = get_soccfg()
        if hasattr(cfg, "reg2deg"):
            return float(cfg.reg2deg(preg))
        return (preg % 2**self.phase_bits) * 360.0 / 2**self.phase_bits

    def us(self, ftsamp: int) -> float:
        return ftsamp / self.freq_dac * 1e6


# ---------------------------------------------------------------------------
# Axis / Parameter resolution at a chosen counter assignment
# ---------------------------------------------------------------------------

_EXP_FACTORS = {"3/2": (3, 2), "5/4": (5, 4), "9/8": (9, 8), "17/16": (17, 16)}


def axis_value(axis: SweepAxis, k: int):
    """Canonical value of a SweepAxis at iteration k (k=0 is the start)."""
    if isinstance(axis, LinearSweepAxis):
        return axis.start + k * axis.step
    if isinstance(axis, ExponentialSweepAxis):
        num, den = _EXP_FACTORS[axis.factor]
        x = axis.start
        for _ in range(k):
            x = x * num // den
        return x
    raise TypeError(f"Cannot evaluate axis type {type(axis).__name__}.")


def format_axis_point(axis: SweepAxis, k: int, um: "UnitModel") -> str:
    """Human-readable value of a sweep axis at iteration k."""
    v = axis_value(axis, k)
    if axis.kind == "time":
        return f"{axis.name}={um.us(v):.4g} us"
    if axis.kind == "coarse_time":
        return f"{axis.name}={v} treg ({um.us(v * FTSAMP_PER_TREG):.4g} us)"
    if axis.kind == "frequency":
        return f"{axis.name}={um.hz(v) / 1e6:.6g} MHz"
    return f"{axis.name}[{k}]={v}"


def resolve_parameter(param: Parameter | None, counters: dict):
    """Canonical value of a Parameter under a counter assignment.

    counters maps axis object -> iteration index (sweep index or repeat
    iteration), exactly the keying AffineTime.evaluate uses.
    """
    if param is None:
        return None
    if param.is_constant:
        return param._constant_value

    ax = param.axis
    if isinstance(ax, Pattern):
        i = counters[ax.axis]
        return ax.values[i % len(ax.values)]
    if isinstance(ax, SweepAxis):
        return axis_value(ax, counters[ax])
    if isinstance(ax, RepeatAxis):
        return counters[ax]          # the counter *is* the value (count kind)
    raise TypeError(f"Cannot resolve Parameter driven by {type(ax).__name__}.")


# ---------------------------------------------------------------------------
# Sweep-space introspection helpers
# ---------------------------------------------------------------------------

def flattened_events(sequence) -> list[tuple[int, TimedInstruction]]:
    """Run the timing pass but keep the flat, program-ordered event list
    (channel interleaving intact) — needed for shot-boundary detection."""
    events, _ = _schedule(sequence.instructions, _count())
    return events


def _axes_of(timed: TimedInstruction):
    """All axes an instruction depends on: timing terms + parameter drivers."""
    axes = [ax for ax, _ in timed.t_start.terms()]
    axes += [ax for ax, _ in timed.t_end.terms()]
    instr = timed.instruction
    for p in (instr.phase, instr.amplitude, instr.frequency):
        if p is not None and not p.is_constant:
            d = p.driving_axis
            if d is not None:
                axes.append(d)
    # de-dup by identity, order-stable
    seen, out = set(), []
    for ax in axes:
        if id(ax) not in seen:
            seen.add(id(ax))
            out.append(ax)
    return out


def sweep_axes(sequence_or_events) -> list[SweepAxis]:
    """Every SweepAxis the program depends on (choose an index for each)."""
    events = (sequence_or_events if isinstance(sequence_or_events, list)
              else flattened_events(sequence_or_events))
    seen, out = set(), []
    for _, timed in events:
        for ax in _axes_of(timed):
            if isinstance(ax, RepeatAxis) and isinstance(ax.bound, SweepAxis):
                ax = ax.bound
            if isinstance(ax, SweepAxis) and id(ax) not in seen:
                seen.add(id(ax))
                out.append(ax)
    return out


# ---------------------------------------------------------------------------
# Concretization: TimedInstruction -> concrete pulseview.PulseIR
# ---------------------------------------------------------------------------

@dataclass
class ConcretePulse:
    """A fully resolved pulse instance plus provenance for plotting."""
    pulse: PulseIR
    name: str
    segment_id: int
    shot: int = 0                    # filled in by shot stacking


def _repeat_bound(axis: RepeatAxis, sweep_indices: dict) -> int:
    if isinstance(axis.bound, int):
        return axis.bound
    # variable bound: N is the *value* of the sweep axis at the chosen index
    if axis.bound not in sweep_indices:
        raise ValueError(
            f"RepeatAxis '{axis.name}' is bounded by SweepAxis "
            f"'{axis.bound.name}'; provide an index for it in sweep_indices."
        )
    return int(axis_value(axis.bound, sweep_indices[axis.bound]))


def concretize(
    events: list[tuple[int, TimedInstruction]],
    sweep_indices: dict | None,
    unit_model: UnitModel,
) -> list[ConcretePulse]:
    """Evaluate every event at the chosen sweep point, expanding repeats.

    Returned list preserves program order (repeat iterations in ascending
    order), with raw evaluated times — shot stacking happens afterwards.
    """
    sweep_indices = dict(sweep_indices or {})

    # validate the sweep point ------------------------------------------------
    needed = sweep_axes(events)
    missing = [ax for ax in needed if ax not in sweep_indices]
    if missing:
        names = ", ".join(f"'{ax.name}'" for ax in missing)
        raise ValueError(
            f"simulate: no iteration index chosen for sweep axes {names}. "
            f"Pass sweep_indices={{axis: k}} with k in [0, axis.num_steps]."
        )
    for ax, k in sweep_indices.items():
        if not (0 <= k <= ax.num_steps):
            raise ValueError(
                f"Sweep index {k} for axis '{ax.name}' outside "
                f"[0, {ax.num_steps}] ({ax.num_steps + 1} points)."
            )

    concrete: list[ConcretePulse] = []

    for ch, timed in events:
        instr = timed.instruction
        deps = _axes_of(timed)

        repeat_axes = [ax for ax in deps if isinstance(ax, RepeatAxis)]
        bounds = [_repeat_bound(ax, sweep_indices) for ax in repeat_axes]

        for combo in product(*(range(b) for b in bounds)):
            counters = dict(sweep_indices)
            counters.update(zip(repeat_axes, combo))

            t0 = timed.t_start.evaluate(counters)
            t1 = timed.t_end.evaluate(counters)

            freq_c = resolve_parameter(instr.frequency, counters)
            if freq_c is None:
                raise ValueError(
                    f"Pulse '{instr.pulse.name}' has no frequency (neither on "
                    f"the DefinePulse nor the play() call)."
                )
            phase_c = resolve_parameter(instr.phase, counters)
            amp_c = resolve_parameter(instr.amplitude, counters)

            if instr.pulse.waveform_mode:
                data = instr.pulse.shape.data.astype(complex)
                if len(data) != t1 - t0:
                    raise ValueError(
                        f"Pulse '{instr.pulse.name}': shape length {len(data)}"
                        f" != scheduled duration {t1 - t0} ftsamp."
                    )
            else:
                # coarse (tProc-counted) pulse: flat envelope over the
                # duration EVALUATED at this sweep point -- this is where a
                # swept length_coarse becomes a concrete per-iteration length
                data = np.ones(t1 - t0, dtype=complex)
            env = EnvelopeIR(name=instr.pulse.name,
                             idata=np.real(data), qdata=np.imag(data))

            concrete.append(ConcretePulse(
                pulse=PulseIR(
                    ch=ch,
                    start_ftsamp=int(t0),
                    end_ftsamp=int(t1),
                    duration_ftsamp=int(t1 - t0),
                    envelope=env,
                    freq=unit_model.hz(freq_c),
                    phase=unit_model.deg(phase_c) if phase_c is not None else 0.0,
                    amp=float(amp_c) if amp_c is not None else 1.0,
                ),
                name=instr.pulse.name,
                segment_id=timed.segment_id,
            ))

    return concrete


# ---------------------------------------------------------------------------
# Shot stacking (sweep cursor resets -> sequential shots on one timeline)
# ---------------------------------------------------------------------------

def _stack_shots(concrete: list[ConcretePulse]) -> tuple[list[ConcretePulse], list[int]]:
    """Offset later shots past earlier ones when raw times regress.

    Returns (restacked pulses, shot start marks in global ftsamp).
    Relative timing *within* a shot is never altered.
    """
    offset = 0
    shot = 0
    global_end = 0
    last_end_on: dict[int, int] = {}     # channel -> raw end within current shot
    marks: list[int] = []
    out: list[ConcretePulse] = []

    for cp in concrete:                  # program order
        ch = cp.pulse.ch
        raw0, raw1 = cp.pulse.start_ftsamp, cp.pulse.end_ftsamp
        if raw0 < last_end_on.get(ch, 0):
            # per-channel time went backwards -> cursor reset -> new shot
            shot += 1
            offset = -(-global_end // FTSAMP_PER_TREG) * FTSAMP_PER_TREG
            marks.append(offset)
            last_end_on = {}
        last_end_on[ch] = raw1
        g0, g1 = raw0 + offset, raw1 + offset
        global_end = max(global_end, g1)
        p = cp.pulse
        out.append(ConcretePulse(
            pulse=PulseIR(ch=ch, start_ftsamp=g0, end_ftsamp=g1,
                          duration_ftsamp=p.duration_ftsamp, envelope=p.envelope,
                          freq=p.freq, phase=p.phase, amp=p.amp,
                          phase_reset=p.phase_reset),
            name=cp.name, segment_id=cp.segment_id, shot=shot,
        ))
    return out, marks


# ---------------------------------------------------------------------------
# Rendering: NCO evolution + mixing per channel
# ---------------------------------------------------------------------------

def render(concrete: list[ConcretePulse]) -> tuple[dict[int, np.ndarray],
                                                   dict[int, np.ndarray]]:
    """Per-channel RF output and |envelope| outline over one global timeline."""
    if not concrete:
        return {}, {}
    total = max(cp.pulse.end_ftsamp for cp in concrete)

    by_ch: dict[int, list[ConcretePulse]] = {}
    for cp in concrete:
        by_ch.setdefault(cp.pulse.ch, []).append(cp)

    rf: dict[int, np.ndarray] = {}
    env: dict[int, np.ndarray] = {}

    for ch, plist in by_ch.items():
        plist.sort(key=lambda cp: cp.pulse.start_ftsamp)
        for a, b in zip(plist, plist[1:]):
            if b.pulse.start_ftsamp < a.pulse.end_ftsamp:
                warnings.warn(
                    f"Channel {ch}: pulses '{a.name}' and '{b.name}' overlap "
                    f"({a.pulse.end_ftsamp} > {b.pulse.start_ftsamp}); "
                    f"summing their outputs."
                )

        out = np.zeros(total)
        outline = np.zeros(total)

        # NCO phase-continuous within the timeline; FCW updates at pulse start
        nco = NCOState(time_ftsamp=0,
                       fcw=calculate_fcw(plist[0].pulse.freq),
                       accum_phase=0)
        for cp in plist:
            p = cp.pulse
            fcw = calculate_fcw(p.freq)
            if fcw != nco.fcw or p.phase_reset:
                nco = evolve_nco(nco, current_sample=p.start_ftsamp,
                                 new_fcw=fcw, phase_reset=p.phase_reset)
            out[p.start_ftsamp:p.end_ftsamp] += mix_pulse(p, nco)
            mag = p.amp * np.abs(p.envelope.idata + 1j * p.envelope.qdata)
            outline[p.start_ftsamp:p.end_ftsamp] = np.maximum(
                outline[p.start_ftsamp:p.end_ftsamp], mag)

        rf[ch] = out
        env[ch] = outline

    return rf, env


# ---------------------------------------------------------------------------
# Result container + plotting
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    rf: dict[int, np.ndarray]
    envelope: dict[int, np.ndarray]
    pulses: list[ConcretePulse]
    shot_marks: list[int]
    sweep_point: dict
    unit_model: UnitModel

    @property
    def total_ftsamp(self) -> int:
        return max((cp.pulse.end_ftsamp for cp in self.pulses), default=0)

    def describe(self) -> str:
        lines = []
        pt = ", ".join(
            f"{format_axis_point(ax, k, self.unit_model)}"
            f" [{ax.canonical_unit} idx {k}]"
            for ax, k in self.sweep_point.items()
        )
        lines.append(f"Sweep point: {pt or '(none)'}")
        lines.append(f"Timeline: {self.total_ftsamp} ftsamp "
                     f"({self.unit_model.us(self.total_ftsamp):.3f} us), "
                     f"{len(self.shot_marks) + 1} shot(s)")
        for cp in self.pulses:
            p = cp.pulse
            lines.append(
                f"  shot{cp.shot} seg{cp.segment_id} ch{p.ch}: {cp.name:>10s} "
                f"@ [{p.start_ftsamp}, {p.end_ftsamp}) ftsamp  "
                f"f={p.freq / 1e6:.3f} MHz  ph={p.phase:.1f} deg  amp={p.amp:.2f}"
            )
        return "\n".join(lines)

    def plot(self, show_envelope: bool = True, annotate: bool = True,
             save: str | None = None, show: bool = False):
        channels = sorted(self.rf)
        if not channels:
            raise ValueError("Nothing to plot: no pulses were rendered.")

        fig, axes = plt.subplots(len(channels), 1, sharex=True,
                                 figsize=(11, 2.2 * len(channels)),
                                 squeeze=False)
        t_us = None
        for row, ch in enumerate(channels):
            ax = axes[row][0]
            sig = self.rf[ch]
            if t_us is None:
                t_us = np.arange(len(sig)) / self.unit_model.freq_dac * 1e6
            ax.plot(t_us, sig, lw=0.6, color="C0")
            if show_envelope:
                ax.fill_between(t_us, self.envelope[ch], -self.envelope[ch],
                                color="C1", alpha=0.18, lw=0)
            for m in self.shot_marks:
                ax.axvline(self.unit_model.us(m), color="k", ls="--",
                           lw=0.8, alpha=0.5)
            if annotate:
                for cp in self.pulses:
                    if cp.pulse.ch != ch:
                        continue
                    mid = self.unit_model.us(
                        (cp.pulse.start_ftsamp + cp.pulse.end_ftsamp) // 2)
                    ax.annotate(cp.name, (mid, 1.02), ha="center", fontsize=7,
                                xycoords=("data", "axes fraction"),
                                annotation_clip=False)
            ax.set_ylabel(f"ch {ch}")
            peak = max(1.0, float(np.max(self.envelope[ch], initial=0.0)))
            ax.set_ylim(-1.15 * peak, 1.15 * peak)
            ax.grid(alpha=0.25)
        axes[-1][0].set_xlabel("time (us)")

        pt = ", ".join(format_axis_point(ax, k, self.unit_model)
                       for ax, k in self.sweep_point.items())
        fig.suptitle(f"Simulated output  ({pt})" if pt else "Simulated output",
                     fontsize=10)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150)
        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def simulate(sequence, sweep_indices: dict | None = None,
             unit_model: UnitModel | None = None,
             stack_shots: bool = True) -> SimResult:
    """Simulate one point of the sweep space of a Sequence.

    sequence      : ir.Sequence (already built)
    sweep_indices : {SweepAxis: iteration index}; use simulate.sweep_axes(seq)
                    to discover which axes need one. May be omitted only when
                    the program has no sweeps.
    unit_model    : canonical->physical inverses; defaults to the mock model
                    (consistent with sim_compat).
    stack_shots   : lay post-reset instructions after the previous shot instead
                    of letting raw times overlap (see module docstring).
    """
    um = unit_model or UnitModel()
    events = flattened_events(sequence)
    concrete = concretize(events, sweep_indices, um)

    if stack_shots:
        concrete, marks = _stack_shots(concrete)
    else:
        marks = []

    rf, env = render(concrete)
    return SimResult(rf=rf, envelope=env, pulses=concrete, shot_marks=marks,
                     sweep_point=dict(sweep_indices or {}), unit_model=um)