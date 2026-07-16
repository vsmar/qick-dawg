"""
Affine timing IR and timing pass for QICK-DAWG.

Purpose
-------
Lower the nested user IR (Sequence of PlayIR / DelayIR / SweepBlock /
RepeatBlock) into a per-channel list of TimedInstruction, where each
absolute start/end time is an *affine form* over the enclosing loop
counters rather than a plain integer.

    t_start(i, k, ...) = base + Sigma_counters (coeff * counter)

Restricting to affine forms (constant coefficients) is what lets the tProc
maintain the schedule with a single register add per loop tick, and what
lets a sweep run as a real hardware loop instead of being unrolled.

Cursor semantics (confirmed)
----------------------------
  * repeat(i)  -- sequential within ONE shot. Iteration k of the body is
                  offset by k * (body duration). The repeat counter gets a
                  static coefficient, affine only if the body duration is a
                  compile-time constant. A swept delay inside a repeat makes
                  the body duration vary -> k*(varying) is bilinear -> rejected.

  * sweep(t)   -- INDEPENDENT shots. The per-channel cursor RESETS at the top
                  of each iteration (readout at the sweep boundary forces a
                  return to treg gridtime, acting as an effective reset). A
                  swept delay does not accumulate across iterations; it only
                  places the following pulse within the current iteration,
                  staying affine in the sweep counter. The body is emitted
                  once, parameterized by the counter, and the loop replays it.

Segments
--------
Every maximal straight-line run gets a unique segment id. The id increments
at each block boundary. Downstream grouping (supernoding) may only fuse
pulses that share a segment id -- this is what keeps a sweep's post-reset
segment (whose base times restart at 0) from being compared against, or
fused with, pulses from a different segment.

Swept pulse lengths
-------------------
Constant-amplitude coarse-mode pulses are counted out by the tProc in treg,
so their length may be swept. A LinearSweepAxis length is affine (start +
i*step, scaled to ftsamp) and is handled exactly like a swept delay: the
pulse END and the cursor after it both carry the counter term. Inside a
repeat() this makes the body duration swept, which the existing bilinear
check rejects. Waveform (shaped) pulses remain constant-length.

Rejected (non-affine) cases -- raised as ValueError
---------------------------------------------------
  * exponential swept delay OR pulse length  (geometric accumulation) # NOTE: could unroll or modify exponential sweep to have a pseudo-affine form
  * swept delay / swept pulse length inside a repeat()  (bilinear)
  * Pattern as a delay duration     (patterns are phase-only anyway)
"""

# NOTE: Forcing cursor time at sweeps might not be good, really this forced gridding happens at readouts we dont need to loop over

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from .ir import DelayIR, PlayIR, RepeatBlock, Sequence, SweepBlock, TriggerIR
from .parameters import Parameter
from .units import FTSAMP_PER_TREG
from .sweep_axis import (
    ExponentialSweepAxis,
    LinearSweepAxis,
    RepeatAxis,
    SweepAxis,
)


# ---------------------------------------------------------------------------
# Affine time
# ---------------------------------------------------------------------------
@dataclass
class AffineTime:
    """Absolute time in ftsamp as an affine form over loop counters.

        value = base + Sigma (coeff * counter)

    Counters are keyed by object *identity*: one axis object is one loop,
    however many times referenced, and structurally-equal-but-distinct axes
    stay separate (matches Parameter.driving_axis).
    """
    base: int = 0
    _terms: dict = field(default_factory=dict)   # id(axis) -> (axis, coeff)

    # -- constructors / combinators -----------------------------------------

    @staticmethod
    def const(n: int) -> "AffineTime":
        return AffineTime(base=int(n), _terms={})

    def shifted(self, dt: int) -> "AffineTime":
        return AffineTime(base=self.base + int(dt), _terms=dict(self._terms))

    def plus_counter(self, axis, coeff: int) -> "AffineTime":
        if coeff == 0:
            return AffineTime(base=self.base, _terms=dict(self._terms))
        terms = dict(self._terms)
        key = id(axis)
        prev_axis, prev_coeff = terms.get(key, (axis, 0))
        new_coeff = prev_coeff + int(coeff)
        if new_coeff == 0:
            terms.pop(key, None)
        else:
            terms[key] = (prev_axis, new_coeff)
        return AffineTime(base=self.base, _terms=terms)

    def scaled(self, k: int) -> "AffineTime":
        terms = {}
        for key, (axis, coeff) in self._terms.items():
            if coeff * k != 0:
                terms[key] = (axis, coeff * k)
        return AffineTime(base=self.base * k, _terms=terms)

    def __add__(self, other: "AffineTime") -> "AffineTime":
        terms = dict(self._terms)
        for key, (axis, coeff) in other._terms.items():
            prev_axis, prev_coeff = terms.get(key, (axis, 0))
            new_coeff = prev_coeff + coeff
            if new_coeff == 0:
                terms.pop(key, None)
            else:
                terms[key] = (prev_axis, new_coeff)
        return AffineTime(base=self.base + other.base, _terms=terms)

    def __sub__(self, other: "AffineTime") -> "AffineTime":
        return self + other.scaled(-1)

    # -- queries ------------------------------------------------------------

    @property
    def is_constant(self) -> bool:
        return not self._terms

    def terms(self) -> list[tuple[object, int]]:
        """(axis, coeff) pairs, identity-stable."""
        return [(axis, coeff) for (axis, coeff) in self._terms.values()]

    def is_grid_aligned(self, grid: int) -> bool:
        """True iff value % grid == 0 for every counter value.
        
        Checks base (accounting for sweep offsets) and steps seperately.
        """
        if grid <= 1:
            return True
        if self.base % grid != 0:
            return False
        return all(coeff % grid == 0 for _, coeff in self.terms())

    def min_value(self) -> int:
        """Minimum over all iterations, counters ranging [0, num_steps).

        num_steps: callable axis -> int, consulted ONLY for counters with a
        negative coefficient (a decreasing sweep). Increasing sweeps take
        their minimum at counter=0, so num_steps is never called for them.
        """
        m = self.base
        for axis, coeff in self.terms():
            if coeff < 0:
                m += coeff * (axis.num_steps - 1)
        return m

    def max_value(self, num_steps) -> int:
        m = self.base
        for axis, coeff in self.terms():
            if coeff > 0:
                m += coeff * (num_steps(axis) - 1)
        return m

    def evaluate(self, counter_values: dict) -> int:
        """Concrete ftsamp for a given assignment of counters (by identity)."""
        total = self.base
        for axis, coeff in self.terms():
            total += coeff * counter_values[axis]
        return total

    def __repr__(self) -> str:
        parts = [str(self.base)]
        for axis, coeff in self.terms():
            parts.append(f"{coeff:+d}*{getattr(axis, 'name', axis)}")
        return "AffineTime(" + " ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Timed instruction
# ---------------------------------------------------------------------------

@dataclass
class TimedInstruction:
    instruction: PlayIR
    t_start: AffineTime
    t_end: AffineTime
    segment_id: int          # straight-line run id; fusion only within one id

    def __repr__(self) -> str:
        name = getattr(self.instruction.pulse, "name", "?")
        return f"Timed(seg{self.segment_id}: {name} @ {self.t_start} .. {self.t_end})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _advance_by_pulse(start: AffineTime, pulse) -> AffineTime:
    """End time of a pulse played at `start`, affine in any length counter.

    * waveform mode          -- fixed shape length (constant).
    * coarse mode, constant  -- length_coarse.value treg (constant).
    * coarse mode, swept     -- the tProc counts the pulse out in treg, so a
      LinearSweepAxis length is affine exactly like a swept delay:
          end(i) = start + (axis.start + i*axis.step) * FTSAMP_PER_TREG
      Both terms are xFTSAMP_PER_TREG, so treg-grid alignment is preserved
      for every iteration. Exponential / pattern / repeat-driven lengths are
      rejected (non-affine or non-durations), mirroring _advance_by_delay.
    """
    if pulse.waveform_mode:
        return start.shifted(pulse.shape.length)

    lp = pulse.length_coarse                      # coarse_time Parameter (treg)
    if lp.is_constant:
        return start.shifted(lp.value * FTSAMP_PER_TREG)

    axis = lp.driving_axis

    if isinstance(axis, LinearSweepAxis):
        return (start.shifted(axis.start * FTSAMP_PER_TREG)
                     .plus_counter(axis, axis.step * FTSAMP_PER_TREG))

    if isinstance(axis, ExponentialSweepAxis):
        raise ValueError(
            f"Pulse '{pulse.name}' length driven by exponential axis "
            f"'{axis.name}' produces geometric (non-affine) timing. "
            f"Not supported without rollout."
        )

    raise ValueError(
        f"Pulse '{pulse.name}' length must be a constant or a "
        f"LinearSweepAxis (kind='coarse_time'); got {type(axis).__name__}."
    )


def _advance_by_delay(cursor: AffineTime, duration: Parameter) -> AffineTime:
    """Advance a per-channel cursor by a DelayIR duration Parameter."""
    if duration is None:
        raise ValueError("DelayIR has no duration parameter.")

    if duration.is_constant:
        return cursor.shifted(duration._constant_value)

    axis = duration.driving_axis

    if isinstance(axis, LinearSweepAxis):
        # value(i) = start + i*step, both canonical ftsamp ints
        return cursor.shifted(axis.start).plus_counter(axis, axis.step)

    if isinstance(axis, ExponentialSweepAxis):
        raise ValueError(
            f"Delay driven by exponential axis '{axis.name}' produces "
            f"geometric (non-affine) timing. Not supported without rollout."
        )

    if isinstance(axis, RepeatAxis):
        raise ValueError(
            f"Delay duration is bound to RepeatAxis '{axis.name}', which is a "
            f"loop count, not a physical duration."
        )

    raise ValueError(
        "Delay duration must be a constant or a LinearSweepAxis; "
        f"got {type(axis).__name__}."
    )


# ---------------------------------------------------------------------------
# Recursive scheduler
# ---------------------------------------------------------------------------

def _schedule(nodes: list, seg_alloc) -> tuple[list[tuple[int, TimedInstruction]],
                                               dict[int, AffineTime]]:
    """Schedule a straight-line list of IR nodes.

    seg_alloc : itertools.count shared across the whole walk; a fresh id is
                taken at entry and after every block, so each uninterrupted
                straight-line run has its own segment id.

    Returns (events, duration):
        events   : list of (channel, TimedInstruction), times RELATIVE to
                   this segment's start (== const 0).
        duration : per-channel end cursor, relative to segment start.
    """
    events: list[tuple[int, TimedInstruction]] = []
    cursor: dict[int, AffineTime] = {}
    seg = next(seg_alloc)

    def cur(ch: int) -> AffineTime:
        return cursor.get(ch, AffineTime.const(0))

    for node in nodes:

        # --- leaf: play ----------------------------------------------------
        if isinstance(node, PlayIR):
            ch = node.channel
            start = cur(ch)
            end = _advance_by_pulse(start, node.pulse)

            grid = node.pulse.preferred_resolution
            grid = 1 if grid == "auto" else grid
            if not start.is_grid_aligned(grid):
                raise ValueError(
                    f"Pulse '{node.pulse.name}' start {start} is not aligned "
                    f"to its {grid}-ftsamp grid for all iterations "
                    f"(base and every swept step must divide by {grid})."
                )

            events.append((ch, TimedInstruction(node, start, end, seg)))
            cursor[ch] = end

        # --- leaf: delay ---------------------------------------------------
        elif isinstance(node, DelayIR):
            ch = node.channel
            cursor[ch] = _advance_by_delay(cur(ch), node.duration)

        # --- block: repeat (sequential, accumulates) -----------------------
        elif isinstance(node, RepeatBlock):
            body_events, body_dur = _schedule(node.body, seg_alloc)
            counter = node.axis
            bound = node.axis.bound

            for ch, dur in body_dur.items():
                if not dur.is_constant:
                    raise ValueError(
                        f"repeat('{node.axis.name}') body has swept timing on "
                        f"channel {ch} ({dur}); k*(varying duration) is "
                        f"non-affine. Move the swept delay outside the repeat."
                    )

            for ch, timed in body_events:
                entry = cur(ch)
                coeff = body_dur.get(ch, AffineTime.const(0)).base
                st = (entry + timed.t_start).plus_counter(counter, coeff)
                en = (entry + timed.t_end).plus_counter(counter, coeff)
                events.append((ch, TimedInstruction(timed.instruction, st, en,
                                                    timed.segment_id)))

            for ch, dur in body_dur.items():
                entry = cur(ch)
                d = dur.base
                if isinstance(bound, int):
                    cursor[ch] = entry.shifted(bound * d)
                elif isinstance(bound, LinearSweepAxis):
                    # bound's actual repeat count at iteration k is
                    # (bound.start + k*bound.step) -- same convention as
                    # _advance_by_delay's LinearSweepAxis branch. The cursor
                    # advances d per repeat, so the affine term needs BOTH a
                    # base offset (d*start) and a scaled coefficient
                    # (d*step), not just a bare coefficient of d: a bare
                    # plus_counter(bound, d) silently drops the d*start term
                    # entirely (only invisible when start==0), and is only
                    # coincidentally right on the coefficient when step==1.
                    cursor[ch] = (entry.shifted(d * bound.start)
                                       .plus_counter(bound, d * bound.step))
                else:
                    raise ValueError(
                        f"repeat('{node.axis.name}') bound '{bound.name}' is "
                        f"a {type(bound).__name__}; only a fixed int or a "
                        f"LinearSweepAxis produces an affine repeat-count "
                        f"cursor advance (geometric bounds are non-affine)."
                    )

            seg = next(seg_alloc)   # continuation after the block is a new run

        # --- block: sweep (independent shots, resets) ----------------------
        elif isinstance(node, SweepBlock):
            body_events, body_dur = _schedule(node.body, seg_alloc)

            # swept-delay counter terms are already baked into body_events;
            # sweep adds NO k*body_dur accumulation.
            for ch, timed in body_events:
                entry = cur(ch)
                events.append((ch, TimedInstruction(timed.instruction,
                                                    entry + timed.t_start,
                                                    entry + timed.t_end,
                                                    timed.segment_id)))

            # independent-shot model: cursor resets after the sweep.
            for ch in body_dur:
                cursor[ch] = AffineTime.const(0)

            seg = next(seg_alloc)

        # --- trigger / other -----------------------------------------------
        elif isinstance(node, TriggerIR):
            continue   # TODO: model trigger timing once it gains a channel/time

        else:
            raise TypeError(f"timing_pass: unexpected IR node {type(node).__name__}")

    return events, cursor


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def timing_pass(sequence: Sequence) -> dict[int, list[TimedInstruction]]:
    """Lower a Sequence into per-channel timed instructions with affine times.

    Program order is preserved per channel (NOT sorted by base time -- sweep
    cursor resets make base times non-monotonic across segments).
    """
    events, _ = _schedule(sequence.instructions, count())

    per_channel: dict[int, list[TimedInstruction]] = {}
    for ch, timed in events:
        per_channel.setdefault(ch, []).append(timed)
    return per_channel