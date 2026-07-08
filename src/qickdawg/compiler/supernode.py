"""
Supernode + waveform construction passes for QICK-DAWG.

Consumes the affine timing IR (timing.TimedInstruction, per channel) and
produces reusable baseband SuperNodes plus per-channel programs of
SuperNodeInstance / ConstantPulseInstance, then lays out waveform memory.

Design decisions baked in (confirmed)
-------------------------------------
1. Grouping stability. Two waveform pulses fuse into one SuperNode only when
   they share a segment id, share a frequency, and the gap between them is a
   *constant* under the 2-treg deadtime. A swept gap can never be an internal
   supernode gap: its minimum over all iterations is required to be >= 2 treg
   (else a hard timing error), so a swept delay is always a boundary.

2. Uniform sweeps only. If a value is swept inside a multi-block supernode it
   must be swept *identically* for every block (same axis object), so it lives
   purely on the instance as a global phase/amp/freq and leaves the baseband
   shape iteration-invariant. Sweeping only one block of a fused pair is
   refused for now (the pulsepol-style case -> write it as separate pulses).

3. Timing violations are reported explicitly rather than silently repaired.

Not yet supported (rejected, noted for future revision)
-------------------------------------------------------
  * A phase Pattern inside a multi-block supernode (would need >1 supernode /
    rollout to enumerate the per-index shapes). A Pattern on a single-pulse
    supernode is fine -- it is just that instance's global phase.

Dependencies
------------
  * pulses.DefinePulse / Shape (waveform_mode, shape, length_treg,
    preferred_resolution, and Shape.content_hash / .data / .length).
  * Converting a canonical phase register value back to degrees uses the
    hardware inverse qd.soccfg.reg2deg (see _preg_to_deg). Only the
    constant-composite case (e.g. pulsepol 0/90/90/0) exercises it; the
    uniform-swept case has relative phase 0 and never calls it.
"""

# TODO: Deal with phase, inconvenient to convert back and forth
# Probably fine to have bith a converted a regular form??

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qickdawg as qd

from timing import AffineTime, TimedInstruction
from units import FTSAMP_PER_TREG, SAMPLE_SIZE, WAVEFORM_MEMORY_SIZE

SAMPLE_AMP = 2 ** (SAMPLE_SIZE - 1) - 1
SPLIT_GAP = 2 * FTSAMP_PER_TREG      # waveform pulses closer than this must fuse


# ---------------------------------------------------------------------------
# Parameter comparison / conversion helpers
# ---------------------------------------------------------------------------

def _same_param(p, q) -> bool:
    """Two Parameters drive the same register value across all iterations.

    Constant: equal canonical value. Swept: the *same* axis/pattern object
    (identity), which is what the uniform-sweep rule requires.
    """
    if p is None or q is None:
        return p is q
    if p.is_constant and q.is_constant:
        return p._constant_value == q._constant_value
    if (not p.is_constant) and (not q.is_constant):
        return p.axis is q.axis
    return False


def _is_pattern_backed(p) -> bool: # TODO: Review
    # Parameter.from_pattern stores the Pattern object directly in .axis
    return (p is not None) and (not p.is_constant) and (p.driving_axis is not p.axis)


def _preg_to_deg(preg: int) -> float:
    """Canonical phase register -> degrees, via the hardware inverse."""
    reg2deg = getattr(qd.soccfg, "reg2deg", None)
    if reg2deg is None:
        raise NotImplementedError(
            "Folding a constant per-block phase offset into a baseband shape "
            "needs a preg->deg inverse. Expose qd.soccfg.reg2deg or add the "
            "phase-register period to units.py."
        )
    return float(reg2deg(int(preg)))


# ---------------------------------------------------------------------------
# SuperNode data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperNodeBlock:
    """One pulse's contribution to a baseband supershape.

    rel_phase is in DEGREES and rel_amplitude is a ratio, both iteration-
    invariant, so they can key the reusable shape.
    """
    shape: object
    rel_start: int          # ftsamp, relative to supernode start
    rel_phase: float        # deg
    rel_amplitude: float

    @property
    def key(self):
        return (self.shape.content_hash, self.rel_start,
                round(self.rel_phase, 9), round(self.rel_amplitude, 9))


@dataclass(frozen=True)
class SuperNode:
    """Unique reusable baseband supershape definition."""
    blocks: tuple[SuperNodeBlock, ...]
    duration_ftsamp: int
    preferred_resolution: int

    @property
    def key(self):
        return tuple(block.key for block in self.blocks)

    @property
    def equivalent_shape(self) -> np.ndarray:
        array = np.zeros(self.duration_ftsamp, dtype=complex)
        for block in self.blocks:
            start = block.rel_start
            end = start + block.shape.length
            array[start:end] += (
                block.shape.data
                * block.rel_amplitude
                * np.exp(1j * np.deg2rad(block.rel_phase))
            )
        return array


@dataclass(frozen=True)
class SuperNodeInstance:
    """A concrete scheduled use of a SuperNode.

    All iteration-varying quantities live here as handles, never in the shape.
    """
    supernode_key: tuple
    channel: int
    t_start: AffineTime            # affine (carries any enclosing repeat offset)
    frequency: object              # Parameter
    global_phase: object           # Parameter (may be swept or Pattern-backed)
    global_amplitude: object       # Parameter


@dataclass(frozen=True)
class ConstantPulseInstance:
    """A concrete scheduled use of a constant (non-waveform) pulse."""
    channel: int
    t_start: AffineTime
    length_treg: int
    frequency: object              # Parameter
    phase: object                  # Parameter
    amplitude: object              # Parameter


# ---------------------------------------------------------------------------
# Uniformity analysis (decision #2)
# ---------------------------------------------------------------------------

def _analyze_phase(phases, multiblock: bool):
    """Return (base_phase_param, rel_phases_deg) enforcing the uniform rule."""
    if any(not p.is_constant for p in phases):
        ref = phases[0]
        for p in phases:
            if not _same_param(p, ref):
                raise ValueError(
                    "A supernode has a phase swept on only some of its pulses. "
                    "Swept values must apply uniformly to the whole supernode; "
                    "write differing per-pulse phases as separate pulses."
                )
        if multiblock and _is_pattern_backed(ref):
            raise ValueError(
                "A phase Pattern inside a multi-pulse supernode is not yet "
                "supported (would need multiple supernodes). Split the pulses."
            )
        return ref, [0.0] * len(phases)

    # all constant -> relative phases fold into the shape, in degrees
    base = phases[0]
    base_deg = _preg_to_deg(base._constant_value)
    rel = [(_preg_to_deg(p._constant_value) - base_deg) % 360 for p in phases]
    return base, rel


def _analyze_amplitude(amps):
    """Return (base_amp_param, rel_amplitudes) enforcing the uniform rule."""
    from parameters import Parameter

    if any(not a.is_constant for a in amps):
        ref = amps[0]
        for a in amps:
            if not _same_param(a, ref):
                raise ValueError(
                    "A supernode has amplitude swept on only some of its pulses. "
                    "Swept values must apply uniformly to the whole supernode."
                )
        return ref, [1.0] * len(amps)

    # all constant -> normalize the shape by the max amplitude (rel in (0, 1])
    values = [a._constant_value for a in amps]
    base_val = max(values)
    if base_val <= 0:
        raise ValueError("Supernode has non-positive maximum amplitude.")
    base = Parameter.constant(name="_supernode_amp", value=base_val, unit="amp")
    rel = [v / base_val for v in values]
    return base, rel


# ---------------------------------------------------------------------------
# Build one SuperNode from a fused group
# ---------------------------------------------------------------------------

def build_supernode(group: list[TimedInstruction]):
    """Return (SuperNode, base_phase, base_amp, frequency, t_start)."""
    if not group:
        raise ValueError("Cannot build a SuperNode from an empty group.")

    t0 = group[0].t_start
    multiblock = len(group) > 1

    freqs = [g.instruction.frequency for g in group]
    phases = [g.instruction.phase for g in group]
    amps = [g.instruction.amplitude for g in group]

    # single NCO -> one frequency for the whole supernode
    frequency = freqs[0]
    for f in freqs[1:]:
        if not _same_param(f, frequency):
            raise ValueError(
                "Fused supernode pulses have different frequencies; a supernode "
                "shares one NCO frequency. Separate them with >= 2 treg."
            )

    base_phase, rel_phases = _analyze_phase(phases, multiblock)
    base_amp, rel_amps = _analyze_amplitude(amps)

    blocks = []
    for g, rphase, ramp in zip(group, rel_phases, rel_amps):
        rel_start = g.t_start - t0
        if not rel_start.is_constant:
            raise ValueError(
                "Internal supernode timing must be constant across iterations."
            )
        if not g.instruction.pulse.waveform_mode:
            raise ValueError(
                f"Non-waveform pulse '{g.instruction.pulse.name}' cannot be in "
                f"a SuperNode."
            )
        blocks.append(SuperNodeBlock(
            shape=g.instruction.pulse.shape,
            rel_start=rel_start.base,
            rel_phase=rphase,
            rel_amplitude=ramp,
        ))

    duration_ftsamp = max((g.t_end - t0).base for g in group)

    def _res(r):
        return 1 if r == "auto" else r
    preferred_resolution = min(_res(g.instruction.pulse.preferred_resolution)
                               for g in group)

    node = SuperNode(blocks=tuple(blocks),
                     duration_ftsamp=duration_ftsamp,
                     preferred_resolution=preferred_resolution)
    return node, base_phase, base_amp, frequency, t0


# ---------------------------------------------------------------------------
# Supernode discovery pass
# ---------------------------------------------------------------------------

def find_supernodes(timed: dict[int, list[TimedInstruction]]):
    """Lower per-channel timed pulses into supernodes + channel programs.

    Returns (unique_supernodes, channel_programs).
    """
    unique_supernodes: dict[tuple, SuperNode] = {}
    channel_programs: dict[int, list] = {}

    for ch, instructions in timed.items():
        program: list = []
        group: list[TimedInstruction] = []

        def flush():
            if not group:
                return
            node, base_phase, base_amp, freq, t0 = build_supernode(group)
            key = node.key
            existing = unique_supernodes.get(key)
            if existing is None or node.preferred_resolution < existing.preferred_resolution:
                unique_supernodes[key] = node
            program.append(SuperNodeInstance(
                supernode_key=key,
                channel=ch,
                t_start=t0,
                frequency=freq,
                global_phase=base_phase,
                global_amplitude=base_amp,
            ))
            group.clear()

        for timed_inst in instructions:
            inst = timed_inst.instruction

            # constant (non-waveform) pulses break any run and are emitted raw
            if not inst.pulse.waveform_mode:
                flush()
                program.append(ConstantPulseInstance(
                    channel=ch,
                    t_start=timed_inst.t_start,
                    length_treg=inst.pulse.length_treg,
                    frequency=inst.frequency,
                    phase=inst.phase,
                    amplitude=inst.amplitude,
                ))
                continue

            # waveform pulse: decide whether it continues the current group
            if group:
                prev = group[-1]
                if prev.segment_id != timed_inst.segment_id:
                    flush()                               # block boundary
                else:
                    gap = timed_inst.t_start - prev.t_end
                    if gap.is_constant:
                        if gap.base >= SPLIT_GAP:
                            flush()                       # far enough -> split
                        elif not _same_param(inst.frequency,
                                             prev.instruction.frequency):
                            raise ValueError(
                                f"Channel {ch}: pulses '{prev.instruction.pulse.name}' "
                                f"and '{inst.pulse.name}' sit {gap.base} ftsamp apart "
                                f"(< {SPLIT_GAP}) so must fuse into one waveform, but "
                                f"they have different frequencies."
                            )
                        # else: small same-freq gap -> keep grouping
                    else:
                        gmin = gap.min_value()
                        if gmin < SPLIT_GAP:
                            raise ValueError(
                                f"Channel {ch}: swept delay between "
                                f"'{prev.instruction.pulse.name}' and "
                                f"'{inst.pulse.name}' can shrink the gap to {gmin} "
                                f"ftsamp, under the {SPLIT_GAP}-ftsamp deadtime. "
                                f"Enforce a minimum swept delay of >= {SPLIT_GAP} ftsamp."
                            )
                        flush()                           # swept gap -> boundary

            group.append(timed_inst)

        flush()
        channel_programs[ch] = program

    return unique_supernodes, channel_programs


# ---------------------------------------------------------------------------
# Waveform construction
# ---------------------------------------------------------------------------

def compute_waveform_memory(supernodes: dict[tuple, SuperNode]) -> dict[tuple, int]:
    """Samples of waveform memory needed per supernode (resolution-scaled).

    NOTE: the sample-count formula is carried over verbatim from the original
    pulseconstructor and is worth re-deriving before trusting it.
    """
    waveform_memory: dict[tuple, int] = {}
    expended = 0
    for key, node in supernodes.items():
        num_samples = (np.ceil(node.duration_ftsamp / FTSAMP_PER_TREG + 1)
                       * FTSAMP_PER_TREG * node.preferred_resolution // FTSAMP_PER_TREG)
        expended += num_samples
        waveform_memory[key] = int(num_samples)

    if expended > WAVEFORM_MEMORY_SIZE:
        raise ValueError(
            f"Total waveform memory required ({expended} samples) exceeds "
            f"available memory ({WAVEFORM_MEMORY_SIZE} samples)."
        )
    return waveform_memory


@dataclass(frozen=True)
class WaveformBase:
    """IQ representation of the basic waveform tile."""
    i_data: np.ndarray
    q_data: np.ndarray
    phase_adjustment: float
    scale_factor: float


def create_waveform_base(supernode: SuperNode) -> WaveformBase:
    """Build the IQ tile for a supernode, choosing a discrete rotation that
    maximizes headroom (fewer NCO phase adjustments downstream)."""
    shape_arr = supernode.equivalent_shape
    filtered = shape_arr[np.abs(shape_arr) > 1 / np.sqrt(2)]

    max_factor = 1.0
    rot_angle = 45.0
    for angle in np.arange(0, 90, 90 / 16):
        rotated = filtered * np.exp(-1j * np.deg2rad(angle))
        max_real = np.max(np.abs(np.real(rotated))) if rotated.size else 1.0
        max_imag = np.max(np.abs(np.imag(rotated))) if rotated.size else 1.0
        factor = 1 / max(max_real, max_imag)
        if factor > max_factor + 1e-4:
            max_factor = factor
            rot_angle = angle

    optimized = shape_arr * np.exp(-1j * np.deg2rad(rot_angle)) * SAMPLE_AMP
    idata = np.real(optimized).astype(np.int16)   # was np.real(x, np.int16) -> error
    qdata = np.imag(optimized).astype(np.int16)

    return WaveformBase(i_data=idata, q_data=qdata,
                        phase_adjustment=rot_angle, scale_factor=max_factor)


def waveform_construction_pass(supernodes: dict[tuple, SuperNode],
                               channel_programs: dict[int, list]):
    waveform_memory = compute_waveform_memory(supernodes)
    waveform_bases = {key: create_waveform_base(node)
                      for key, node in supernodes.items()}
    # waveform_layout(...) -- memory placement optimization, still TODO
    return waveform_memory, waveform_bases