# from tokenize import group
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Literal, Callable
from typing import Optional

from ..parameters import Parameter

# TODO: Tie error values for different issues, (sweep issue, timing issue, amplitude issue, wvfm memory issue, inst memory issue)
        
# ---------------------------------------------------------------------------
# Pulse Definition
# ---------------------------------------------------------------------------

DAC_SAMPLE_RATE = 4915.2e6  # Hz
FTSAMP_PER_TREG = 16
WAVEFORM_MEMORY_SIZE = 2**16  # samples
SAMPLE_SIZE = 16  # bits
SAMPLE_AMP = 2**(SAMPLE_SIZE - 1) - 1  # max amplitude for I and Q channels

@dataclass(frozen=True)
class Shape:
    """
    User facing class to define the normalized shape of a pulse.

    Amplitude is set elsewhere. Shape values should satisfy:
        abs(shape) <= 1

    Users can use predefined shapes, or define their own with an array or a function.
    Complex shapes are supported and automatically inferred from complex array values or functions.

    Time units are in DAC samples (ftsamp).

    TODO: Standardize this to be able to use parameter arguments now.
    """ 
    data: np.ndarray
    params: dict | None = None # for debugging and tracing
    content_hash: int = field(init=False)

    def __post_init__(self):
        arr = np.asarray(self.data)
        Shape._validate_array(arr)

        peak = np.max(np.abs(arr))
        arr = arr / peak
        arr = np.ascontiguousarray(arr)

        h = hashlib.blake2b(digest_size=16)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.view(np.uint8))

        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "content_hash", h.hexdigest())

    @property
    def length(self) -> int:
        return len(self.data)
    
    @property     # FIXME: Remove if not needed
    def is_complex(self) -> bool:
        if self.data is None:
            return False
        return np.any(np.imag(self.data) != 0)

    @staticmethod
    def square(length: int):
        Shape._validate_length(length)
        data = np.ones(length, dtype=float)
        return Shape(data=data, params={"source": "square"})
    
    @staticmethod
    def hermite(length: int, eta: float): # TODO: CHECK MATH
        """
        length: length of pulse in ftsamp        
        """
        Shape._validate_length(length)
        t = np.arange(length, dtype=float)
        u = ((t - length/2) / 0.1667 * length) ** 2
        f = (1 - eta * u) * np.exp(-u)

        arr = np.array(f, dtype=float)
        return Shape(data=arr, params={"hermite": "square", "eta": eta})
    
    @staticmethod
    def custom(data: np.ndarray):
        return Shape(data=np.asarray(data), params={"source": "custom_array"})
    
    @staticmethod
    def from_func(func: Callable, length: int, **params):
        Shape._validate_length(length)
        t = np.arange(0, length)
        arr = np.asarray(func(t, **params))
        return Shape(data=arr, params={"source": "custom_func", **params})
    
    @staticmethod
    def _validate_length(length: int):
        if not isinstance(length, int):
            raise TypeError("Shape length must be an integer number of DAC samples.")
        if length <= 0:
            raise ValueError("Shape length must be positive.")

    @staticmethod
    def _validate_array(arr: np.ndarray):
        if arr.ndim != 1:
            raise ValueError("Custom shape data must be a 1D real or complex array.")
        
        if len(arr) < FTSAMP_PER_TREG + 2: # This should always result in a valid waveform length of 3 treg or more (based on the other constraints)
            raise ValueError(f"Custom shape data must be at least ({FTSAMP_PER_TREG + 2} DAC samples).") 

        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("Shape data must contain numeric real or complex values.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Shape data must not contain NaN or infinite values.")


@dataclass(frozen=True)
class DefinePulse:
    """ User-facing default pulse definition.

    NOTE:
        Any waveform can be encoded with up to 1 amplitude, but select waveforms can have higher amplitude cielings 
        (sqrt(2) if using 90 degree phases, or 2 for constant pulses).

    TODO:
        Add user feedback that defines the maximum amplitude a pulse can have.
    
    """
    name: str
    amplitude: float    # [0, 2], where 1 is the universal cieling for arbitrary waveforms, but for select waveforms it can go up to sqrt(2), constant allows for 2
    phase: float        # deg
    freq: float         # Hz
    channel: int

    shape: Shape | None = None      # Only set when waveform_mode = True
    length_treg: int | None = None  # This should only be set if waveform_mode = False
    preferred_resolution: Literal['auto', 1, 2, 4, 8, 16] = 'auto' # TODO: update based on 2^n

    @property
    def waveform_mode(self) -> bool:
        return self.shape is not None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Pulse name must be a non-empty string.")

        if not (0 <= self.amplitude <= 1):
            raise ValueError("Pulse amplitude must satisfy 0 <= amp <= 1.")
        
        if self.freq < 0:
            raise ValueError("Pulse frequency must be non-negative.")

        if self.shape is None and self.length_treg is None:
            raise ValueError("Cannot have length_Treg and shape."
                             "Must specify either a Shape (arbitrary waveform mode)"
                             "or length_treg (constant mode).")
        
        if self.length_treg is not None and self.length_treg < 3:
            raise ValueError(f"{self.name} length_treg must be at least 3 treg {3 * FTSAMP_PER_TREG} DAC samples).")
        
        if self.shape is None:
            self.preferred_resolution = 16

    @property
    def length_ftsamp(self) -> int:
        if self.waveform_mode:
            return self.shape.length
        else:
            return self.length_treg * FTSAMP_PER_TREG
    
    @property
    def length_sec(self) -> float:
        # Helper function to check pulse length
        return self.length_ftsamp / DAC_SAMPLE_RATE

# ---------------------------------------------------------------------------
# High level IR types
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sequence context manager
# ---------------------------------------------------------------------------

# TODO: Add a repeat structure? To work with something like:
# phase = parameter.pattern([0, 90, 0, 90, 90, 0, 90, 0])
# repeat(N):
#   play(pulse, phase=phase)

_active_sequence = None

class Sequence:
    def __init__(self):
        self.instructions = []

    def __enter__(self):
        global _active_sequence

        if _active_sequence is not None:
            raise RuntimeError("Nested Sequence contexts are not supported.")

        _active_sequence = self
        return self

    def __exit__(self, *args):
        global _active_sequence
        _active_sequence = None

    def append(self, inst):
        self.instructions.append(inst)
        return inst
    
    def __repr__(self):
        lines = [f"Sequence ({len(self.instructions)} instructions)"]
        for inst in self.instructions:
            lines.append(f"  {inst}")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# User functions
# ---------------------------------------------------------------------------

# NOTE: Assume Hz, s, and deg when not defined as a parameter?


def _record(inst):
    if _active_sequence is not None:
        _active_sequence.instructions.append(inst)
    return inst

def play(pulse: DefinePulse, *,
        start_time: int | None = None,
        phase: float | None = None,
        amplitude: float | None = None,
        freq: float | None = None,
        channel: int | None = None,
    ):

    return _record(PlayIR(
                pulse=pulse,
                start_time=start_time,
                phase=phase,
                amplitude=amplitude,
                freq=freq,
                channel=channel,
        )
    )

def delay(duration: int, channel: int | None = None):
    return _record(DelayIR(duration=duration, channel=channel))


# ---------------------------------------------------------------------------
# Initial timing pass
# ---------------------------------------------------------------------------

@dataclass
class TimedInstruction:
    instruction: PlayIR
    t_start:     int    # absolute start time in ftsamp
    t_end:       int    # absolute end time in ftsamp

    # TODO: Add universal syncing (userfacing, and then have the impacts propogated through the original timing graph)

def required_gap(prev_pulse: DefinePulse | None, next_pulse: DefinePulse) -> int:
    if prev_pulse is None:
        return 0
    # waveform -> constant transition requires 2 treg deadtime
    if prev_pulse.waveform_mode and not next_pulse.waveform_mode:
        return 2 * FTSAMP_PER_TREG
    return 0

def pulse_length_ftsamp(pulse: DefinePulse) -> int:
    if pulse.waveform_mode:
        return pulse.shape.length
    return pulse.length_treg * FTSAMP_PER_TREG


def timing_pass(sequence: Sequence):
    """Timing should be handled in ftsamp.

    This pass does not repair timing. It only lowers user scheduling choices
    and errors when they are not representable or hardware-safe.
    """
    channels = {}

    for inst in sequence.instructions:
        ch = inst.channel
        channels.setdefault(ch, []).append(inst)

    timed = {}

    for ch, instructions in channels.items():
        # Cursor: scheduling anchor.
        # last_pulse_end: end of previous scheduled pulse or committed delay.
        cursor = 0
        last_pulse_end = 0
        last_pulse = None
        timed[ch] = []

        for inst in instructions:
            if isinstance(inst, PlayIR):
                if inst.treg_offset is None:
                    pulse_start = last_pulse_end
                else:
                    pulse_start = cursor + inst.treg_offset * FTSAMP_PER_TREG

                if last_pulse is not None:
                    min_gap = required_gap(last_pulse.instruction.pulse, inst.pulse)
                    actual_gap = pulse_start - last_pulse.t_end

                    if actual_gap < min_gap:
                        raise ValueError(
                            f"Pulse {inst.pulse.name} cannot be scheduled at "
                            f"{pulse_start} ftsamp. The previous pulse ended at "
                            f"{last_pulse_end} ftsamp, giving only {actual_gap} "
                            f"ftsamp of separation, but this transition requires "
                            f"at least {min_gap} ftsamp. Insert a delay or schedule "
                            f"the pulse later."
                        )

                grid = inst.pulse.preferred_resolution if inst.pulse.preferred_resolution != 'auto' else 1

                if pulse_start % grid != 0:
                    raise ValueError(
                        f"Pulse {inst.pulse.name} cannot be scheduled at "
                        f"{pulse_start} ftsamp. It must start on a "
                        f"{grid}-ftsamp grid."
                    )

                pulse_end = pulse_start + pulse_length_ftsamp(inst.pulse)
                last_pulse = TimedInstruction(instruction=inst, t_start=pulse_start, t_end=pulse_end)
                
                timed[ch].append(last_pulse)

                last_pulse_end = pulse_end

            elif isinstance(inst, DelayIR):
                cursor = max(last_pulse_end, cursor) + inst.duration
                last_pulse_end = cursor

    return timed


# ---------------------------------------------------------------------------
# Supernode IR
# ---------------------------------------------------------------------------

# NOTE: Fundamental block is a pulse Shape made up of other shapes, with amplitude and phase scaling
# already applied.

"""
SuperNode:
    frequency-independent reusable baseband supershape structure (shape)

SuperNode:
    frequency-independent reusable baseband supershape structure (instance)
"""
# FIXME: Need to add a way to track constant pulses in this pass (this can be done outside of the supernode (Seperate IR))
@dataclass(frozen=True)
class SuperNodeBlock:
    """
    Represents the baseband shape, of a single pulse
    """
    shape: Shape
    rel_start: int
    rel_phase: float
    rel_amplitude: float

    @property
    def key(self):
        return (
            self.shape.content_hash,
            self.rel_start,
            self.rel_phase,
            self.rel_amplitude,
        )
    
# FIXME: Use these for defining keys, if we find that our current approach yields unique hashes for same waveforms
def _canonical_phase_deg(phase: float, ndigits: int = 12) -> float:
    return round(float(phase) % 360, ndigits)

def _canonical_ratio(value: float, ndigits: int = 12) -> float:
    return round(float(value), ndigits)

def _canonical_frequency(freq: float, ndigits: int = 6) -> float:
    return round(float(freq), ndigits)


@dataclass(frozen=True)
class SuperNode:
    """
    Unique reusable baseband supershape definition.
    """
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
                block.shape.data * block.rel_amplitude * np.exp(1j * np.deg2rad(block.rel_phase))
            )

        return array



@dataclass(frozen=True) # should i track key rather than supernode, i can always pick out the key from the supernode if not
class SuperNodeInstance:
    """
    A concrete scheduled use of a SuperNode.
    """
    supernode_key: tuple
    channel: int
    t_start: int            # absolute ftsamp
    frequency: float        # Hz
    global_phase: float     # deg
    global_amplitude: float


@dataclass(frozen=True)
class ConstantPulseInstance:
    """
    A concrete scheduled use of a constant pulse.

    This is not a SuperNode because it has no reusable waveform shape.
    """
    channel: int
    t_start: int
    length_treg: int
    frequency: float
    phase: float
    amplitude: float


# NOTE: its possible that a user opts to use 0 amplitude pulses as no-ops
# TODO: We can either make them a special case, where they are allowed if we find a supernode
# in a next pass with matching duration and assign a global amplitude of 0 for instances. 
# Or we could remove them, and force the translation pass to increase the delays to account for their removal.
# Each approach has it's own merits.
def make_supernode(group: list[TimedInstruction]) -> SuperNode:
    if not group:
        raise ValueError("Cannot build SuperNode from empty group.")

    t0 = group[0].t_start
    base_phase = group[0].instruction.phase
    base_amp = max(timed.instruction.amplitude for timed in group)

    blocks = []

    for timed in group:
        inst = timed.instruction

        if not inst.pulse.waveform_mode: # filtered out in the timing pass
            raise ValueError(
                f"Cannot include non-waveform pulse {inst.pulse.name} in a SuperNode."
            )

        blocks.append(
            SuperNodeBlock(
                shape=inst.pulse.shape,
                rel_start=timed.t_start - t0,
                rel_phase=(inst.phase - base_phase) % 360,
                rel_amplitude=inst.amplitude / base_amp,
            )
        )

    duration_ftsamp = max(p.t_end for p in group) - t0

    # NOTE: Change depending on how we want to handle 'auto'
    def _resolution_value(res):
        return 1 if res == "auto" else res
    
    preferred_resolution = min(
        _resolution_value(timed.instruction.pulse.preferred_resolution)
        for timed in group
    )

    return SuperNode(
        blocks=tuple(blocks),
        duration_ftsamp=duration_ftsamp,
        preferred_resolution=preferred_resolution
    )


def find_supernodes(timed: dict[int, list[TimedInstruction]]) -> tuple[dict[tuple, SuperNode], 
                                                                 dict[int, list[SuperNodeInstance | ConstantPulseInstance]]]:
    """
    Identify unique reusable waveform SuperNodes and lower each channel's
    timed pulse list into SuperNodeInstance objects.

    Returns:
        unique_supernodes:
            dict of resolvable waveform nodes keyed by SuperNode.key

        supernode_shapes:
            dict of shape arrays keyed by SuperNode.key

        channel_programs:
            dict[channel, list[SuperNodeInstance | ConstantPulseInstance]]
    """
    unique_supernodes: dict[tuple, SuperNode] = {}
    channel_programs: dict[int, list[SuperNodeInstance | ConstantPulseInstance]] = {}

    split_gap = 2 * FTSAMP_PER_TREG

    # Parse instructions, if they are waveform pulses with less than 2 treg gap and same
    # frequency, group them into a supernode.
    for ch, instructions in timed.items():
        channel_programs[ch] = []
        current_group: list[TimedInstruction] = []

        def flush_group():
            if not current_group:
                return
            
            candidate = make_supernode(current_group)
            key = candidate.key

            if key not in unique_supernodes:
                unique_supernodes[key] = candidate
            else:
                if candidate.preferred_resolution < unique_supernodes[key].preferred_resolution:
                    unique_supernodes[key] = candidate

            first = current_group[0].instruction

            channel_programs[ch].append(
                SuperNodeInstance(
                    supernode_key=key,
                    channel=ch,
                    t_start=current_group[0].t_start,
                    frequency=first.frequency,
                    global_phase=first.phase,
                    global_amplitude=first.amplitude,
                )
            )

            current_group.clear()

        for i, timed_inst in enumerate(instructions):
            inst = timed_inst.instruction

            if not inst.pulse.waveform_mode:
                channel_programs[ch].append(
                    ConstantPulseInstance(
                        channel=ch,
                        t_start=timed_inst.t_start,
                        length_treg=inst.pulse.length_treg,
                        frequency=inst.frequency,
                        phase=inst.phase,
                        amplitude=inst.amplitude,
                    )
                )
                continue

            current_group.append(timed_inst)

            is_last = i + 1 == len(instructions)

            if is_last:
                flush_group()
                continue

            next_inst = instructions[i + 1]
            gap = next_inst.t_start - timed_inst.t_end

            same_frequency = (
                next_inst.instruction.frequency == inst.frequency
            )

            next_is_waveform = next_inst.instruction.pulse.waveform_mode

            if gap >= split_gap or not next_is_waveform:
                flush_group()
                # TODO: handle constant pulses here, they dont have to be made into a supernode (they have no shape)
            elif not same_frequency:
                raise ValueError(f"Pulse sequence on channel {ch} can not be realized,"
                                 f" as timing requires combination of 2+ pulses into a single waveform supernode, "
                                 f"but the pulses have different frequencies.")

        flush_group()

    return unique_supernodes, channel_programs


##########################################
# Waveform creation step!
##########################################
def compute_waveform_memory(supernodes: dict[tuple, SuperNode]) -> dict[tuple, int]:
    """
    Compute the amount of waveform memory needed for each supernode based on preferred resolution and duration.
    Create waveform tiles for each supernode (for given resolution).
    """
    waveform_memory = {}
    expended_memory = 0

    for key, supernode in supernodes.items():
        # Calculate the number of samples needed based on preferred resolution
        num_samples = np.ceil(supernode.duration_ftsamp / FTSAMP_PER_TREG + 1) * FTSAMP_PER_TREG \
                                    * supernode.preferred_resolution // FTSAMP_PER_TREG
        
        expended_memory += num_samples
        waveform_memory[key] = num_samples

    # For now just flag the error and let the user reinitialize with a different preferred resolution
    if expended_memory > WAVEFORM_MEMORY_SIZE:
        raise ValueError(f"Total waveform memory required ({expended_memory} samples)" 
                         f"exceeds available memory ({WAVEFORM_MEMORY_SIZE} samples).")

    return waveform_memory


@dataclass(frozen=True)
class WaveformBase:
    """
    IQ rrepresentation of the basic waveform tile.
    """
    i_data: np.ndarray
    q_data: np.ndarray
    phase_adjustment: float # The angle we adjusted the waveform by
    scale_factor: float     # The scale factor for the waveform (maximum amplitude)

def create_waveform_base(supernode: SuperNode) -> WaveformBase:
    """
    creates the IQ representation of the basic waveform tile
    NOTE: in a future pass, we may want to allow the optimizer to rotate the waveform base by 90 degrees intervals
        to simplify instruction calls (reduce NCO phase adjustments)
    """
    shape_arr = supernode.equivalent_shape

    # Filter only values that can impact amplitude NOTE: Shape arrays are normalized to 1
    filtered_arr = shape_arr[np.abs(shape_arr) > 1/np.sqrt(2)]

    max_factor = 1
    rot_angle = 45

    # Only allow discrete rotations to simplify NCO phase adjustments and avoid similar shapes from being rotated by differing angles
    # TODO: tie-breaking?, really only 45 degrees and 0 degrees are somewhat preferential
    for angle in np.arange(0, 90, 90/16):
        rotated = filtered_arr * np.exp(-1j * np.deg2rad(angle))
        max_real = np.max(np.abs(np.real(rotated)))
        max_imag = np.max(np.abs(np.imag(rotated)))

        factor = 1/max(max_real, max_imag)

        if factor > max_factor + 1e-4:
            max_factor = factor
            rot_angle = angle

    # Generate shape
    optimized_shape = shape_arr * np.exp(-1j * np.deg2rad(rot_angle)) * SAMPLE_AMP
    idata = np.real(optimized_shape, np.int16) # NOTE: casting as regular int should work fine too
    qdata = np.imag(optimized_shape, np.int16)

    return WaveformBase(i_data=idata, q_data=qdata, phase_adjustment=rot_angle, scale_factor=max_factor)

def waveform_layout(supernodes: dict[tuple, SuperNode], waveform_memory: dict[tuple, int]):
    """
        TODO: Optimize the layout based on instruction and memory requirements. 
        We can use multiplicative structured tiling, or shifting tiling if memory is less tight to reduce tproc demand.

        For multiplication it will generally be more optimal to organize:   
            Waveform1[index1, index2, ...], Waveform2[index1, index2, ...], etc.
        For tiling it might be more optimal to organize:
            index1[Waveform1, Waveform2, ...], index2[Waveform1, Waveform2, ...], etc., particularly if the resolutions match
            and otherwise by waveform first.
            This is more dependent on the specific lengths of the waveforms 
            (if they are close to power-of-2 lengths, or if we can get closer with concatenating different types)

        We also need to determine what the speed up on instructions is. If multiplication is the same as shifting no need to shift

        # TODO:
        #   Create waveform tiles for each supernode (for given resolution)

    """
    pass

# TODO: Waveform construction pass:
def waveform_construction_pass(supernodes: dict[tuple, SuperNode], channel_programs: dict[int, list[TimedInstruction]]):
    waveform_memory = compute_waveform_memory(supernodes)
    waveform_bases = {key: create_waveform_base(node) for key, node in supernodes.items()}
    # waveform_layout(supernodes, waveform_memory)

    return waveform_memory, waveform_bases





# TODO: Determine how to handle changes in resolution, for memory we probably want to stick to indexing by smallest offset to highest offset
# and at all memory indexes. This means that the memory step size should be 1. but that also means having to add a shifting operation when 
# going between resolutions. It does keep the delay instructions more consistent (operating in single ftsamp). But might require an additional
# register to avoid going back and forth constantly.

# NOTE: Additional consideration: any pulse that does not see variable timing prior to it
# (fine timing sweep or have many repetitions with fine timing involved),
#  could be allocated a single waveform tile. But my compiler has currently been built 
# 

# TODO: 
# - run through the pulse view and see if its generating correctly
# - add front end compatibility for the other instruction types (sync, trigger, readout)
# - add sweep compatibility
# - Start on the compilation pass, at this point i've basically removed a lot of the user supplied context 
#           (like syncs and timings, which i might need to find a way to lower to the compiler, but for now I'll brute force??)



# Ideas: need a function to capture behavior based on sweep values, if avoiding a roll out