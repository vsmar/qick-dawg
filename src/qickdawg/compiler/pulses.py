"""
Pulse shape and pulse definition, user-facing.

DefinePulse:
    Fundamental description of a pulse, with global parameters.
    Parameters like frequency, phase, amplitude can be overwritten 
    for specific instance in the play instruction.

Shape:
    User facing class to define the normalized shape of a pulse.
    Users can use predefined shapes, or define their own 
    with an array (in samples) or a function. Complex numbers are supported.

Parameter conventions (shared with ir.play / ir.delay via parameters.py)
------------------------------------------------------------------------
  bare int | float  -> constant Parameter evaluated in the SI default units for the
                       field's kind (s, Hz, deg, amp ratio)
  Parameter         -> passed through, user specifies units at construction,
                       e.g. Parameter.constant("len", 512, "ftsamp")
                       (Can be constants anywhere, or sweeps / patterns for DefinePulse fields)

DefinePulse.amplitude / .phase / .frequency accept swept or patterned values.

Two pulse modes, two length rules:
  * arbitrary waveform (shape=...): the Shape fixes the length. Shape factory
    lengths accept CONSTANTS ONLY -- a shape's length fixes its waveform-
    memory footprint, so sweeping it is structurally unsupported.
  * constant-amplitude coarse mode (length_coarse=...): the tProc itself
    counts the pulse out in treg, so the length is a coarse_time Parameter
    and MAY BE SWEPT (LinearSweepAxis with kind="coarse_time"). The timing
    pass keeps a swept length affine exactly like a swept delay.

NOTE: bare-number lengths are SECONDS under this convention (SI default for
time), no longer raw DAC samples. Use Parameter.constant(name, n, "ftsamp")
or "treg" (with kind="coarse_time" for coarse lengths) for sample-exact values.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

import hashlib
import numpy as np

from .parameters import Parameter, coerce_param
from .units import FTSAMP_PER_TREG, DAC_SAMPLE_RATE, WAVEFORM_MEMORY_SIZE, SAMPLE_SIZE

# A single shape longer than this cannot fit waveform memory.
# Catches the mistake of passing a sample count where seconds are expected.
_MAX_SHAPE_FTSAMP = WAVEFORM_MEMORY_SIZE * FTSAMP_PER_TREG


def _shape_length_ftsamp(length, where: str) -> int:
    """Shape length -> ftsamp int. Constant fine-time values only."""
    param = coerce_param(length, "time", name=where)
    if param is None:
        raise ValueError(f"{where}: a length is required.")
    if not param.is_constant:
        raise ValueError(
            f"{where} must be a constant; it is driven by "
            f"{type(param.axis).__name__} "
            f"'{getattr(param.axis, 'name', param.name)}'. Shapes cannot be "
            f"swept -- use a constant-amplitude coarse-mode pulse "
            f"(DefinePulse(length_coarse=...)) for sweepable lengths."
        )
    n = int(param.value)
    if n > _MAX_SHAPE_FTSAMP:
        raise ValueError(
            f"{where}: {n} ftsamp is beyond what waveform memory can hold "
            f"({_MAX_SHAPE_FTSAMP} ftsamp at resolution 16). If a bare-number "
            f"length was meant as samples, note bare lengths are seconds; use "
            f"Parameter.constant(name, n, 'ftsamp')."
        )
    return n


# ---------------------------------------------------------------------------
# Pulse Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """
    User facing class to define the normalized shape of a pulse.

    Amplitude is set elsewhere. Shape values should satisfy:
        abs(shape) <= 1

    Users can use predefined shapes, or define their own with an array or a function.
    Complex shapes are supported and automatically inferred from complex array values or functions.

    Factory `length` arguments follow the shared parameter convention:
    bare numbers are SECONDS; pass Parameter.constant(name, n, "ftsamp") /
    "treg" / "us" to choose units. Lengths must be constant — swept or
    patterned lengths are rejected (see module docstring). Arrays passed to
    custom() are taken literally, one value per DAC sample.
    """
    data: np.ndarray
    params: dict | None = None  # for debugging and tracing
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
    def square(length):
        n = _shape_length_ftsamp(length, "Shape.square length")
        data = np.ones(n, dtype=float)
        return Shape(data=data, params={"source": "square"})

    @staticmethod
    def hermite(length, eta: float):  # TODO: CHECK MATH
        """
        length: seconds, or a constant time Parameter (e.g. unit="ftsamp")
        """
        n = _shape_length_ftsamp(length, "Shape.hermite length")
        t = np.arange(n, dtype=float)
        u = ((t - n / 2) / (0.1667 * n)) ** 2
        f = (1 - eta * u) * np.exp(-u)

        arr = np.array(f, dtype=float)
        return Shape(data=arr, params={"source": "hermite", "eta": eta})

    @staticmethod
    def custom(data: np.ndarray):
        return Shape(data=np.asarray(data), params={"source": "custom_array"})

    @staticmethod
    def from_func(func: Callable, length, **params):
        n = _shape_length_ftsamp(length, "Shape.from_func length")
        t = np.arange(0, n)
        arr = np.asarray(func(t, **params))
        return Shape(data=arr, params={"source": "custom_func", **params})

    @staticmethod
    def _validate_array(arr: np.ndarray):
        if arr.ndim != 1:
            raise ValueError("Custom shape data must be a 1D real or complex array.")

        if len(arr) < FTSAMP_PER_TREG + 2:  # This should always result in a valid waveform length of 3 treg or more (based on the other constraints)
            raise ValueError(f"Custom shape data must be at least ({FTSAMP_PER_TREG + 2} DAC samples).")

        if len(arr) > _MAX_SHAPE_FTSAMP:
            raise ValueError(
                f"Shape is {len(arr)} DAC samples, beyond what waveform memory "
                f"can hold ({_MAX_SHAPE_FTSAMP} ftsamp at resolution 16). If a "
                f"bare-number length was meant as samples, note bare lengths "
                f"are seconds; use Parameter.constant(name, n, 'ftsamp')."
            )

        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("Shape data must contain numeric real or complex values.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Shape data must not contain NaN or infinite values.")


@dataclass(frozen=True)
class DefinePulse:
    """ User-facing default pulse definition.

    amplitude / phase / frequency follow the shared parameter convention and
    are stored as Parameter after __post_init__:
        bare number             -> constant, SI units (amp ratio, deg, Hz)
        Parameter               -> as constructed (explicit units)
        SweepAxis / RepeatAxis  -> swept handle
        Pattern                 -> patterned handle (restricted to phase)
    The global parameters are defaults, play() values override them for a pulse instance.

    Modes:
        arbitrary waveform: Requires shape, length inherited from shape
        constant-amplitude/phase coarse mode: Requires length_coarse, bare
        numbers are SECONDS (no shape). length_coarse is a coarse_time
        Parameter (canonical treg) and is SWEEPABLE: pass a LinearSweepAxis
        created with kind="coarse_time" and the timing pass keeps the pulse
        end affine in that counter.

    NOTE:
        Any waveform can be encoded with up to 1 amplitude, but select waveforms can have higher amplitude cielings
        (up to sqrt(2) if relative phases in a shape are mod 90 degrees, or 2 for constant-amp/phase pulses).
    """
    name: str
    amplitude: object    # -> Parameter; constants must satisfy [0, ceiling]
    phase: object        # -> Parameter; bare numbers are deg
    frequency: object    # -> Parameter; bare numbers are Hz
    channel: int

    shape: Shape | None = None    # Only set when waveform_mode = True
    length_coarse: object | None = None  # Constant-amp/phase coarse mode length; bare numbers are SECONDS
    preferred_resolution: Literal['auto', 1, 2, 4, 8, 16] = 'auto'  # TODO: update based on 2^n

    @property
    def waveform_mode(self) -> bool:
        return self.shape is not None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Pulse name must be a non-empty string.")

        # -- one mode only ---------------------------------------------------
        if self.shape is None and self.length_coarse is None:
            raise ValueError(
                f"Pulse '{self.name}': must specify either a Shape (arbitrary "
                f"waveform mode) or a length (constant-amp/phase coarse mode)."
            )
        if self.shape is not None and self.length_coarse is not None:
            raise ValueError(
                f"Pulse '{self.name}': cannot have both a shape and a length; "
                f"a Shape carries its own length."
            )
        if self.shape is not None and not isinstance(self.shape, Shape):
            raise TypeError(
                f"Pulse '{self.name}': shape must be a Shape, "
                f"got {type(self.shape).__name__}."
            )

        # -- value parameters (swept / patterned allowed) ---------------------
        for fname in ("amplitude", "phase", "frequency"):
            param = coerce_param(getattr(self, fname), fname,
                                 name=f"{self.name}.{fname}")
            if param is None:
                raise ValueError(f"Pulse '{self.name}': {fname} is required.")
            object.__setattr__(self, fname, param)

        # -- constant-amplitude/phase coarse mode length (sweepable) ----------
        if self.length_coarse is not None:
            param = coerce_param(self.length_coarse, "coarse_time",
                                 name=f"{self.name}.length_coarse")
            if param.min_value is None or param.min_value < 3:
                raise ValueError(
                    f"Pulse '{self.name}': length must be at least 3 treg "
                    f"({3 * FTSAMP_PER_TREG} DAC samples) over every "
                    f"iteration, got minimum {param.min_value} treg."
                )
            object.__setattr__(self, "length_coarse", param)
            # constant-amp/phase coarse mode pulses are tProc-generated at treg granularity
            object.__setattr__(self, "preferred_resolution", 16)

    # TODO: Implement preliminary amplitude validation

    @property
    def length_treg(self) -> Parameter | None:
        """Coarse-mode length handle (a coarse_time Parameter, maybe swept)."""
        return self.length_coarse

    @property
    def length_ftsamp(self) -> int:
        """Concrete length in ftsamp. Only defined when the length is fixed:
        waveform mode, or coarse mode with a constant length. A swept coarse
        length has no single length -- the timing pass and simulator read
        length_coarse directly and keep the end time affine in its counter.
        """
        if self.waveform_mode:
            return self.shape.length
        if self.length_coarse.is_constant:
            return self.length_coarse.value * FTSAMP_PER_TREG
        raise ValueError(
            f"Pulse '{self.name}' has a swept coarse length "
            f"('{self.length_coarse.name}'); there is no single "
            f"length_ftsamp. Use length_coarse (affine in its counter)."
        )

    @property
    def length_sec(self) -> float:
        # Helper function to check pulse length
        return self.length_ftsamp / DAC_SAMPLE_RATE