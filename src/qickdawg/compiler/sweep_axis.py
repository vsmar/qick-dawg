"""
Loop counter abstractions for QICK-DAWG.
Manage sweeps, loops and patterned loops.

SweepAxis
---------
Outer loop counter, used to sweep over a parameter space.
Value has physical meaning (time, frequency, repititions, etc)


RepeatAxis
----------
Structural loop counter, used to repeat a block of code.
Can be tied to a fixed value or a swept parameter.

Pattern
-------
Short patterns, for physical properties indexed by a specific Axis (SweepAxis or RepeatAxis).
Represents a fixed cyclic sequence of values (like phase in CPMG XY8).

Pattern Encoding Constraints
----------------------------
Currently only phase is supported. 
Additionally the masking + shift procedure used, places restrictions on the encodable values and the pattern length.
"""

import math
from dataclasses import dataclass
from typing import Literal

from units import UNIT_TABLE, UnitSpec

REGISTER_SIZE = 16 # TODO: Verify whether using 32 bits of the register is prohibitive 
                   # I believe it takes more instructions to load, but might be worth enabling


@dataclass(frozen=True)
class SweepAxis:
    """ Loop counter with physical meaning (only takes fixed values).
    
    start and stop are inclusive. stop is adjusted to fall on a valid step.
    num_steps is 0 indexed (excludes the initial value from count).
    """
    name: str
    start: int | float
    stop: int | float
    num_steps: int
    canonical_unit: str
    kind: str

    @property
    def min_value(self):
        return min(self.start, self.stop)


@dataclass(frozen=True)
class LinearSweepAxis(SweepAxis):
    step: int | float

    @classmethod
    def create(cls, name, start, stop, unit, step) -> SweepAxis:
        spec: UnitSpec = UNIT_TABLE[unit]
        c_start = spec.convert(start)
        c_stop = spec.convert(stop) 
        c_step = spec.convert(step)
        
        if c_step == 0:
            raise ValueError(f"SweepAxis '{name}' has a step == 0")
        if (c_stop - c_start) / c_step < 1:
            raise ValueError(f"SweepAxis '{name}' has invalid bounds, or step")
        
        num_steps = math.ceil((c_stop - c_start) / c_step)
        c_stop = c_start + num_steps * c_step

        return cls(name, c_start, c_stop, num_steps, spec.canonical_unit, spec.kind, c_step)


_EXP_FACTORS: dict[str, tuple[int, int]] = {
    "3/2": (3, 2),
    "5/4": (5, 4),
    "9/8": (9, 8),
    "17/16": (17, 16)
}

@dataclass(frozen=True)
class ExponentialSweepAxis(SweepAxis):
    factor: Literal["3/2", "5/4", "9/8", "17/16"]
    denom: int

    @classmethod
    def create(cls, name, start, stop, factor, unit) -> SweepAxis:
        spec: UnitSpec = UNIT_TABLE[unit]
        c_start = spec.convert(start)
        c_stop = spec.convert(stop)

        if factor not in _EXP_FACTORS.keys():
            raise ValueError(f"ExponentialSweepAxis '{name}' factor must be in {_EXP_FACTORS.keys()}, "
                             f"got {factor}")

        numerator, denominator = _EXP_FACTORS[factor]

        if c_start >= c_stop:
            raise ValueError(
                f"SweepAxis '{name}' has a start value greater than the stop value"
            )
        if (c_stop - c_start * numerator/denominator) <= 0:
            raise ValueError(
                f"SweepAxis '{name}' has bounds that fit less than a single step"
            )
        if c_start < denominator:
            raise ValueError(
                f"SweepAxis '{name}' has a canonical start value less than the denominator {denominator}"
                f"this leads to infinite loops in the hardware"
            )
        
        x, num_steps = c_start, 0
        while x < c_stop:
            x *= numerator
            x //= denominator
            num_steps += 1
        c_stop = x

        return cls(name, c_start, c_stop, num_steps, spec.canonical_unit, spec.kind, factor, denominator)


@dataclass(frozen=True)
class RepeatAxis:
    name: str
    bound: int | SweepAxis
    canonical_unit = "count"

    def __post_init__(self):
        if isinstance(self.bound, int):
            if self.bound <= 0:
                raise ValueError(
                    f"RepeatAxis '{self.name}': fixed bound must be a "
                    f"positive integer, got {self.bound}."
                )
        elif isinstance(self.bound, SweepAxis):
            if self.bound.kind != "count":
                raise ValueError(
                    f"RepeatAxis '{self.name}': a SweepAxis bound must have "
                    f"kind='count', got kind='{self.bound.kind}' for axis "
                    f"'{self.bound.name}'."
                )
        else:
            raise TypeError(
                f"RepeatAxis '{self.name}': bound must be an int or SweepAxis, "
                f"got {type(self.bound).__name__}."
            )

@dataclass(frozen=True)
class Pattern:
    values: tuple[int | float, ...]
    canonical_unit: str
    kind: str
    axis: SweepAxis | RepeatAxis

    @classmethod
    def create(cls, values: list[int | float], unit: str, axis: SweepAxis | RepeatAxis) -> Pattern:
        spec: UnitSpec = UNIT_TABLE[unit]

        if spec.kind != "phase":
            raise ValueError(f"Patterns only supports phase, got {spec.kind}")
        
        # Check if pattern length is implementable
        n = len(values)
        if n <= 0:
            raise ValueError(f"Pattern must have at least one value")
        elif (n & (n - 1)) != 0:
            raise ValueError(f"Pattern length must be a power of 2, got {n}")
        
        # Check if there is a valid encoding for the pattern
        c_values = [spec.convert(v) for v in values]
        diffs = [v - c_values[0] for v in c_values]

        maximum = max(diffs)
        nonzero = [d for d in diffs if d != 0]
        gcd = math.gcd(*nonzero)
        encoding_bits = math.log2(maximum / gcd)
        if encoding_bits % 1 != 0:
            raise ValueError(
                f"Pattern values cannot be encoded in a valid bit representation"
            )
        else:
            encoding_bits = int(encoding_bits)

        if encoding_bits * n > REGISTER_SIZE:
            raise ValueError(
                f"Pattern values cannot be encoded in a valid bit representation. \n"
                f"Requires {encoding_bits} bits per value,"
                f"but pattern length is {n}, exceeding {REGISTER_SIZE} bits total."
            )

        return cls(tuple(c_values), spec.canonical_unit, spec.kind, axis)