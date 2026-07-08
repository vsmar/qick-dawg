"""
Parameter handles for explicitly defined parameters.


Necessary for handling parameters values dependent on sweeps and patterns.
Optional for fixed parameters, but allows user to specify units (MHz, GHz, degrees, us, etc.).
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Literal
 
from sweep_axis import LinearSweepAxis, ExponentialSweepAxis, Pattern, RepeatAxis, SweepAxis
from units import UNIT_TABLE


@dataclass(frozen=True)
class Parameter:
    name: str | None
    axis: SweepAxis | RepeatAxis | Pattern | None
    canonical_unit: str
    _constant_value: int | float | None = None

    @classmethod
    def constant(cls, name: str | None, value: int | float, unit: str) -> Parameter:
        spec = UNIT_TABLE[unit]
        return cls(name, None, spec.canonical_unit, _constant_value=spec.convert(value))
    
    @classmethod
    def from_axis(cls, axis: SweepAxis | RepeatAxis) -> Parameter:
        if not isinstance(axis, (SweepAxis, RepeatAxis)):
            raise TypeError(
                f"from_axis expects SweepAxis or RepeatAxis, got {type(axis).__name__}. "
                f"For patterns use from_pattern()."
            )
        canonical_unit = getattr(axis, "canonical_unit", "count")
        return cls(axis.name, axis, canonical_unit)

    @classmethod
    def from_pattern(cls, pattern: Pattern) -> Parameter:
        if not isinstance(pattern, Pattern):
            raise TypeError(
                f"from_pattern expects a Pattern, got {type(pattern).__name__}."
            )
        return cls(pattern.axis.name, pattern, pattern.canonical_unit)

    @property
    def is_constant(self) -> bool:
        return self.axis is None

    @property
    def driving_axis(self) -> SweepAxis | RepeatAxis | None:
        """The structural axis actually driving this parameter's register,
        unwrapping one level if this Parameter came from a Pattern."""
        if isinstance(self.axis, Pattern):
            return self.axis.axis
        return self.axis