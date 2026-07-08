"""
Parameter handles for explicitly defined parameters.


Necessary for handling parameters values dependent on sweeps and patterns.
Optional for fixed parameters, but allows user to specify units (MHz, GHz, degrees, us, etc.).
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from itertools import count
from typing import Literal
 
from sweep_axis import LinearSweepAxis, ExponentialSweepAxis, Pattern, RepeatAxis, SweepAxis
from units import UNIT_TABLE, SI_DEFAULT_UNIT


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
    
    @property
    def min_value(self) -> int | float | None:
        if self.axis is None:
            return self._constant_value
        return self.axis.min_value # TODO: fallback needed for RepeatAxis and Pattern?



# ----------- Helper Function -----------------------------------------------
_param_autoname = count()

def coerce_param(
    value: int | float | Parameter | SweepAxis | RepeatAxis | Pattern | None,
    kind: str,
    *,
    name: str | None = None,
    field_name: str | None = None
) -> Parameter | None:
    """Normalize a user-supplied value to a Parameter, or None.

    Allows user to use the swept axis or pattern as a handle,
    by executing the appropriate conversion to a Parameter.

    Identical parameters with different names are still identifiable by internal representation.

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
    if name:       
        return Parameter.constant(
            name=name,
            value=value,
            unit=unit,
        )
    else:
        return Parameter.constant(
            name=f"_{field_name}_{next(_param_autoname)}",
            value=value,
            unit=unit,
        )