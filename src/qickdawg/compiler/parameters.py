"""
Parameter handles for explicitly defined parameters.


Necessary for handling parameters values dependent on sweeps and patterns.
Optional for fixed parameters, but allows user to specify units (MHz, GHz, degrees, us, etc.).

Unit kinds: some unit names exist under more than one kind ("us"/"treg"/
"ftsamp" are both fine "time" and "coarse_time"). Parameter.constant takes an
optional kind to disambiguate; without it, fine time wins for shared names.
coerce_param always knows the kind from its call site and checks that any
pre-built Parameter / axis landed in that kind's canonical unit, so e.g. a
fine-time axis passed where a coarse length is expected fails at definition.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from itertools import count
from typing import Literal
 
from .sweep_axis import LinearSweepAxis, ExponentialSweepAxis, Pattern, RepeatAxis, SweepAxis
from .units import KIND_CANONICAL, SI_DEFAULT_UNIT, resolve_unit


@dataclass(frozen=True)
class Parameter:
    name: str | None
    axis: SweepAxis | RepeatAxis | Pattern | None
    canonical_unit: str
    _constant_value: int | float | None = None

    @classmethod
    def constant(cls, name: str | None, value: int | float, unit: str,
                 kind: str | None = None) -> Parameter:
        spec = resolve_unit(unit, kind)
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
    def value(self) -> int | float:
        """Canonical value of a constant Parameter (public accessor)."""
        if not self.is_constant:
            raise ValueError(
                f"Parameter '{self.name}' is driven by "
                f"{type(self.axis).__name__}; it has no single value."
            )
        return self._constant_value

    @property
    def min_value(self) -> int | float | None:
        if self.axis is None:
            return self._constant_value
        return self.axis.min_value



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

    Every non-bare value is checked against the kind's canonical unit, so a
    handle built in the wrong unit system is rejected at the boundary.
    """
    if value is None:
        return None

    if isinstance(value, Parameter):
        param = value
    elif isinstance(value, Pattern):
        param = Parameter.from_pattern(value)
    elif isinstance(value, (SweepAxis, RepeatAxis)):
        param = Parameter.from_axis(value)
    elif isinstance(value, (int, float)):
        # bare number — wrap as SI constant of this kind
        unit = SI_DEFAULT_UNIT[kind]
        return Parameter.constant(
            name=name if name else f"_{field_name}_{next(_param_autoname)}",
            value=value,
            unit=unit,
            kind=kind,
        )
    else:
        raise TypeError(
            f"{name or field_name or 'param'}: expected a number, Parameter, "
            f"SweepAxis, RepeatAxis or Pattern, got {type(value).__name__}."
        )

    expected = KIND_CANONICAL.get(kind)
    if expected is not None and param.canonical_unit != expected:
        raise ValueError(
            f"{name or field_name or 'param'}: expected a {kind} value "
            f"(canonical '{expected}'), got canonical "
            f"'{param.canonical_unit}' from {param.name!r}. "
            + ("For coarse (tProc-timed) quantities build the handle with "
               "kind='coarse_time'." if expected == "treg" else "")
        )
    return param