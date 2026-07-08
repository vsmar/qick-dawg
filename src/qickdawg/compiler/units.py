"""
Unit conversions for QICK.


Hardware units: freg, preg, treg, samples, gain

Manages conversions between SI units and the hardware units for:
time:       ftsamp, treg, ns, us, s
phase:      preg, deg
frequency:  freg, Hz, MHz, GHz
Amplitude:  amp

NOTE: Does not handle the amplitude rectification directly, as that depends on waveform creation.
"""

from dataclasses import dataclass
from typing import Callable

import qickdawg as qd

# Define all params in one place
FTSAMP_PER_TREG = 16
DAC_SAMPLE_RATE = 4915.2e6  # Hz
WAVEFORM_MEMORY_SIZE = 2**16  # samples
SAMPLE_SIZE = 16  # bits (determines max amplitude for I and Q channels)

@dataclass(frozen=True)
class UnitSpec:
    canonical_unit: str
    kind: str
    converter: Callable[[float | int], int | float]
    requires_int: bool = False
    canonical_int: bool = True # Currently an exception for the amplitude
    negative_allowed: bool = False

    def convert(self, value: int | float) -> int | float:
        if not self.negative_allowed and value < 0:
            if self.canonical_int:
                raise ValueError(f"{self.canonical_unit} units cannot be negative, got {value}") 
            else:
                raise ValueError(f"{self.kind} type units cannot be negative, got {value}")
        
        value = self.converter(value) # rounding is done in the converter function, if needed

        if self.requires_int and not isinstance(value, int): # NOTE: convert to warning if too frequent
            raise TypeError(f"{self.canonical_unit} must be an integer " +
                            f"when expressed in {self.canonical_unit}, got {type(value)}")
        
        return value
    
    # TODO: add bounds / constraints on units?

UNIT_TABLE = {
    "ftsamp":   UnitSpec("ftsamp", "time", lambda x: x, requires_int=True),
    "treg":     UnitSpec("ftsamp", "time", lambda x: x * FTSAMP_PER_TREG, requires_int=True),
    "ns":       UnitSpec("ftsamp", "time", lambda x: qd.soccfg.us2cycles(x / 1e3 * FTSAMP_PER_TREG)),
    "us":       UnitSpec("ftsamp", "time", lambda x: qd.soccfg.us2cycles(x * FTSAMP_PER_TREG)),
    "s":        UnitSpec("ftsamp", "time", lambda x: qd.soccfg.us2cycles(x * 1e6 * FTSAMP_PER_TREG)),

    "preg":     UnitSpec("preg", "phase", lambda x: x, requires_int=True, negative_allowed=True),
    "deg":      UnitSpec("preg", "phase", lambda x: qd.soccfg.deg2reg(x)),

    "freg":     UnitSpec("freg", "frequency", lambda x: x, requires_int=True),
    "Hz":       UnitSpec("freg", "frequency", lambda x: qd.soccfg.freq2reg(x / 1e6)),
    "MHz":      UnitSpec("freg", "frequency", lambda x: qd.soccfg.freq2reg(x)),
    "GHz":      UnitSpec("freg", "frequency", lambda x: qd.soccfg.freq2reg(x * 1e3)),

    # TODO: Check if nco amplitude takes negative values
    "amp":      UnitSpec("amp", "amplitude", lambda x: float(x), canonical_int=False, negative_allowed=False),
    "count":    UnitSpec("count", "count", lambda x: x),
}

COARSE_TIME_UNIT_TABLE = {
    "ftsamp":   UnitSpec("treg", "coarse_time", lambda x: x / FTSAMP_PER_TREG, requires_int=True),
    "treg":     UnitSpec("treg", "coarse_time", lambda x: x, requires_int=True),
    "us":       UnitSpec("treg", "coarse_time", lambda x: qd.soccfg.us2cycles(x)),
    "ns":       UnitSpec("treg", "coarse_time", lambda x: qd.soccfg.us2cycles(x / 1e3)),
    "s":        UnitSpec("treg", "coarse_time", lambda x: qd.soccfg.us2cycles(x * 1e6)),
}

SI_DEFAULT_UNIT = {
    "coarse_time": "s",
    "time": "s",
    "phase": "deg",
    "frequency": "Hz",
    "amplitude": "amp",
    "count": "count",
}
