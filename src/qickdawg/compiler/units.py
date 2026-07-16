"""
Unit conversions for QICK.


Hardware units: freg, preg, treg, ftsamp, gain

Manages conversions between SI units and the hardware units, per kind:

    time         fine timing, canonical "ftsamp"  (1 DAC sample)
    coarse_time  tProc-granular timing, canonical "treg" (16 ftsamp) --
                 used by constant-amplitude coarse-mode pulses whose length
                 the tProc itself counts (and can therefore sweep)
    phase        canonical "preg"
    frequency    canonical "freg"
    amplitude    canonical "amp" (float ratio)
    count        canonical "count"

Several unit *names* exist under more than one kind ("us" is both a fine and
a coarse time). resolve_unit(unit, kind) disambiguates; when kind is omitted,
fine "time" wins for shared names, so existing fine-time code keeps working
and coarse users say kind="coarse_time" explicitly.

soccfg swap-in
--------------
All register conversions go through get_soccfg(). If qickdawg (and its board
soccfg) is importable it is used; otherwise a SoftSocCfg implementing the
same formulas in software is installed automatically, so sequences can be
compiled and simulated off-board with numerically consistent conversions.
Swap explicitly at any time:

    import units
    units.set_soccfg(my_board_soccfg)          # hardware
    units.set_soccfg(units.SoftSocCfg())       # software, default geometry
    units.set_soccfg(units.SoftSocCfg(fs_dac_mhz=6144.0))   # custom board

NOTE: Does not handle the amplitude rectification directly, as that depends on waveform creation.
"""

import warnings
from dataclasses import dataclass
from typing import Callable

# Define all params in one place
FTSAMP_PER_TREG = 16
DAC_SAMPLE_RATE = 4915.2e6  # Hz
WAVEFORM_MEMORY_SIZE = 2**16  # samples
SAMPLE_SIZE = 16  # bits (determines max amplitude for I and Q channels)


# ---------------------------------------------------------------------------
# soccfg provider (hardware when available, software otherwise)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SoftSocCfg:
    """Software soccfg: the standard QICK conversion formulas, no board.

    Mirrors the calls the unit table needs (us2cycles / deg2reg / freq2reg)
    plus the inverses the simulator needs (reg2deg / reg2freq / cycles2us).
    Defaults match the 4x2-class geometry used across this codebase
    (fs_dac = 4915.2 MHz, 32-bit frequency and phase words, tProc clock =
    fs_dac / 16). Adjust the fields for a different board, or set_soccfg()
    the real one -- the formulas are chosen so that swapping in hardware
    changes rounding only, not semantics.
    """
    fs_dac_mhz: float = DAC_SAMPLE_RATE / 1e6
    b_freq: int = 32
    b_phase: int = 32

    @property
    def f_tproc_mhz(self) -> float:
        return self.fs_dac_mhz / FTSAMP_PER_TREG

    # -- time ---------------------------------------------------------------
    def us2cycles(self, us: float) -> int:
        return int(round(us * self.f_tproc_mhz))

    def cycles2us(self, cycles: int) -> float:
        return cycles / self.f_tproc_mhz

    # -- frequency ----------------------------------------------------------
    def freq2reg(self, f_mhz: float) -> int:
        return int(round(f_mhz * 2**self.b_freq / self.fs_dac_mhz)) % 2**self.b_freq

    def reg2freq(self, reg: int) -> float:
        """freg -> MHz (qick convention)."""
        return reg * self.fs_dac_mhz / 2**self.b_freq

    # -- phase ---------------------------------------------------------------
    def deg2reg(self, deg: float) -> int:
        return int(round((deg % 360.0) * 2**self.b_phase / 360.0)) % 2**self.b_phase

    def reg2deg(self, reg: int) -> float:
        return (reg % 2**self.b_phase) * 360.0 / 2**self.b_phase


_soccfg = None
_fallback_announced = False


def get_soccfg():
    """The active soccfg: qickdawg's board config if available, else SoftSocCfg."""
    global _soccfg, _fallback_announced
    if _soccfg is None:
        cfg = None
        try:
            import qickdawg as qd
            cfg = getattr(qd, "soccfg", None)
        except ImportError:
            cfg = None
        if cfg is None:
            cfg = SoftSocCfg()
            if not _fallback_announced:
                warnings.warn(
                    "units: qickdawg soccfg not available -- using SoftSocCfg "
                    "(software conversions). Call units.set_soccfg(...) to "
                    "swap in a board config.",
                    stacklevel=2,
                )
                _fallback_announced = True
        _soccfg = cfg
    return _soccfg


def set_soccfg(cfg) -> None:
    """Swap the conversion backend (board soccfg or a SoftSocCfg)."""
    global _soccfg
    _soccfg = cfg


# ---------------------------------------------------------------------------
# Unit specs
# ---------------------------------------------------------------------------

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


def _ftsamp_to_treg(x):
    """Exact ftsamp -> treg; coarse quantities must land on treg boundaries."""
    if x % FTSAMP_PER_TREG != 0:
        raise ValueError(
            f"coarse_time value of {x} ftsamp is not a multiple of "
            f"{FTSAMP_PER_TREG}; coarse (tProc-timed) quantities are treg-"
            f"granular."
        )
    return int(x // FTSAMP_PER_TREG)


# Nested by kind: unit names may repeat across kinds with different
# converters AND different canonical units (time -> ftsamp, coarse -> treg).
UNIT_TABLE: dict[str, dict[str, UnitSpec]] = {
    "time": {
        "ftsamp": UnitSpec("ftsamp", "time", lambda x: x, requires_int=True),
        "treg":   UnitSpec("ftsamp", "time", lambda x: x * FTSAMP_PER_TREG, requires_int=True),
        "ns":     UnitSpec("ftsamp", "time", lambda x: get_soccfg().us2cycles(x / 1e3 * FTSAMP_PER_TREG)),
        "us":     UnitSpec("ftsamp", "time", lambda x: get_soccfg().us2cycles(x * FTSAMP_PER_TREG)),
        "s":      UnitSpec("ftsamp", "time", lambda x: get_soccfg().us2cycles(x * 1e6 * FTSAMP_PER_TREG)),
    },
    "coarse_time": {
        "ftsamp": UnitSpec("treg", "coarse_time", _ftsamp_to_treg, requires_int=True),
        "treg":   UnitSpec("treg", "coarse_time", lambda x: x, requires_int=True),
        "us":     UnitSpec("treg", "coarse_time", lambda x: get_soccfg().us2cycles(x)),
        "ns":     UnitSpec("treg", "coarse_time", lambda x: get_soccfg().us2cycles(x / 1e3)),
        "s":      UnitSpec("treg", "coarse_time", lambda x: get_soccfg().us2cycles(x * 1e6)),
    },
    "phase": {
        "preg": UnitSpec("preg", "phase", lambda x: x, requires_int=True, negative_allowed=True),
        "deg":  UnitSpec("preg", "phase", lambda x: get_soccfg().deg2reg(x)),
    },
    "frequency": {
        "freg": UnitSpec("freg", "frequency", lambda x: x, requires_int=True),
        "Hz":   UnitSpec("freg", "frequency", lambda x: get_soccfg().freq2reg(x / 1e6)),
        "MHz":  UnitSpec("freg", "frequency", lambda x: get_soccfg().freq2reg(x)),
        "GHz":  UnitSpec("freg", "frequency", lambda x: get_soccfg().freq2reg(x * 1e3)),
    },
    "amplitude": {
        # TODO: Check if nco amplitude takes negative values
        "amp": UnitSpec("amp", "amplitude", lambda x: float(x), canonical_int=False, negative_allowed=False),
    },
    "count": {
        "count": UnitSpec("count", "count", lambda x: x),
    },
}


def resolve_unit(unit: str, kind: str | None = None) -> UnitSpec:
    """Look up a UnitSpec by unit name, disambiguated by kind.

    kind given   -> exact lookup in that kind's table.
    kind omitted -> fine "time" wins for shared names (backward compatible);
                    otherwise the unique owner of the name. Coarse-time
                    lookups therefore always require kind="coarse_time".
    """
    if kind is not None:
        table = UNIT_TABLE.get(kind)
        if table is None:
            raise KeyError(f"Unknown unit kind '{kind}'. Kinds: {list(UNIT_TABLE)}")
        spec = table.get(unit)
        if spec is None:
            raise KeyError(
                f"Unit '{unit}' is not defined for kind '{kind}'. "
                f"Available: {list(table)}"
            )
        return spec

    if unit in UNIT_TABLE["time"]:
        return UNIT_TABLE["time"][unit]
    owners = [k for k, t in UNIT_TABLE.items() if unit in t]
    if len(owners) == 1:
        return UNIT_TABLE[owners[0]][unit]
    if not owners:
        raise KeyError(f"Unknown unit '{unit}'.")
    raise KeyError(f"Unit '{unit}' is ambiguous across kinds {owners}; pass kind=.")


# canonical unit each kind's values land in (for boundary checks)
KIND_CANONICAL = {
    "time": "ftsamp",
    "coarse_time": "treg",
    "phase": "preg",
    "frequency": "freg",
    "amplitude": "amp",
    "count": "count",
}

SI_DEFAULT_UNIT = {
    "time": "s",
    "coarse_time": "s",
    "phase": "deg",
    "frequency": "Hz",
    "amplitude": "amp",
    "count": "count",
}