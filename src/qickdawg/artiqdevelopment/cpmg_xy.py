"""
CPMG-XY
=======================================================================
Basic CPMG XY class for generating Artiq triggerable sequences.

Triggering:
 - Call using the run_rounds() method for retriggering
    - Needs to be passed the soc parameter (i.e. qd.soc)
 - Use PMOD1_0 for external triggering
"""

import os
import numpy as np

from .nvaverageprogram import NVAveragerProgram
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times


class CPMG_XY(NVAveragerProgram):
    """
    An NVAveragerProgram class that generates RF gain and frequency stepping
    sequences implementing a CPMG XY protocol.
    """

    required_cfg = [
        # Microwave parameters
        "mw_channel",              # DAC Channel
        "mw_freg",                 # MW Frequency (~1405 MHz for DQP)
        "mw_nqz",                  # Nyquist zone (1 while mw_freq < 2.495 GHz)
        "mw_gain",                 # MW Gain (0–32767)

        # Timing parameters (integer multiple of 3.26 ns)
        "tau_treg",                # Delay between XY pulses
        "half_pi_pulse_len_treg",  # π/2 pulse length
        "pi_pulse_len_treg",       # π pulse length
        "n_pulses",                # Number of π pulses
    ]

    def initialize(self):
        """Initialize microwave generator and set up registers."""
        self.check_cfg()

        if not (0 <= self.cfg.mw_gain <= 32767):
            raise ValueError("Microwave gain must be between 0 and 32767.")

        # Get MW registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Set default π pulse registers
        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style="const",
            freq=self.cfg.mw_freg,
            length=self.cfg.pi_pulse_len_treg,
            gain=self.cfg.mw_gain,
        )
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(0))

        # Looping register for number of CPMG repetitions
        self.n_cpmg_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name="ncpmg",
            init_val=self.cfg.n_pulses / 2 - 1,
        )

        # Configurable additional initialization delay
        self.synci(100)

    def body(self):
        """
        CPMG XY sequence:
          - X π/2 pulse
          - (delay)
          - Loop: [X π pulse – delay – Y π pulse – delay] repeated n times
          - Final X π/2 pulse
        """
        self.n_cpmg_register.reset()  # Reset loop counter

        # Initial π/2 pulse
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.tau_treg) # delay

        # ----------------------------- LOOP START ----------------------------- #
        self.label("LOOP_ncpmg")

        for phase_deg in [0, 90]:
            self.set_pulse_registers(
                ch=self.cfg.mw_channel,
                length=self.cfg.pi_pulse_len_treg,
                phase=self.deg2reg(phase_deg),
            )
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all(self.cfg.tau_treg) # delay

        self.loopnz(
            self.n_cpmg_register.page,
            self.n_cpmg_register.addr,
            "LOOP_ncpmg",
        )
        # ----------------------------- LOOP END ------------------------------- #

        # Final π/2 pulse
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            length=self.cfg.half_pi_pulse_len_treg,
            phase=0,
        )
        self.pulse(ch=self.cfg.mw_channel)

        # NOTE: QICK-DAWG code tends to use sync_all,
        # but wait() may be more exact when only a single channel is configured.
