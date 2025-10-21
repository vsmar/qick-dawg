'''
RFTest CPMG-XY
=======================================================================
RFTest Envelope class used to test the shape of RF envelopes.
'''

# Use a QickSweep, not sure which one to use
from qick.averager_program import QickSweep
from ..nvpulsing.nvqicksweep import NVQickSweep # NVQickSweep requires a readout_integration time parameter

from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import floor
import os 

class RFTest_CPMG(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "mw_freg", # Microwave freq # ~1405 MHz per Tommy
        "trigger_gate_pmod", # PMOD pin for external trigger
        
        # Pulse Parameters
        "pi_len", # length of pi pulse
        "n_pulses", # number of pulses

        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", #MW Gain
        "reps",

        # Temporary parameters for development
        "trigger_width_treg",
        "relax_delay_treg"]
    
    def initialize(self):
        # NVConfiguration class does not have Gain units unlike freq, time, or phase
        self.check_cfg()

        if self.cfg.mw_gain < 0:
            assert 0, 'Smallest Microwave gain must be postive'
        elif self.cfg.mw_gain > 32767: # 30000 in lockinodmr
            assert 0, 'Largest Microwave gain exceeds maximum value'

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.setup_readout()

        half_pi = self.cfg.pi_len // 2

        # CPMG waveform
        idata = np.zeros(2**16)
        qdata = np.zeros(2**16)
        iq_index = 0 # variable for stepping through and setting I and Q data
        for i, step in enumerate(('half_X ' + 'X X '* (self.cfg.n_pulses//2 - 1) + 'half_X').split()):
            if step == "half_X":
                idata[iq_index:half_pi + iq_index] = 1
                iq_index += half_pi + self.cfg.tau_len
            elif step == "X":
                idata[iq_index:self.cfg.pi_len + iq_index] = 1
                iq_index += self.cfg.pi_len + self.cfg.tau_len
            elif step == "Y":
                qdata[iq_index:self.cfg.pi_len + iq_index] = 1
                iq_index += self.cfg.pi_len + self.cfg.tau_len
        idata = idata[:((iq_index + 15) // 16) * 16] # sample only up to nearest multiple of 16
        qdata = qdata[:((iq_index + 15) // 16) * 16] # sample only up to nearest multiple of 16
        idata*=self.soccfg.get_maxv(self.cfg.mw_channel)
        qdata*=self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="measure", idata=idata, qdata=qdata)
        self.set_pulse_registers(ch=self.cfg.mw_channel, style="arb", waveform="measure",
                                freq=self.cfg.mw_freg, phase=0, gain=self.cfg.mw_gain)
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.trigger(
            pins=[self.cfg.trigger_gate_pmod],
            width=self.cfg.trigger_width_treg,
            t=0
        )

        self.pulse(ch=self.cfg.mw_channel)
        self.wait_all()
        self.sync_all()
        self.sync_all(self.cfg.relax_delay_treg)