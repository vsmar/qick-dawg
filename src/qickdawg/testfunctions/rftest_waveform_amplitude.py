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

class RFTest_Amplitude(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "mw_freg", # Microwave freq # ~1405 MHz per Tommy
        "trigger_gate_pmod", # PMOD pin for external trigger
        
        # Pulse Parameters
        "waveform_len_samples", # length of pi pulse
        "tau_treg", # length of tau delay

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

        # Configure the waveforms for different sample offsets
        # Waveforms must have at least a length of 3 treg
        self.waveform_len_treg = max(int(np.ceil((self.cfg.waveform_len_samples + 15) / 16)), 3)

        zero_data = np.zeros(self.waveform_len_treg * 16)
        i_data = np.zeros(self.waveform_len_treg * 16)
        i_data[0: self.cfg.waveform_len_samples] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name=f"waveform_arb", idata=i_data, qdata=zero_data)

        # Set up registers for storing tau treg and sample offsets
        self.tau = self.new_gen_reg(self.cfg.mw_channel,
                                            name='tau',
                                            init_val=self.cfg.tau_treg)
        
        self.default_pulse_registers(ch=self.cfg.mw_channel, phase=0)
        
        # Set first half pi x
        self.set_pulse_registers(style='arb', freq=self.cfg.mw_freg, ch=self.cfg.mw_channel, waveform="waveform_arb", gain=self.cfg.mw_gain)

        # CPMG waveform
        self.synci(200)  # give processor some time to configure pulses



    def body(self):
        self.sync_all()

        # Half pi pulse, 0 sample offset, phase = x
        self.pulse(ch=self.cfg.mw_channel)

        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                     style='const',
                                     freq=self.cfg.mw_freg,
                                     length=self.cfg.waveform_len_samples,
                                     gain=self.cfg.mw_gain)
        
        self.sync(self.tau.page, self.tau.addr)


        self.pulse(ch=self.cfg.mw_channel)
