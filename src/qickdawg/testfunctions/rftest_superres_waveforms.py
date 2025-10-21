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
        "pi_len_samples", # length of pi pulse
        "tau_len_samples", # length of tau delay
        "n_cpmg", # number of pulses

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

        # configure the x, y waveforms for different sample offsets
        self.pi_waveform_len_treg = int(np.ceil((self.cfg.pi_len_samples + 15) / 16))
        self.half_pi_waveform_len_treg = int(np.ceil((self.cfg.pi_len_samples//2 + 15) / 16))

        zero_data = np.zeros(self.pi_waveform_len_treg * 16)
        for i in range(16):
            # half pi
            i_data = np.zeros(self.half_pi_waveform_len_treg * 16)
            i_data[i:i + self.cfg.pi_len_samples//2] = 1
            i_data*=self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=i_data, qdata=zero_data[:self.half_pi_waveform_len_treg * 16])
            
            # pi pulse
            i_data = np.zeros(self.pi_waveform_len_treg * 16)
            i_data[i:i + self.cfg.pi_len_samples] = 1
            i_data*=self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=i_data, qdata=zero_data)

        # How much delay is in the waveforms:
        self.pi_len_unused = self.pi_waveform_len_treg*16 - self.cfg.pi_len_samples
        self.half_pi_len_unused = self.half_pi_waveform_len_treg*16 - self.cfg.pi_len_samples//2
        
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)

        self.sample_step = self.cfg.tau_len_samples % 16
        self.treg_step = self.cfg.tau_len_samples // 16

        # CPMG waveform
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.trigger(
            pins=[self.cfg.trigger_gate_pmod],
            width=self.cfg.trigger_width_treg,
            t=0
        )
        self.sync_all()

        """
        # Half pi pulse, 0 sample offset, phase = x
        self.pulse(ch=self.cfg.mw_channel)

        self.sample_offset = (self.pi_len_unused-self.half_pi_len_unused)%16

        for i in range(self.cfg.n_cpmg):
            self.sample_offset = self.sample_offset + self.sample_step
            self.treg_offset = self.sample_offset//16
            self.treg_offset = self.treg_offset + self.treg_step
            self.sample_offset = int(self.sample_offset % 16)

            # Pi pulse
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pi_{self.sample_offset}", phase=self.deg2reg((i%2)*90))
            self.sync_all(self.treg_offset)
            self.pulse(ch=self.cfg.mw_channel)

        self.sample_offset = self.sample_offset + self.sample_step
        self.treg_offset = self.sample_offset//16
        self.treg_offset = self.treg_offset + self.treg_step
        self.sample_offset = int(self.sample_offset % 16)

        # Pi/2 X pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"half_pi_{self.sample_offset}", phase=self.deg2reg(0))
        self.sync_all(self.treg_offset)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
        """

        for i in range(3):
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pi_0", phase=self.deg2reg(0))
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all(self.cfg.tau_len_samples)

        for i in range(16):
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pi_{i}", phase=self.deg2reg(0))
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all(self.cfg.tau_len_samples)

        self.sync_all()



