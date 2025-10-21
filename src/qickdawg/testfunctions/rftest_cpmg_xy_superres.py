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

        # Configure the waveforms for different sample offsets
        # Waveforms must have at least a length of 3 treg
        self.pi_waveform_len_treg = max(int(np.ceil((self.cfg.pi_len_samples + 15) / 16)), 3)
        self.half_pi_waveform_len_treg = max(int(np.ceil((self.cfg.pi_len_samples//2 + 15) / 16)), 3)

        zero_data = np.zeros(self.pi_waveform_len_treg * 16)
        for i in range(16):
            # half pi
            i_data = np.zeros(self.half_pi_waveform_len_treg * 16)
            i_data[i : i + self.cfg.pi_len_samples // 2] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=i_data, qdata=zero_data[: self.half_pi_waveform_len_treg * 16])
            
            # pi pulse
            i_data = np.zeros(self.pi_waveform_len_treg * 16)
            i_data[i: i + self.cfg.pi_len_samples] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=i_data, qdata=zero_data)

        # Compute how much delay is in the waveforms
        self.pi_len_unused = self.pi_waveform_len_treg*16 - self.cfg.pi_len_samples
        self.half_pi_len_unused = self.half_pi_waveform_len_treg*16 - self.cfg.pi_len_samples//2

        # Set up registers for storing tau treg and sample offsets
        self.tau_step = self.new_gen_reg(self.cfg.mw_channel,
                                            name='tau_step',
                                            init_val=self.cfg.tau_len_samples - self.pi_len_unused)
        
        # we can initialize sample_offset to already account for the first half_pi_pulse
        self.sample_offset = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='sample_offset',
                                                    init_val=(self.pi_len_unused-self.half_pi_len_unused)%16)
        self.treg_offset = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)
        
        # This is by default tau_len_samples // 16
        # And +1 when tau_sample_offset overflows
        self.treg_step = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_step',
                                                init_val=self.cfg.tau_len_samples // 16)

        self.sample_step = self.new_gen_reg(self.cfg.mw_channel,
                                                name='sample_step',
                                                init_val=self.cfg.tau_len_samples % 16)
        
        # CPMG loop register
        self.n_cpmg_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='ncpmg',
                                                    init_val=self.cfg.n_cpmg - 1)
        
        # comparison register
        self.comparison = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='comparison_val',
                                                    init_val=0)
        
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain)
        
        # Set first half pi x
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)

        # CPMG waveform
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all()

        # Half pi pulse, 0 sample offset, phase = x
        self.pulse(ch=self.cfg.mw_channel)

        # Loop pi-X, tau, pi-Y, tau
        self.n_cpmg_register.reset()
        self.label("LOOP_ncpmg")
        
        # X pulse
        # Configures assembly code for picking the waveform
        self.set_waveform("Execute_X_Pi_Pulse", "pi_", phase=0)
        self.sync(self.treg_offset.page, self.treg_offset.addr) # See below about placement
        self.pulse(ch=self.cfg.mw_channel)

        # Y pulse
        self.set_waveform("Execute_Y_Pi_Pulse", "pi_", phase=1)
        self.sync(self.treg_offset.page, self.treg_offset.addr) # I need to move syncs after pulses for correct timing behavior (to do this I need to modify my computing instructions)
        self.pulse(ch=self.cfg.mw_channel)

        self.loopnz(
                self.n_cpmg_register.page,
                self.n_cpmg_register.addr,
                'LOOP_ncpmg')
        
        # Pi/2 X pulse
        self.set_waveform("Execute_Last_Pulse", "half_pi_")
        self.sync(self.treg_offset.page, self.treg_offset.addr) # See above about placement
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Reset for next loop
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        self.sample_offset.reset() # reset the sample_offset adjustment 

  
    def set_waveform(self, label, pulse_type="pi_", phase=0):
        """
        Configures the assembly code necessary for setting the waveform
        """
        self.offset_computations()
        self.select_waveform(4, 8, 16, label, pulse_type, phase)
        self.label(label)
    
    def offset_computations(self):
        """
        Compute the sample_offset and treg_offset for the next pulse
        """
        # sample_offset = sample_offset + sample_step
        self.math(self.sample_offset.page, self.sample_offset.addr, self.sample_offset.addr, "+", self.sample_step.addr)
        # treg_offset = sample_offset >> 4
        self.mathi(self.sample_offset.page, self.treg_offset.addr, self.sample_offset.addr, ">>", 4)
        # treg_offset = treg_offset + treg_step
        self.math(self.treg_offset.page, self.treg_offset.addr, self.treg_offset.addr, "+", self.treg_step.addr)
        # sample_offset = sample_offset & 15
        self.mathi(self.sample_offset.page, self.sample_offset.addr, self.sample_offset.addr, "&", 15)

    def select_waveform(self, depth, center, span, label, pulse_type="pi_", phase=0):
        """
        A binary search tree to select the correct waveform for a given sample_offset
        """
        center = int(center)
        if (depth==0):
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"{pulse_type}{center}", phase=self.deg2reg(phase))
            self.condj(self.sample_offset.page, self.sample_offset.addr, "==", self.sample_offset.addr, label)
            return

        self.regwi(self.comparison.page, self.comparison.addr, center)
        self.condj(
                self.sample_offset.page,
                self.sample_offset.addr,
                ">=",
                self.comparison.addr,
                f"{pulse_type}{phase}_pulse_offset_{center}")
        
        self.select_waveform(depth-1, center-span/4, span/2, label, pulse_type, phase)

        self.label(f"{pulse_type}{phase}_pulse_offset_{center}")
        self.select_waveform(depth-1, center+span/4, span/2, label, pulse_type, phase)