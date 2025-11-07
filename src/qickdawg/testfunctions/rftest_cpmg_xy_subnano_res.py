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
        self.tau_samples = self.new_gen_reg(self.cfg.mw_channel,
                                            name='tau_step',
                                            init_val=self.cfg.tau_len_samples - self.pi_len_unused)
        
        # we can initialize sample_offset to already account for the first half_pi_pulse
        self.sample_offset = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='sample_offset',
                                                    init_val=(self.pi_len_unused-self.half_pi_len_unused))
        self.treg_offset = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)
        
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

        # If sweeping Tau then perform the computation:
        # samples_to_next_pulse = Actual Tau (samples) - Pi Pulse Waveform unused (samples)
        
        self.sync_all()

        # Half pi pulse, 0 sample offset, phase = x
        self.offset_computations()
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.treg_offset.page, self.treg_offset.addr)

        # Loop pi-X, tau, pi-Y, tau
        self.n_cpmg_register.reset()
        self.label("LOOP_ncpmg") # loop back to here
        
        # X pulse
        # Configures assembly code for picking the waveform
        self.set_waveform("Execute_X_Pi_Pulse", "pi_", phase=0) # Select next pulse's waveform
        self.offset_computations() # compute timing for the pulse following this one
        self.pulse(ch=self.cfg.mw_channel) # execute pulse
        self.sync_all(self.treg_offset.page, self.treg_offset.addr) 

        # Y pulse
        self.set_waveform("Execute_Y_Pi_Pulse", "pi_", phase=90) # Select next pulse's waveform
        self.offset_computations() # compute timing for the pulse following this one
        self.pulse(ch=self.cfg.mw_channel) # execute pulse
        # Using sync here, causes timing mismatch (tproc jumps ahead when taking loopnz)
        self.sync_all(self.treg_offset.page, self.treg_offset.addr) 
        
        self.loopnz( # loop if some cpmg pulses not yet executed 
                self.n_cpmg_register.page,
                self.n_cpmg_register.addr,
                'LOOP_ncpmg')
        
        # Pi/2 X pulse
        self.set_waveform("Execute_Last_Pulse", "half_pi_")
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Reset for next loop
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        self.sample_offset.reset() # reset the sample_offset adjustment 

  
    def set_waveform(self, label, pulse_type="pi_", phase=0):
        """
        Configures the assembly code necessary for setting the waveform
        """
        self.select_waveform(4, 8, 16, label, pulse_type, phase) # Enter binary search tree
        self.label(label) # label to branch back to after selecting waveform
    
    def offset_computations(self):
        """
        Compute the sample_offset and treg_offset for the next pulse. 
        Computes: 
        1. wait till the start of the next pulse (from the end of the previous waveform) (in sample timing resolution 200ps)
        2. Amount of wait done on the FPGA (converting to treg, ie dividing by 16 and taking the floor)
        3. Amount of wait done in waveform (samples),   equal to: (previous offset + the step in samples) % 16

        Tau_samples is a misnomer, instead it is:
            Actual Tau (samples) - Pi Pulse Waveform unused (samples)
            $$$ This is done to account for the delay in the pi pulse waveform itself
        """
        # sample_offset = sample_offset + sample_step
        # Computes the total delay needed until the next pulse from the end of this waveform
        # by adding amount of samples to wait + current sample offset
        self.math(self.sample_offset.page, self.sample_offset.addr, self.sample_offset.addr, "+", self.tau_samples.addr)
        # treg_offset = sample_offset >> 4 (global offset)
        # Computes how long to stall the FPGA output in tproc cycles
        # from the total delay.
        # This operation also converts from samples (200ps) to treg (3.2ns)
        self.mathi(self.sample_offset.page, self.treg_offset.addr, self.sample_offset.addr, ">>", 4)
        # sample_offset = sample_offset & 15
        # Computes the remaining samples that the pulse should be delayed by
        # This is equivalent to: total delay (in samples) - fpga delay (in samples)
        self.mathi(self.sample_offset.page, self.sample_offset.addr, self.sample_offset.addr, "&", 15)

    def select_waveform(self, depth, center, span, label, pulse_type="pi_", phase=0):
        """
        A binary search tree to select the correct waveform for a given sample_offset
        """
        center = int(center)
        if (depth==0): # once at the lowest level of the binary tree (only one matches all conditions)
            # Set pulse register (exact waveform & phase)
            # Optionally phase could be selected for elsewhere (either via NCO, or setting up 2 waveforms)
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"{pulse_type}{center}", phase=self.deg2reg(phase))
            # Branch to exit binary tree and rejoin sequence
            self.condj(self.sample_offset.page, self.sample_offset.addr, "==", self.sample_offset.addr, label)
            return
        
        # QICK's only branch instruction is a conditional branch where you compare two registers
        # Only 16 registers per page, and only same page registers can be compared
        # We set the comparison register to an immediate (preset integer value)
        # based on what condition we want to evaluate 
        # Example: to do an if(sample_offset>=4), we would write in 4
        self.regwi(self.comparison.page, self.comparison.addr, center)
        self.condj(
                self.sample_offset.page,
                self.sample_offset.addr,
                ">=",
                self.comparison.addr,
                f"{pulse_type}{phase}_pulse_offset_{center}")
        
        # Recurse for the case where condition is not met
        self.select_waveform(depth-1, center-span/4, span/2, label, pulse_type, phase)

        # if conditional branch taken:
        self.label(f"{pulse_type}{phase}_pulse_offset_{center}") # Go here
        # Recurse for the case where condition is met
        self.select_waveform(depth-1, center+span/4, span/2, label, pulse_type, phase)