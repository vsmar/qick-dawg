'''
RFTest CPMG-XY
=======================================================================
RFTest Envelope class used to test the shape of RF envelopes.
'''

# Use a QickSweep, not sure which one to use
from qick.averager_program import QickSweep
from .nvqicksweep import NVQickSweep # NVQickSweep requires a readout_integration time parameter

from .nvaverageprogram import NVAveragerProgram
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import floor
import os 

class RFTest_Phase(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "mw_freg", # Microwave freq # ~1405 MHz per Tommy
        
        # Gaussian Pulse Parameters
        "pulse_delay_treg",
        "pulse_len_treg",
        "n_pulses", # number of pulses

        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", #MW Gain
        "reps",

        # Temporary parameters for development
        "pre_init",
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

        # Initialize pulse register to start?
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.pulse_len_treg,
            phase=0)
                
        self.n_cpmg_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ncpmg',
            init_val=self.cfg.n_pulses - 1)

        # self.add_sweep(QickSweep(self, 
        #     self.tau_register,
        #     self.cfg.tau_start_treg, 
        #     self.cfg.tau_end_treg,
        #     self.cfg.nsweep_points))
        
        self.synci(580)  # give processor some time to configure pulses

        if self.cfg.pre_init: # not sure the purpose of this
            pass # I don't think I need this block to test

    def body(self):
        # pi pulse width (aim for cleanest and shortest) - pi/2 should be executable
        # RF power ~17 dBm
        # RF frequency: 1405 MHz
        # Phase of X and Y are 90 degrees offset
        # Tau range from 100ns to 1ms (steps of 10ns or less)
        # Pulse shape: rect, hermite or gaussian
        # N = 128 or 256 pi pulses

        # Single CPMG_XY8 unit
        self.n_cpmg_register.reset()
        self.label("LOOP_ncpmg")
        for project_phase in np.array(range(0, 360, 10)): # increase by 10
            # X/Y Pulse
            self.set_pulse_registers(
                ch=self.cfg.mw_channel,
                style='const',
                freq=self.cfg.mw_freg,
                gain=self.cfg.mw_gain,
                length=self.cfg.pulse_len_treg,
                phase=self.deg2reg(project_phase)
            )
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all(self.cfg.pulse_delay_treg)
        self.loopnz(self.n_cpmg_register.page,
                    self.n_cpmg_register.addr,
                    'LOOP_ncpmg')
       
        self.sync_all(self.cfg.relax_delay_treg)
        self.wait_all()
