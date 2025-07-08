'''
RFTest
=======================================================================
RF Test class used to program sequences to evaluate the RFSoC RF Power 
and Rise times.
'''


from qick.averager_program import QickSweep
from .nvaverageprogram import NVAveragerProgram
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import floor
import os 

class RFTest(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "pulse_len_treg",
        "laser_gate_pmod",
        "adc_channel", #not used
        "relax_delay_treg",
        "trigger_width_treg",
        "mw_channel",
        "mw_nqz",
        "gain_start",
        "gain_end",
        "nsweep_points",
        "pre_init",
        "reps",
        "repitition"]
    
    def initialize(self):
        # NVConfiguration class does not have Gain units unlike freq, time, or phase
        # need to call: cfg.add_unitless_linear_sweep(gain, start, stop, delta, nsweep_points)
        self.check_cfg() #?

        if self.cfg.gain_start < 0:
            assert 0, 'Smallest Microwave gain must be postive'
        elif self.cfg.gain_end > 32767: # 30000 in lockinodmr
            assert 0, 'Largest Microwave gain exceeds maximum value'

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.setup_readout()

        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.gain_start,
            length=self.cfg.pulse_len_treg, # pulse len
            phase=0)

        self.mw_gain_register = self.get_gen_reg(self.cfg.mw_channel, "gain")
        
        # Qick initializes [name]_start_[value] using NVConfiguration.add_linear_sweep
        # Which handles fGHz, fMHz, freg, tus, tns, pdegrees, preg (frequency, time, phase)
        self.add_sweep(QickSweep(self,
                          self.mw_gain_register,
                          self.cfg.gain_start,
                          self.cfg.gain_end,
                          self.cfg.nsweep_points)) # gain has to be an int, is this accounted for later in generation??
        
        self.synci(100)  # give processor some time to configure pulses

        if self.cfg.pre_init: # not sure the purpose of this
            pass # I don't think I need this block to test

    def body(self):
        # Pulse MW channel
        for rep in range(self.cfg.repitition):
            self.pulse(ch=self.cfg.mw_channel, t=rep*self.cfg.pulse_len_treg)
        
        self.trigger(
                    pins=[self.cfg.laser_gate_pmod],
                    width=self.cfg.trigger_width_treg,              # 500 ns pulse
                    t=0                     # start immediately
                    )

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)