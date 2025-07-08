'''
RFTest Spectrum
=======================================================================
RFTest Spectrum class used to program continuous RF output to evaluate the RFSoC RF Power.
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

class RFTestSpectrum(NVAveragerProgram):
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
        "mw_gain",
        "nsweep_points",
        "cycles_start",
        "cycles_end",
        "nsweep_points",
        "pre_init"]
    
    def initialize(self):
        # NVConfiguration class does not have Gain units unlike freq, time, or phase
        # need to call: cfg.add_unitless_linear_sweep(gain, start, stop, delta, nsweep_points)
        self.check_cfg() #?

        if self.cfg.cycles_end <= self.cfg.cycles_start:
            assert 0, 'trigger_width_end_treg must be greater than trigger_width_start_treg'
        if self.cfg.mw_gain > 30000:
            assert 0, 'Gain limit is 30000'

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.setup_readout()

        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.pulse_len_treg, # pulse len
            phase=0)

        self.cycles_register = self.new_gen_reg(self.cfg.mw_channel, name='cycles', 
                                               init_val=self.cfg.cycles_start)
        
        # Qick initializes [name]_start_[value] using NVConfiguration.add_linear_sweep
        # Which handles fGHz, fMHz, freg, tus, tns, pdegrees, preg (frequency, time, phase)
        self.add_sweep(QickSweep(self,
                          self.cycles_register,
                          self.cfg.cycles_start,
                          self.cfg.cycles_end,
                          self.cfg.nsweep_points)) # gain has to be an int, is this accounted for later in generation??
        
        self.synci(100)  # give processor some time to configure pulses

        if self.cfg.pre_init: # not sure the purpose of this
            pass # I don't think I need this block to test

    def body(self):
        # Pulse MW channel
        self.pulse(ch=self.cfg.mw_channel, t=0)

        self.trigger(
                    pins=[self.cfg.laser_gate_pmod],
                    width=self.cfg.trigger_width_treg,              # 500 ns pulse ????
                    t=0                     # start immediately
                    )

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)