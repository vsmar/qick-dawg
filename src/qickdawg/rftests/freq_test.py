'''
Frequency Test
=======================================================================
Frequency Test class used to program sequences to evaluate the RFSoC RF Power 
and Rise times.
'''


from qick.averager_program import QickSweep
from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import floor
import os 

class FreqTest(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "pulse_len_treg",
        "trigger_gate_pmod",
        "adc_channel", #not used
        "relax_delay_treg",
        "trigger_width_treg",
        "mw_channel",
        "mw_nqz",
        "gain_start",
        "gain_end",
        "nsweep_points",
        "n_repetitions"]
    
    def initialize(self):
        self.check_cfg()

        if self.cfg.gain_start < 0:
            raise ValueError("Smallest Microwave gain must be positive")
        #elif self.cfg.gain_end > 32767:
        #    raise ValueError("Largest Microwave gain exceeds maximum value")

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.gain_start,
            length=self.cfg.pulse_len_treg, # pulse len
            phase=0)

        self.mw_gain_register = self.get_gen_reg(self.cfg.mw_channel, "gain")
        
        # NVConfiguration class does not have Gain units unlike freq, time, or phase
        # need to call: cfg.add_unitless_linear_sweep(gain, start, stop, delta, nsweep_points)

        # Qick initializes [name]_start_[value] using NVConfiguration.add_linear_sweep
        # Which handles fGHz, fMHz, freg, tus, tns, pdegrees, preg (frequency, time, phase)
        self.add_sweep(QickSweep(self,
                          self.mw_gain_register,
                          self.cfg.gain_start,
                          self.cfg.gain_end,
                          self.cfg.nsweep_points))
        
        self.n_reps_register = self.new_gen_reg(
                                self.cfg.mw_channel,
                                name='nreps',
                                init_val=self.cfg.n_repetitions)
        
        self.synci(100)  # give processor some time to configure pulses

    def body(self):
        # Pulse MW channel
        self.n_reps_register.reset()

        self.trigger(
            pins=[self.cfg.trigger_gate_pmod],
            width=self.cfg.trigger_width_treg,
            t=0)

        self.label("LOOP_nreps")

        self.pulse(ch=self.cfg.mw_channel)

        self.loopnz(self.n_reps_register.page,
                    self.n_reps_register.addr,
                    'LOOP_nreps')

        self.sync_all(self.cfg.relax_delay_treg)