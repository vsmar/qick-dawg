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

class RFTest_CPMG_XY(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "mw_freg", # Microwave freq # ~1405 MHz per Tommy
        "trigger_gate_pmod", # PMOD pin for external trigger
        
        # Tau parameters
        "tau_start_treg", # start of tau range
        "tau_end_treg", # end of tau range
        "tau_delta_treg", # step size of tau range
        "nsweep_points",

        # Gaussian Pulse Parameters
        "pulse_sigma_treg", # sigma for gaussian pulse
        "half_pi_pulse_len_treg", # pi/2 pulse length (aim for cleanest and shortest)
        "pi_pulse_len_treg", # pi pulse length (aim for cleanest and shortest)
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

        # half pi pulse waveform
        self.add_gauss(ch=self.cfg.mw_channel, 
                       name='gaussian_half_pi', 
                       sigma=self.cfg.pulse_sigma_treg, 
                       length=self.cfg.half_pi_pulse_len_treg)

        # pi pulse waveform
        self.add_gauss(ch=self.cfg.mw_channel, 
                       name='gaussian_pi', 
                       sigma=self.cfg.pulse_sigma_treg, 
                       length=self.cfg.pi_pulse_len_treg)

        # Initialize pulse register to start?
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='arb',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            waveform='gaussian_half_pi',
            phase=0)

        # Tau Register
        self.tau_register = self.new_gen_reg(self.cfg.mw_channel,
                                      name='tau', 
                                      init_val=self.cfg.tau_start_treg)
        self.add_sweep(QickSweep(self, 
            self.tau_register,
            self.cfg.tau_start_treg, 
            self.cfg.tau_end_treg,
            self.cfg.nsweep_points))

        self.synci(100)  # give processor some time to configure pulses

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

        # pi/2 pulse
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
        # Delay before pi pulses
        self.sync(self.tau_register.page, self.tau_register.addr)

        # Pulse MW channel
        for pulse in range(self.cfg.n_pulses):
            # X pulse
            self.set_pulse_registers(
                ch=self.cfg.mw_channel,
                style='arb',
                freq=self.cfg.mw_freg,
                gain=self.cfg.mw_gain,
                waveform='gaussian_pi',
                phase=self.deg2reg(0))
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all()
            # delay
            self.sync(self.tau_register.page, self.tau_register.addr)
            # Y pulse
            self.set_pulse_registers(
                ch=self.cfg.mw_channel,
                style='arb',
                freq=self.cfg.mw_freg,
                gain=self.cfg.mw_gain,
                waveform='gaussian_pi',
                phase=self.deg2reg(90))
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all()
            # delay
            self.sync(self.tau_register.page, self.tau_register.addr)

        # Final pi/2 pulse
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='arb',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            waveform='gaussian_half_pi',
            phase=0)

        self.pulse(ch=self.cfg.mw_channel)

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)