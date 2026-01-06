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

# To follow timing conventions, I've decided to start using
# _tdds to refer to a samples timing resolution in a waveform/dds

class ENVELOPE_PHASE_CHECK(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_freg", # MW FREQ (~1405 MHz for QDP)
        "mw_gain", #MW Gain
        "reps", # repetitions
        
        # Pulse Parameters
        "pi2_len_samples", # length of pi/2 pulse

        # Temporary parameters for development
        "trigger_gate_pmod", # PMOD pin for external trigger
        "trigger_width_treg",
        "relax_delay_treg"]
    
    def initialize(self):
        """
        Sets up the waveforms, registers and sweeps, for pi and pi/2 pulses, and delays. 
        Also sets up the pulse phase sequence used. Currently only implemented with 2 phases, could be expanded.
        """
        # ---------------------
        # General Config
        # ---------------------
        self.check_cfg()

        if self.cfg.mw_gain < 0:
            assert 0, 'Smallest Microwave gain must be postive'
        elif self.cfg.mw_gain > 32767: # 2**15 - 1
            assert 0, 'Largest Microwave gain exceeds maximum value'

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Get samples per clock (16 with current version of QICK-DAWG)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Readout for QICK-DAWG
        self.setup_readout()

        # ---------------------
        # Waveform Set-up
        # ---------------------
        # Waveforms must have at least a length of 3 treg
        self.pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.pi2_len_samples*2 + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)
        self.half_pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.pi2_len_samples + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)
        
        # 0 degree phase shift
        data = np.zeros(self.pi_waveform_len_treg * 16)
        data[:self.cfg.pi2_len_samples*2] = 1
        data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_+q+i", idata=data, qdata=data)

        # 90
        data = np.zeros(self.pi_waveform_len_treg * 16)
        data[:self.cfg.pi2_len_samples*2] = 1
        data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_-q+i", idata=data, qdata=-data)

        # Initialize pulse register defaults
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain)
        
        # waveform address register
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr')
        
        # pulse phase register (NCO phase)
        self.phase_register = self.get_gen_reg(self.cfg.mw_channel, name='phase')
        
        # give processor some time to configure pulses
        self.synci(200)

    def body(self):
        """
        The pulse & delay instructions, and the selection computations
        """

        # First pulse: 0 degrees NCO, +i, +q
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_+q+i", phase=0)
        self.sync_all()
        self.pulse(ch=self.cfg.mw_channel)

        # First pulse: 90 degrees NCO, +i, +q
        self.synci(40)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_+q+i", phase=self.deg2reg(-90))
        self.sync_all()
        self.pulse(ch=self.cfg.mw_channel)

        # First pulse: 90 degrees NCO, +i, +q
        self.synci(40)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_+q+i", phase=0)
        self.sync_all()
        self.pulse(ch=self.cfg.mw_channel)

        # First pulse: 90 degrees NCO, +i, +q
        self.synci(40)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_-q+i", phase=0)
        self.sync_all()
        self.pulse(ch=self.cfg.mw_channel)

        self.synci(200)
