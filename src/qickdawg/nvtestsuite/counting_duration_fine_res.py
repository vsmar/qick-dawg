'''
Counting Duration
=======================================================================
Min resolution of 200ps for steps between pulses in Rabi sequence
using fine control of waveform start address and phase.

Modified from Tommy's RabiFineRes program. See his notebook for more details on the method.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from .readout_helpers import ReadoutHelpers
import numpy as np

class CountingDurationFineRes(NVAveragerProgram, ReadoutHelpers):
    '''
    Rabi sub-nanosecond resolution pulsing program
    '''
    required_cfg = [
        "mw_duration_tdds", # length of mw
        "freq_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "laser_gate_pmod", # should be 0 for PMOD0_0
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq

        # Readout and delays
        "adc_channel",
        "laser_init_treg",
        "readout_integration_treg",
        "mw_to_laser_delay_treg", # Positive 
        "laser_readout_offset_treg",
        "get_reference",  # Whether to acquire a reference readout with MW gain = 0
    ]

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Setup readout validation and gain register
        self.setup_readout_registers(self.cfg.mw_channel)

        # Get samps per clk for later calculations. should be 16 for mw with current version rfsoc 11/14/2025
        # if this changes from 16 then need to change waveform generation part
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        # Configure the waveforms for different fine resolution pulse steps
        # Waveforms must have at least a length of 3 treg units 
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_duration_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk  # in tdds units
        
        i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        i_data[:self.cfg.mw_duration_tdds] = 1
        q_data[:self.cfg.mw_duration_tdds] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=i_data, qdata=q_data)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="pulse",
                                     phase = 0)
                
        # Setup laser
        self.setup_readout()

        # start from a close to initialized state
        if self.cfg.pre_init:
            self.pre_init()
        
        self.synci(200)  # Give tproc time to get ahead

    def body(self):
        # Initialize
        self.laser_init()

        # MW Pulse
        self.program_pulse()

        # Readout with optional reference
        self.signal_and_reference_readout(self.program_pulse)

    def program_pulse(self):
        """Program the MW pulse sequence"""
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

