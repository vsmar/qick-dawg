'''
ODMR Fine Resolution
=======================================================================
Combines frequency sweep of PODMR with fine resolution pulse control of CountingDurationFineRes.
Uses single readout with shared readout helpers.

Author: Victor Marcenac
'''
from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from .readout_helpers import ReadoutHelpers
from qick.averager_program import QickSweep
import numpy as np


class PODMRFineRes(NVAveragerProgram, ReadoutHelpers):
    '''
    Pulsed-ODMR with fine resolution pulse control.
    Sweeps microwave frequency while applying MW pulses with fine resolution control.
    '''
    required_cfg = [
        "adc_channel",
        "readout_integration_treg",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "mw_duration_tdds",
        "mw_start_fMHz",
        "mw_end_fMHz",
        "mw_start_freg",
        "mw_end_freg",
        "nsweep_points",
        "pre_init",
        "laser_gate_pmod",
        "laser_init_treg",
        "laser_readout_offset_treg",
        "relax_delay_treg",
        "reps",
        "mw_to_laser_delay_treg",
    ]

    def initialize(self):
        self.check_cfg()
        self.check_laser_init_timing()

        self.setup_readout()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Setup helper-managed MW gain register used by reference readout path.
        self.setup_readout_registers(self.cfg.mw_channel)

        # Get samps per clk for later calculations
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        
        # Configure the waveforms for fine resolution pulse steps
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_duration_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk
        
        # Create waveform with exact duration
        i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        i_data[:self.cfg.mw_duration_tdds] = 1
        q_data[:self.cfg.mw_duration_tdds] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=i_data, qdata=q_data)
        
        # MW pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_start_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="pulse",
                                     phase=0)

        # Explicitly arm the first pulse for runtimes where defaults alone
        # do not populate next_pulse before pulse() is called.
        self.set_pulse_registers(ch=self.cfg.mw_channel)
        
        # Setup frequency sweep
        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")
        self.add_sweep(QickSweep(self,
                                self.mw_frequency_register,
                                self.cfg.mw_start_fMHz,
                                self.cfg.mw_end_fMHz,
                                self.cfg.nsweep_points))
        

        
        # start from a close to initialized state
        if self.cfg.pre_init:
            self.pre_init()
        
        self.synci(200)  # Give tproc time to get ahead
        
        self.wait_all()
        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def body(self):
        # Initialize
        self.laser_init()

        # MW Pulse (frequency is swept externally, gain controlled via register)
        self.program_pulse()
        
        # Readout with optional reference
        self.signal_and_reference_readout(self.program_pulse)

    def program_pulse(self):
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

    def analyze_results(self, data):
        """
        Analyze PODMR sweep results, renaming sweep_pts to frequencies.
        Uses parent helper's analysis for signal/reference/contrast extraction.
        
        Returns ItemAttribute with: signal, reference (opt), contrast (opt), frequencies
        """
        d = super().analyze_results(data)
        # Rename sweep_pts to frequencies for clarity
        if hasattr(d, 'sweep_pts'):
            d.frequencies = d.sweep_pts
            del d.sweep_pts
        return d
