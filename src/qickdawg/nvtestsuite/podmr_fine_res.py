'''
ODMR Fine Resolution
=======================================================================
Combines frequency sweep of PODMR with fine resolution pulse control of CountingDurationFineRes.
Uses single readout with shared readout helpers.

Author: Victor Marcenac
'''

from itemattribute import ItemAttribute
from qickdawg.util.apply_on_axis_0_n_times import apply_on_axis_0_n_times

from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from .readout_helpers import ReadoutHelpers
from qick.averager_program import QickSweep
import numpy as np


class PODMRFineRes(ReadoutHelpers, NVAveragerProgram):
    '''
    Pulsed-ODMR with fine resolution pulse control.
    Sweeps microwave frequency while applying MW pulses with fine resolution control.
    '''
    required_cfg = [
        # Channels and pmods
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod", # should be 0 for PMOD0_0

        # MW pulse parameters
        "mw_pi_tdds",
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain

        # Sweep parameters
        "mw_start_fMHz",
        "mw_end_fMHz",
        "nsweep_points",

        # Readout and delays
        "mw_to_laser_delay_treg", # How long do we need to delay the mw, to ensure the laser pulse is correctly timed before it
        "relax_delay_treg", # delay between laser and next MW pulse

        # Readout
        "laser_on_treg",
        "readout_reference_start_treg",
        "readout_integration_treg",
        "laser_readout_offset_treg",

        # Other
        "reps",
        "pre_init",
        "get_reference",  # Whether to acquire a reference readout with MW gain = 0  
    ]

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.setup_helper_registers(self.cfg.mw_channel)

        # Setup laser
        self.setup_readout()
        
        # Get samps per clk for later calculations
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        # Configure the waveforms for fine resolution pulse steps (must be >= 3 treg units)
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pi_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk
        # Create waveform with exact duration
        i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        i_data[:self.cfg.mw_pi_tdds] = 1
        q_data[:self.cfg.mw_pi_tdds] = 1
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
        # Explicitly arm the pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel)
        
        # Setup frequency sweep
        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")
        self.add_sweep(QickSweep(self,
                                self.mw_frequency_register,
                                self.cfg.mw_start_fMHz,
                                self.cfg.mw_end_fMHz,
                                self.cfg.nsweep_points))
        
        # start from a close to initialized state
        self.pre_init()

    def body(self):
        self.laser_init()
        self.program_pulse()
        self.signal_and_reference_readout(self.program_pulse)

    def program_pulse(self):
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
    
    def acquire(self, raw_data=False, *arg, **kwarg):
        # self.acquire --> ReadoutHelpers.acquire --> NVAveragerProgram.acquire
        data = super().acquire(raw_data=raw_data, sweep_param='mw_fMHz', *arg, **kwarg)
        return data

    def analyze_results(self, data):
        """
        Method that takes in a 1D array of data points from self.acquire() and analyzes the
        results based on the number of reps and frequency points

        Parameters
        ----------
        data
            (1D np.array) data returned from self.acquire()

        returns
            (qickdawg.ItemAttribute instance) with attributes
            .frequencies (len(nsweep_points) np array, MHz units) - frequencies swept over
            .signal (nfrequency np.array, adc units)
                - average adc signal with MW pulse
            .reference (nfrequency np.array, adc units)
                - signal at the end of the reinitialization pulse
            .contrast (nfrequency np.array, fractional units)
                - (signal - reference) / reference
        """
        data = np.reshape(data, self.data_shape)
        
        d = ItemAttribute()

        d.signal1 = data[..., 0]
        d.reference1 = data[..., 1]
        d.signal2 = data[..., 2]
        d.reference2 = data[..., 3]

        # Average over all axes except the last (frequency) axis
        n = len(d.signal1.shape) - 1
        
        for key in ['signal1', 'reference1', 'signal2', 'reference2']:
            d[key] = apply_on_axis_0_n_times(d[key], np.sum, n)
            d[key] = d[key] / (self.cfg.readout_integration_tns * 1e-9 * self.cfg.reps)

        # Calculate signal/reference
        d.contrast = d.signal1 / d.signal2
        
        # Add frequency axis
        d.frequencies = self.qick_sweeps[0].get_sweep_pts()
        
        return d