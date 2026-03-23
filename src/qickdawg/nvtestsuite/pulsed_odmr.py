'''
PODMR
=======================================================================
An NVAveragerProgram class to run the Pulsed-Optically Detected Magnetic Resonance experiment,
ie. a MW pulse sequence pi - Read out.

Author: Victor Marcenac
'''
from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from ..nvpulsing.nvqicksweep import NVQickSweep
from ..util import apply_on_axis_0_n_times


import numpy as np
from itemattribute import ItemAttribute
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class PODMR(NVAveragerProgram):
    '''
    TODO:
    An NVAveragerProgram class that generates and executes a sequence used
    to measure Pulsed-ODMR

    Parameters
    ----------
    soccfg : `qick.QickConfig`
    cfg : `.NVConfiguration`
        instance of `.NVConfiguration` class with attributes:
        .adc_channel : int
            ADC channel for gathering data, usually 0 or 1
        .mw_channel : int
            qick channel that provides microwave excitation
            0 or 1 for RFSoC4x2
            0 to 6 for ZCU111 or ZCU216
        .mw_nqz : int
            nyquist zone for microwave generator (1 or 2)
        .mw_gain : int
            gain of micrwave channel, in register values, from 0 to 2**15-1

    '''
    required_cfg = ["adc_channel",
                    "readout_integration_treg",
                    "mw_channel",
                    "mw_nqz", # MW Nyquist Zone (1 or 2)
                    "mw_gain", # MW Gain (12000)
                    "mw_pi_treg",
                    "mw_start_freg", # SWEEP PARAM: start freq
                    "mw_end_freg", # SWEEP PARAM: end freq
                    "nsweep_points", # SWEEP PARAM: num of freqs
                    "pre_init",
                    "laser_gate_pmod",
                    "laser_on_treg",
                    "laser_readout_offset_treg",
                    "relax_delay_treg",
                    "reps",
                    "readout_reference_start_treg",
                    "mw_laser_delay_treg", # 0 should be ok
                    ]

    def initialize(self):
        '''
        TODO
        '''
        self.check_cfg()

        self.setup_readout()

        # configure pulse defaults and initial parameters for microwave
        self.declare_gen(
            ch=self.cfg.mw_channel,
            nqz=self.cfg.mw_nqz)        

        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            length=self.cfg.mw_pi_treg,
            gain=self.cfg.mw_gain,
            phase=0)

        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                 freq=self.cfg.mw_start_freg)

        # configure the sweep
        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")

        # Then in initialize():
        self.add_sweep(NVQickSweep(self,
                                self.mw_frequency_register,
                                self.cfg.mw_start_fMHz,  # Use fMHz like LockinODMR
                                self.cfg.mw_end_fMHz,
                                self.cfg.nsweep_points))
        
        self.synci(100)  # give processor some time to configure pulses
        if (self.cfg.ddr4 is True) or (self.cfg.mr is True):
            self.trigger(ddr4=self.cfg.ddr4, mr=self.cfg.mr, adc_trig_offset=0)
        self.synci(100)

        if self.cfg.pre_init:
            self.trigger(
                pins=[self.cfg.laser_gate_pmod],
                width=self.cfg.laser_on_treg, 
                adc_trig_offset=0)
            self.sync_all(self.cfg.laser_on_treg)

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)

    def body(self):
        '''
        TODO:
        Initially lets just do:
        mw_pi
        laser ON
        wait laser_readout_delay
        readout (for desired time)
        ...
        readout reference at some duration after


        IMPORTANT: This means that readout_reference_start_treg > laser_readout_offset_treg + readout_integration_treg, 
        which is risky to put on the user if we expect them to optimize the laser on duration

        Good rule of thumb put it maybe 0.5-1 us after

        ADDITIONALLY: relax_delay_treg > laser_readout_offset_treg, as relax_delay_treg is executed at the ebd of ttl_readout,
        and ttl_readout doesn't otherwise account for the delay in toggling the laser (maybe it assumes it's strictly DAC lag)
        '''
        # frequency is updated in the external NVAveragerProgram sweep loop

        # pi
        self.pulse(ch=self.cfg.mw_channel, t=0)
        self.synci(self.cfg.mw_pi_treg) # if laser on time is longer than MW time this might not be necessary
        self.synci(self.cfg.mw_laser_delay_treg)
        # laser on
        self.ttl_readout()
        
    def acquire(self, raw_data=False, *arg, **kwarg):
        data = super().acquire(readouts_per_experiment=2, *arg, **kwarg)

        if raw_data is False:
            data = self.analyze_results(data)

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
        d.signal = data[..., 0]
        d.reference = data[..., 1]
        
        # Average over all axes except the last (frequency) axis
        n = len(d.signal.shape) - 1
        
        for key in ['signal', 'reference']:
            d[key] = apply_on_axis_0_n_times(d[key], np.sum, n)
            d[key] = d[key] / (self.cfg.readout_integration_tns * 1e-9 * self.cfg.reps)

                
        # Calculate signal/reference
        d.contrast = d.signal / d.reference
        
        # Add frequency axis
        d.frequencies = self.qick_sweeps[0].get_sweep_pts()
        
        return d