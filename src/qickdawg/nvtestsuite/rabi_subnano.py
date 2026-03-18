'''
200 ps resolution Rabi
=======================================================================
Generates and executes a measurement of a sweep of microwave pulse lengths
with up to 200 ps resolution to observe Rabi oscillations.

This implementation is derived from the flattop style pulse implementation, 
playing a product pulse for fine adjustments and a dds pulse for course adjustments.
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

class RABI_SUBNANO(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        "adc_channel",
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_freg", # MW FREQ (~1405 MHz for QDP)
        "mw_gain", #MW Gain
        "reps", # repetitions

        # MW length Sweep Parameters ### Need to add sample compatability and evaluate as non time
        "mw_start_tsamp",
        "mw_end_tsamp",
        "nsweep_points",

        # Laser params
        "laser_gate_pmod",
        "laser_init_treg",

        # Readout and delays
        "readout_integration_treg",
        "mw_to_laser_delay_treg", # Positive 
        "laser_readout_offset_treg",
        ]

    def initialize(self):
        """
        Sets up the waveforms, registers and sweeps, for pi and pi/2 pulses, and delays. 
        Also sets up the pulse phase sequence used. Currently only implemented with 2 phases, could be expanded.
        """
        self.check_cfg()

        if self.cfg.mw_gain < 0:
            assert 0, 'Smallest Microwave gain must be postive'
        elif self.cfg.mw_gain > 32767: # 2**15 - 1
            assert 0, 'Largest Microwave gain exceeds maximum value'
        elif self.cfg.mw_start_tsamp < 1:
            assert 0, 'Smallest possible tsamp value is 1 sample (200ps)'

        # Readout for QICK-DAWG
        # self.setup_readout()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        

        # CONSTANTS:
        # Get samples per clock (16 with current version of QICK-DAWG)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        self.tuning_num = int(np.log2(self.samps_per_clk)) # need a better way to describe the relationship of 4 to 16 in the context of resolving all options in binary
        self.wvfm_length_treg = 3


        # Generate (fine-stepping) waveform memories                 
        for i in range(1, self.samps_per_clk+1):
            waveform = np.zeros(self.wvfm_length_treg * self.samps_per_clk)
            waveform[-i:] = 1
            waveform *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"fine_adjustment_{i}", idata=waveform, qdata=waveform)

        # Coarse duration register
        self.coarse_pulse_duration = self.new_gen_reg(self.cfg.mw_channel,
                                                name='coarse_pulse_duration',
                                                init_val=0)
        
        # Initialize pulse register defaults
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="fine_adjustment_16",
                                     phase=0)
        
        # Setup Tau Sweep (tsamp units, ie 200ps)
        self.pulse_length_register = self.new_gen_reg(self.cfg.mw_channel, "mw_tsamps", init_val=self.cfg.mw_start_tsamp)
        self.add_sweep(QickSweep(self, self.pulse_length_register, self.cfg.mw_start_tsamp - 1, self.cfg.mw_end_tsamp - 1, self.cfg.nsweep_points))
        
        # waveform address register
        self.wvfm_addr_register = self.get_gen_reg(self.cfg.mw_channel, name='addr')
        
        # Setup laser
        self.setup_readout()

        # start from a close to initialized state
        if self.cfg.pre_init:
            self.trigger( # Laser
                pins=[self.cfg.laser_gate_pmod],
                adc_trig_offset=0,
                width=self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg,
                t=0)
            self.wait_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg)
            self.sync_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg + 200)
        else:
            self.sync_all(200)  # give processor some time to configure pulses
        
        # give processor some time to configure pulses
        self.synci(200)

    def body(self):
        """
        Produces a RABI ODMR Sequence:
        - Laser init
        - Delay (matches MW length)
        - Laser Readout
        - Laser init
        - MW 
        - Laser Readout
        """
        self.laser_init()

        self.program_pulses(rf_on=True)

        self.readout()
        
        self.laser_init()

        self.sync_all()

        self.program_pulses(rf_on=False)

        self.readout()


    def program_pulses(self, rf_on=True):
        if rf_on:
            # MW PULSES
            # Fine delay adjustment and assignment
            self.bitwi(self.wvfm_addr_register.page, self.wvfm_addr_register.addr,
                    self.pulse_length_register.addr, "&", (self.samps_per_clk - 1))
            self.wvfm_addr_register.set_to(self.wvfm_addr_register, '*', self.wvfm_length_treg, physical_unit = False)
            # Coarse adjustment
            self.bitwi(self.coarse_pulse_duration.page, self.coarse_pulse_duration.addr,
                    self.pulse_length_register.addr, '>>', self.tuning_num)

            # PLAY RF PULSE:
            # Play waveform (start pulse)
            self.set_pulse_registers(self.cfg.mw_channel, outsel = "product", stdysel = "last")
            self.pulse(ch=self.cfg.mw_channel, t=0)
            # Coarse adjustment delay
            self.sync(self.coarse_pulse_duration.page, self.coarse_pulse_duration.addr)
            # Play 0 pulse (turn off pulse)
            self.set_pulse_registers(self.cfg.mw_channel, outsel = "zero")
            self.pulse(ch=self.cfg.mw_channel)
        else:
            self.synci(self.wvfm_length_treg * 2)
            self.sync(self.coarse_pulse_duration.page, self.coarse_pulse_duration.addr)
        
        self.sync_all()


    def laser_init(self):
        self.trigger(pins = [self.cfg.laser_gate_pmod], width = self.cfg.laser_init_treg)
        self.wait_all(self.cfg.laser_init_treg)
        self.sync_all(self.cfg.laser_init_treg + self.cfg.mw_to_laser_delay_treg)
    
    def readout(self):
        # RO
        self.trigger_no_off( # Laser
            pins=[self.cfg.laser_gate_pmod],
            t=0)
        self.trigger( # Laser + ADC
            adcs=self.cfg.adcs,
            pins=[self.cfg.laser_gate_pmod],
            adc_trig_offset=0,
            width=self.cfg.readout_integration_treg,
            t=self.cfg.laser_readout_offset_treg)
        self.wait_all(self.cfg.readout_integration_treg)
        self.sync_all(self.cfg.readout_integration_treg +self.cfg.pulse_seq_delay_treg)
        

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