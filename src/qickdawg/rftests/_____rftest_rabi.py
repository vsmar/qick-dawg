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

class SUBNANO_RABI(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
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
        "laser_on_treg",
        "readout_reference_start_treg",

        # delays
        "mw_laser_delay_treg", # MW typically triggers sooner than laser
        "laser_readout_offset_tus",
        "relax_delay_treg"
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
        self.wvfm_length_treg = 16


        # Generate (fine-stepping) waveform memories                 
        for i in range(1, self.samps_per_clk+1):
            waveform = np.zeros(self.wvfm_length_treg * self.samps_per_clk)
            waveform[-i*10:] = 1
            waveform *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"fine_adjustment_{i}", idata=waveform, qdata=waveform)

        # TODO: resolve discrepancy with 200 ps pulse start 

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
        
        # Setup laser parameters for readout
        # self.declare_readout(ch=self.cfg.laser_gate_pmod, length=self.cfg.laser_on_treg)

        # give processor some time to configure pulses
        self.synci(200)

    def body(self):
        """
        The pulse & delay instructions, and the selection computations
        """
        self.trigger(
                    pins=[self.cfg.laser_gate_pmod],
                    width=self.cfg.laser_on_treg,              # 500 ns pulse ????
                    t=0                     # start immediately
                    )

        # TODO: Add delay between initialization - 
        # if mw_laser_delay is > 0:
        # play laser pulse and make sure it ends, mw_laser_delay before playing RF pulse
        if self.cfg.mw_laser_delay_treg > 0:
            # play laser init pulse
            self.synci(self.cfg.mw_laser_delay_treg) # this should wait for x duration after pulse is done (syn_all needed?)
            self.sync_all() # needed?


        # COMPUTE RF PULSE TIMING:
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
        self.pulse(ch=self.cfg.mw_channel)
        # Coarse adjustment delay
        self.sync(self.coarse_pulse_duration.page, self.coarse_pulse_duration.addr)
        # Play 0 pulse (turn off pulse)
        self.set_pulse_registers(self.cfg.mw_channel, outsel = "zero")
        self.pulse(ch=self.cfg.mw_channel)



        self.sync_all()

        # Add delay before readout for signal to settle
        # might want to make this command timing based on the delay necessary.

        # READOUT - Turn on laser for detection
        # self.set_readout_registers(ch=self.cfg.laser_gate_pmod, length=self.cfg.laser_on_treg, freq=0, gain=32767)
        # self.pulse(ch=self.cfg.laser_gate_pmod)
    
        
        # Perform the actual readout acquisition
        # self.ttl_readout()
