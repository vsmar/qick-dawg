'''
Subnanosecond Ramsey
=======================================================================
Subnanosecond Ramsey class used to test the shape of RF envelopes.
Implements full experiment with laser initialization and readout.
'''

from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from ..nvpulsing.nvqicksweep import NVQickSweep
from .readout_helpers import ReadoutHelpers

import numpy as np
from math import floor


class RamseyFineRes(ReadoutHelpers, NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        # Channels and pmods
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod", # should be 0 for PMOD0_0

        # MW pulse parameters
        "mw_pi2_ftsamp", # length of pi/2 pulse
        "mw_nqz", # 1 < 2.495 GHz
        "mw_freg",
        "mw_gain",

        # Delay Sweep parameters
        "tau_start_ftsamp",
        "tau_end_ftsamp",
        "nsweep_points",
        "scaling_mode",
        "scaling_factor", # Only used if scaling_mode is 'exponential'

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
        self.setup_helper_registers(self.cfg.mw_channel)

        # Setup laser
        self.setup_readout()

        # Get samples per clock (16 with current version of QICK-DAWG)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # ---------------------
        # Waveform Set-up
        # ---------------------
        # Waveforms must have at least a length of 3 treg
        self.half_pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.mw_pi2_ftsamp + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)
        
        # Generate waveforms with each offset
        for i in range(self.samps_per_clk):
            # half pi pulses
            data = np.zeros(self.half_pi_waveform_len_treg * 16)
            data[i : i + self.cfg.mw_pi2_ftsamp] = 1
            data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=data, qdata=data)

        # Amount of delay in the waveforms
        self.half_pi_len_unused = self.half_pi_waveform_len_treg*16 - self.cfg.mw_pi2_ftsamp
        
        # Register to hold the FPGA/tproc delay between pulses
        self.treg_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)

        # Initialize ftsamp_offset_register
        self.ftsamp_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='ftsamp_offset',
                                                    init_val=0)
        
        # Initialize pulse register defaults
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain)
        
        # Setup Tau Sweep (ftsamp units, ie 200ps)
        self.tau_samples = self.new_gen_reg(self.cfg.mw_channel, "tau_samples")

        if self.cfg.scaling_mode == 'exponential':
            self.add_sweep(NVQickSweep(
                self,
                self.tau_samples,
                self.cfg.tau_start_ftsamp,
                self.cfg.tau_end_ftsamp,
                expts=self.cfg.nsweep_points,
                scaling_mode=self.cfg.scaling_mode,
                scaling_factor=self.cfg.scaling_factor))

        elif self.cfg.scaling_mode == 'linear':
            self.add_sweep(NVQickSweep(
                self,
                self.tau_samples,
                self.cfg.tau_start_ftsamp,
                self.cfg.tau_end_ftsamp,
                self.cfg.nsweep_points))
        else:
            assert 0, 'cfg.scaling_mode must be "linear" or "exponential"'
        
        # waveform address register
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr')
        
        # Pre-initialization
        self.pre_init()
        
    def body(self):
        """
        The full pulse sequence: laser init -> RF pulses -> readout
        """
        self.laser_init() # misnomer this just delays for mw_to_laser_delay_treg
        self.program_pulses(1)
        self.signal_and_reference_readout(lambda: self.program_pulses(2))

    def program_pulses(self, iter):
        # Pi/2 X pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # pi/2 X pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        self.set_waveform()
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Reset for next loop
        self.ftsamp_offset_register.reset() # reset to initial value 

    def set_waveform(self):
        """
        Configures the assembly code necessary for setting the waveform
        This also precedes the pulse with the delay that matches it.
        2tau delay before pi pulse
        tau delay before pi/2 pulse
        """
        # ----------------------------
        # Waveform Selection & Offsets
        # ----------------------------
        # Compute offset
        self.offset_computations()
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr) 

        # Update the address register to correspond to the correct waveform
        self.address_register.set_to(self.ftsamp_offset_register, '*', 
                                     self.half_pi_waveform_len_treg, physical_unit = False)

    def offset_computations(self):
        """
        Compute the ftsamp_offset_register and treg_offset_register for the next pulse. 
        Computes: 
        1. wait till the start of the next pulse (from the end of the previous waveform) (in sample timing resolution 200ps)
        2. Amount of wait done on the FPGA (converting to treg, ie dividing by 16 and taking the floor)
        3. Amount of wait done in waveform (samples),   equal to: (previous offset + the step in samples) % 16

        Tau_samples is a misnomer, instead it is:
            Actual Tau (samples) - Pi Pulse Waveform unused (samples)
            $$$ This is done to account for the delay in the pi pulse waveform itself
        """
        # ftsamp_offset_register = ftsamp_offset_register + sample_step
        # Computes the total delay needed until the next pulse from the end of this waveform
        # by adding amount of samples to wait + current sample offset
        self.math(self.ftsamp_offset_register.page, self.ftsamp_offset_register.addr, 
                  self.ftsamp_offset_register.addr, "+", self.tau_samples.addr)
        self.mathi(self.ftsamp_offset_register.page, self.ftsamp_offset_register.addr, 
                    self.ftsamp_offset_register.addr, "-", self.half_pi_len_unused)


        # treg_offset_register = ftsamp_offset_register >> 4 (global offset)
        # Computes how long to stall the FPGA output in tproc cycles
        # from the total delay.
        # This operation also converts from samples (200ps) to treg (3.2ns)
        self.bitwi(self.ftsamp_offset_register.page, self.treg_offset_register.addr, 
                   self.ftsamp_offset_register.addr, ">>", int(np.log2(self.samps_per_clk)))
        # ftsamp_offset_register = ftsamp_offset_register & 15
        # Computes the remaining samples that the pulse should be delayed by
        # This is equivalent to: total delay (in samples) - fpga delay (in samples)
        self.bitwi(self.ftsamp_offset_register.page, self.ftsamp_offset_register.addr, 
                   self.ftsamp_offset_register.addr, "&", (self.samps_per_clk-1))
        
    def acquire(self, raw_data=False, *arg, **kwarg):
        # self.acquire --> ReadoutHelpers.acquire --> NVAveragerProgram.acquire
        data = super().acquire(raw_data=raw_data, sweep_param='tau_ftsamp', *arg, **kwarg)
        return data
