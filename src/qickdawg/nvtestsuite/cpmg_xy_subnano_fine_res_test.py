'''
Subnanosecond CPMG-XY Test
=======================================================================
Subnanosecond CPMG-XY Test class used to test the shape of RF envelopes.
Implements full experiment with laser initialization and readout.
'''

from qick.averager_program import QickSweep
from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from .readout_helpers import ReadoutHelpers

import numpy as np
from math import floor


class SubnanoCPMGXY(NVAveragerProgram, ReadoutHelpers):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        # MW Pulse Parameters
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_freg", # MW FREQ (~1405 MHz for QDP)
        "mw_gain", # MW Gain
        "reps", # repetitions
        
        # Pulse Parameters
        "pi2_len_samples", # length of pi/2 pulse
        "n_cpmg", # number of pulses

        # Delay Sweep Parameters
        "tau_samples_start",
        "tau_samples_end",
        "nsweep_points",
        
        # Readout parameters
        "adc_channel",
        "laser_gate_pmod",
        "laser_init_treg",
        "readout_integration_treg",
        "mw_to_laser_delay_treg",
        "laser_readout_offset_treg",
        "pulse_seq_delay_treg",
        "get_reference",  # Whether to acquire a reference readout with MW gain = 0
        "pre_init"  # Whether to use pre-initialization pulse
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
        
        # Setup readout validation and gain register
        self.setup_readout_registers(self.cfg.mw_channel)

        # Readout for QICK-DAWG
        self.setup_readout()

        # Get samples per clock (16 with current version of QICK-DAWG)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # ---------------------
        # Waveform Set-up
        # ---------------------
        # Waveforms must have at least a length of 3 treg
        self.pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.pi2_len_samples*2 + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)
        self.half_pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.pi2_len_samples + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)
        
        # Generate waveforms with each offset
        for i in range(self.samps_per_clk):
            # pi pulses
            data = np.zeros(self.pi_waveform_len_treg * 16)
            data[i: i + self.cfg.pi2_len_samples*2] = 1
            data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=data, qdata=data)

            # half pi pulses
            data = np.zeros(self.half_pi_waveform_len_treg * 16)
            data[i : i + self.cfg.pi2_len_samples] = 1
            data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=data, qdata=data)

        # Amount of delay in the waveforms
        self.pi_len_unused = self.pi_waveform_len_treg*16 - self.cfg.pi2_len_samples*2
        self.half_pi_len_unused = self.half_pi_waveform_len_treg*16 - self.cfg.pi2_len_samples
        
        # Register to hold the FPGA/tproc delay between pulses
        self.treg_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)

        # Initialize tdds_offset_register to account for the first half_pi_pulse
        # (ie, difference in delay in a half pi vs a pi waveform)
        self.tdds_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='tdds_offset',
                                                    init_val=(self.pi_len_unused-self.half_pi_len_unused))
        
        # Number of pi pulses
        self.n_cpmg_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='ncpmg',
                                                    init_val=self.cfg.n_cpmg - 1)
        
        # Initialize pulse register defaults
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain)
        
        # Setup Tau Sweep (tdds units, ie 200ps)
        tau_start = self.cfg.tau_samples_start - self.pi_len_unused
        tau_end = self.cfg.tau_samples_end - self.pi_len_unused

        self.tau_samples = self.new_gen_reg(self.cfg.mw_channel, "tau_samples", init_val=tau_start)
        self.add_sweep(QickSweep(self, self.tau_samples, tau_start, tau_end, self.cfg.nsweep_points))
        
        # waveform address register
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr')

        # pulse phase register (NCO phase)
        self.phase_register = self.get_gen_reg(self.cfg.mw_channel, name='phase')

        # Setup register to store the pi-pulse phase sequence
        self.phase_to_list()

        # Initializat first half_pi_pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        
        # Pre-initialization
        if self.cfg.pre_init:
            self.pre_init()
        
        self.synci(200)  # Give tproc time to get ahead

    def phase_to_list(self):
        """
        Default: XYXYYXYX = 0b10100101
        
        # Idea:
        # Write out the pi pulse phases as a phase list (length should be a power of 2, ie CPMG2, 4, 8, 16 all supported)
        # flipping is usually not necesasary due to symmetry
        """
        self.phase_list = ["X", "Y", "X", "Y", "Y", "X", "Y", "X"]
        phase_seq = int(''.join(['0' if val == 'X' else '1' for val in np.flip(self.phase_list)]), 2)
        print(str(bin(phase_seq)))
        self.phase_seq_register = self.new_gen_reg(self.cfg.mw_channel, "phase_seq_register", init_val=phase_seq)

    def body(self):
        """
        The full pulse sequence: laser init -> RF pulses -> readout
        """
        # Initialize
        self.laser_init()
        
        # RF Pulse sequence (CPMG-XY)
        self.program_pulses()
        
        # Readout with optional reference
        self.signal_and_reference_readout(self.program_pulses)

    def program_pulses(self):
        # This subtracts 1 tau from the 2 pulse delay that set_waveform will create
        self.math(self.tdds_offset_register.page, self.tdds_offset_register.addr, 
                      self.tdds_offset_register.addr, "-", self.tau_samples.addr)
        self.mathi(self.tdds_offset_register.page, self.tdds_offset_register.addr, 
                      self.tdds_offset_register.addr, "+", self.pi_len_unused)
        self.sync_all()

        # Pi/2 X pulse
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # ---------------------------------------------------------------------------------------
        # LOOP: CPMG Pi Pulse
        # ---------------------------------------------------------------------------------------
        self.n_cpmg_register.reset()
        self.label("LOOP_ncpmg") # Loop start point

        # pi pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0", phase=0)
        self.set_waveform()
        self.pulse(ch=self.cfg.mw_channel) # execute pulse
        self.sync_all() # Corrects time cursor bug

        # loop for all pi pulses 
        self.loopnz(self.n_cpmg_register.page, self.n_cpmg_register.addr, 'LOOP_ncpmg')
        # ---------------------------------------------------------------------------------------
        # End of Loop
        # ---------------------------------------------------------------------------------------

        # pi/2 X pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        self.set_waveform(pi_pulse=False, phase=0)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Reset for next loop
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        self.tdds_offset_register.reset() # reset to initial value 

    def set_waveform(self, pi_pulse=True, phase=None):
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
        self.offset_computations(double_tau=pi_pulse)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr) 

        # Update the address register to correspond to the correct waveform
        self.address_register.set_to(self.tdds_offset_register, '*', 
                                     self.pi_waveform_len_treg+self.half_pi_waveform_len_treg, physical_unit = False)
        if not pi_pulse: # pi/2 Waveforms are stored after pi waveforms
            self.address_register.set_to(self.address_register, '+', 
                                     self.pi_waveform_len_treg, physical_unit = False)
        
        # ----------------------------
        # Phase Selection & Offsets
        # ----------------------------
        if phase is not None:
            # Set the phase to the predetermined value
            self.phase_register.set_to(phase, physical_unit=True)
        else:
            # Determine the index of the current pulse
            # = n_cpmg_register % len(phase_list)
            self.bitwi(self.phase_register.page, self.phase_register.addr, 
                   self.n_cpmg_register.addr, "&", int(len(self.phase_list)-1))
            
            # Isolate the pulse type from the phase sequence
            # phase_register = phase_seq_register >> (n_cpmg_register % sequence length)
            self.bitw(self.phase_register.page, self.phase_register.addr,
                       self.phase_seq_register.addr, '>>', self.phase_register.addr)
            # Mask the LSB (unnecessary if using 2 bit encoding (full range))
            self.bitwi(self.phase_register.page, self.phase_register.addr, self.phase_register.addr, "&", 1)
            
            # Decode into the phase value
            # Multiplying values > 16 bits  fails, so instead left bit shift by 30
            self.bitwi(self.phase_register.page, self.phase_register.addr, self.phase_register.addr, "<<", 30)

    def offset_computations(self, double_tau=False):
        """
        Compute the tdds_offset_register and treg_offset_register for the next pulse. 
        Computes: 
        1. wait till the start of the next pulse (from the end of the previous waveform) (in sample timing resolution 200ps)
        2. Amount of wait done on the FPGA (converting to treg, ie dividing by 16 and taking the floor)
        3. Amount of wait done in waveform (samples),   equal to: (previous offset + the step in samples) % 16

        Tau_samples is a misnomer, instead it is:
            Actual Tau (samples) - Pi Pulse Waveform unused (samples)
            $$$ This is done to account for the delay in the pi pulse waveform itself
        """
        # tdds_offset_register = tdds_offset_register + sample_step
        # Computes the total delay needed until the next pulse from the end of this waveform
        # by adding amount of samples to wait + current sample offset
        self.math(self.tdds_offset_register.page, self.tdds_offset_register.addr, 
                  self.tdds_offset_register.addr, "+", self.tau_samples.addr)
        if double_tau: # Add another tau of delay (= 2*tau)
            self.math(self.tdds_offset_register.page, self.tdds_offset_register.addr, 
                      self.tdds_offset_register.addr, "+", self.tau_samples.addr)
            # avoid double counting the pi pulse delay
            self.mathi(self.tdds_offset_register.page, self.tdds_offset_register.addr, 
                      self.tdds_offset_register.addr, "-", self.pi_len_unused)
        # treg_offset_register = tdds_offset_register >> 4 (global offset)
        # Computes how long to stall the FPGA output in tproc cycles
        # from the total delay.
        # This operation also converts from samples (200ps) to treg (3.2ns)
        self.bitwi(self.tdds_offset_register.page, self.treg_offset_register.addr, 
                   self.tdds_offset_register.addr, ">>", int(np.log2(self.samps_per_clk)))
        # tdds_offset_register = tdds_offset_register & 15
        # Computes the remaining samples that the pulse should be delayed by
        # This is equivalent to: total delay (in samples) - fpga delay (in samples)
        self.bitwi(self.tdds_offset_register.page, self.tdds_offset_register.addr, 
                   self.tdds_offset_register.addr, "&", (self.samps_per_clk-1))

    def analyze_results(self, data):
        """
        Analyze CPMG-XY sweep results, renaming sweep_pts to tau_samples.
        Uses parent helper's analysis for signal/reference/contrast extraction.
        
        Returns ItemAttribute with: signal, reference (opt), contrast (opt), tau_samples
        """
        d = super().analyze_results(data)
        # Rename sweep_pts to tau_samples for clarity
        if hasattr(d, 'sweep_pts'):
            d.tau_samples = d.sweep_pts
            del d.sweep_pts
        return d
