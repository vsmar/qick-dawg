'''
Rabi sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for steps between pulses in Rabi sequence
using fine control of waveform start address and phase.

Modified from Tommy's RabiFineRes program. See his notebook for more details on the method.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from .readout_helpers import ReadoutHelpers
import numpy as np



class RabiFineRes(NVAveragerProgram, ReadoutHelpers):
    '''
    Rabi sub-nanosecond resolution pulsing program
    '''
    required_cfg = [        
        "mw_duration_tdds_start",
        "mw_duration_tdds_end",
        "nsweep_points",
        "freq_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "laser_gate_pmod", # should be 0 for PMOD0_0
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq
        "adc_channel",
        "laser_init_treg",
        # Readout and delays
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
        # Waveforms must have at least a length of 3 treg units but want multiple of 2 so use 4 so 4*16= 64 = 2^6 for 16 samps per clk
        self.mw_pulse_waveform_len_treg = 4
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk  # in tdds units
        
        for i in np.arange(0, self.mw_pulse_waveform_len_tdds+1, 1):
            i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
            q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
            i_data[:i] = 1
            q_data[:i] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pulse_{i}", idata=i_data, qdata=q_data)

        # default special registers but need to manually modify them later
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr') # for specifying which waveform
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     phase = 0)
        
        # mw duration register
        self.mw_duration_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_duration',
                                                   init_val=self.cfg.mw_duration_tdds_start)

        # mw coarse and fine pulse loop register
        self.coarse_mw_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_coarse',
                                                   init_val=0)
        self.fine_mw_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_fine',
                                                   init_val=0)

        self.add_sweep(NVQickSweep(self,
                                   reg=self.mw_duration_register,
                                   start=self.cfg.mw_duration_tdds_start,
                                   stop=self.cfg.mw_duration_tdds_end,
                                   expts=self.cfg.nsweep_points))
        
        # Setup laser
        self.setup_readout()

        # start from a close to initialized state
        if self.cfg.pre_init:
            self.pre_init()
        
        self.synci(200)  # Give tproc time to get ahead

    def body(self):
        self.laser_init()
        self.program_pulses()
        self.signal_and_reference_readout(self.program_pulses)

    def program_pulses(self):
        # set coarse and fine registers based on duration. coarse is x//64 and then multiply by 4 to get treg units
        self.bitwi(self.coarse_mw_register.page, self.coarse_mw_register.addr, self.mw_duration_register.addr, ">>", int(np.log2(self.mw_pulse_waveform_len_tdds)))
        self.bitwi(self.coarse_mw_register.page, self.coarse_mw_register.addr, self.coarse_mw_register.addr, "<<", int(np.log2(self.mw_pulse_waveform_len_treg)))
        self.bitwi(self.fine_mw_register.page, self.fine_mw_register.addr, self.mw_duration_register.addr, "&", self.mw_pulse_waveform_len_tdds - 1)
        
        # if there is no coarse part just do fine part
        self.condj(self.coarse_mw_register.page, self.coarse_mw_register.addr, "==", 0, "JUMP_NO_COARSE")

        # since using sync all (and need to use it for accurate timing), it will always play a pulse so need to subtract onewaveform length
        self.coarse_mw_register.set_to(self.coarse_mw_register, "-", self.mw_pulse_waveform_len_treg, physical_unit=False)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pulse_{self.mw_pulse_waveform_len_tdds}", mode = "periodic")
        self.pulse(ch=self.cfg.mw_channel) 
        self.sync_all()
        self.sync(self.coarse_mw_register.page, self.coarse_mw_register.addr)

        self.label("JUMP_NO_COARSE")
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pulse_{0}", mode = "oneshot")
        self.address_register.set_to(self.fine_mw_register, '*', self.mw_pulse_waveform_len_treg, physical_unit = False)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()


    def analyze_results(self, data):
        """
        Analyze Rabi sweep results, renaming sweep_pts to duration.
        Uses parent helper's analysis for signal/reference/contrast extraction.
        
        Returns ItemAttribute with: signal, reference (opt), contrast (opt), duration
        """
        d = super().analyze_results(data)
        # Rename sweep_pts to duration for clarity
        if hasattr(d, 'sweep_pts'):
            d.duration = d.sweep_pts
            del d.sweep_pts
        return d