'''
PulsePol sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for delay steps between pulses in PulsePol sequence
using fine control of waveform start address and phase.
Sequence:    (pi/2_X - tau - pi_Y - tau - pi/2_X - pi/2_Y - tau - pi_X - tau -pi/2_Y) x 2Npol
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np


class PulsePolFineRes(NVAveragerProgram):
    '''
    PulsePol sub-nanosecond resolution pulsing program
    '''
    required_cfg = [
        "mw_pi2_tdds",  # length of pi/2 pulse
        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "freq_freg",  # Microwave freq
        "n_pulsepol",  # number of pulsepol rounds
        "mw_channel",  # MW Channel
        "mw_nqz",  # 1 at 1405 MHz
        "mw_gain",  # MW Gain
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",
        # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between pulse seq end and trigger start of next seq
    ]

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Get samps per clk for later calculations. should be 16 for mw with current version 11/14/2025
        # if this changes from 16 then need to change waveform generation part
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Configure the waveforms for different fine resolution delay steps
        # must be 16 of them for pi and pi/2 pulses. Need to do all if sweeping at this fine resolution
        self.pi_len_tdds = self.cfg.mw_pi2_tdds * 2
        # Waveforms must have at least a length of 3 treg units
        # NOTE: pi_xy_waveform_len_treg may be redundant depending on bootstrapping / pulseshaping results
        self.pi_xy_waveform_len_treg = max(int(np.ceil((self.pi_len_tdds + self.samps_per_clk - 1) / self.samps_per_clk)), 3)
        self.pi_waveform_len_treg = max(int(np.ceil((self.pi_len_tdds + self.samps_per_clk - 1) / self.samps_per_clk)), 3)
        self.half_pi_waveform_len_treg = max(int(np.ceil((self.cfg.mw_pi2_tdds + self.samps_per_clk - 1) / self.samps_per_clk)), 3)
        
        # Generating 4 types of waveforms
        # 1. pi/2 pulse (arbitrary phase)
        # 2. pi-pulse (arbitrary phase)
        # 3. pi/2 (X) - pi/2 (Y)
        # 4. pi/2 (Y) - pi/2 (X)
        # NOTE: You need both variations of XY, and YX as it's a 90 degree lag 
        # (can't mix an XY with any NCO phase to get a YX)

        for i in range(self.samps_per_clk):
            # pi/2 (x) pulse
            idata = np.zeros(self.half_pi_waveform_len_treg * 16)
            qdata = np.zeros(self.half_pi_waveform_len_treg * 16)
            idata[i : i + self.cfg.mw_pi2_tdds] = 1
            qdata[i : i + self.cfg.mw_pi2_tdds] = 1
            idata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            qdata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=idata, qdata=qdata)

            # pi (x) pulse
            idata = np.zeros(self.pi_waveform_len_treg * 16)
            qdata = np.zeros(self.pi_waveform_len_treg * 16)
            idata[i: i + self.cfg.mw_pi2_tdds*2] = 1
            qdata[i: i + self.cfg.mw_pi2_tdds*2] = 1
            idata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            qdata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=idata, qdata=qdata)

            # pi/2_X - pi/2_Y
            idata = np.zeros(self.pi_xy_waveform_len_treg * 16)
            qdata = np.zeros(self.pi_xy_waveform_len_treg * 16)
            idata[i : i + self.cfg.mw_pi2_tdds] = 1
            qdata[i : i + self.cfg.mw_pi2_tdds] = 1
            idata[i + self.cfg.mw_pi2_tdds : i + self.cfg.mw_pi2_tdds*2] = -1
            qdata[i + self.cfg.mw_pi2_tdds : i + self.cfg.mw_pi2_tdds*2] = 1
            idata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            qdata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_xy_{i}", idata=idata, qdata=qdata)

            # pi/2_Y - pi/2_X
            idata = np.zeros(self.pi_xy_waveform_len_treg * 16)
            qdata = np.zeros(self.pi_xy_waveform_len_treg * 16)
            idata[i : i + self.cfg.mw_pi2_tdds] = 1
            qdata[i : i + self.cfg.mw_pi2_tdds] = 1
            idata[i + self.cfg.mw_pi2_tdds : i + self.cfg.mw_pi2_tdds*2] = 1
            qdata[i + self.cfg.mw_pi2_tdds : i + self.cfg.mw_pi2_tdds*2] = -1
            idata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            qdata *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_yx_{i}", idata=idata, qdata=qdata)



        # default special registers but need to manually modify them later
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr')  # for specifying which waveform
        self.phase_register = self.get_gen_reg(self.cfg.mw_channel, name='phase')  # for specifying (NCO) phase

        # there is dead time in the waveforms due to the waveform length being in treg units and using dds units
        # need to account for this in delays so need the values
        self.pi_len_unused_tdds = self.pi_waveform_len_treg * self.samps_per_clk - self.pi_len_tdds
        self.half_pi_len_unused_tdds = self.half_pi_waveform_len_treg * self.samps_per_clk - self.cfg.mw_pi2_tdds

        # two registers needed to account for unused time for a half pi pulse and pi pulse
        # one for coarse adjustment (treg_offset) and one for fine adjustment (tdds_offset)
        # we can initialize tdds_offset to already account for the first half_pi_pulse
        self.tdds_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                     name='tdds_offset',
                                                     init_val=self.pi_len_unused_tdds - self.half_pi_len_unused_tdds)

        self.treg_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                     name='treg_offset',
                                                     init_val=0)

        # PulsePol loop register
        self.n_pulsepol_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='n_pulsepol',
                                                    init_val=self.cfg.n_pulsepol - 1)

        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain)

        # Set up register for storing and sweeping delays
        self.delay_register = self.new_gen_reg(self.cfg.mw_channel,
                                               name='delay',
                                               init_val=self.cfg.delay_tdds_start - self.pi_len_unused_tdds)

        self.add_sweep(NVQickSweep(
            self,
            self.delay_register,
            self.cfg.delay_tdds_start - self.pi_len_unused_tdds,
            self.cfg.delay_tdds_end - self.pi_len_unused_tdds,
            self.cfg.nsweep_points))

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        # Format:
        # 1. π/2_x
        # 2. skip to step 4
        # 3. τ - π/2_y - π/2_x
        # 4. τ - π_y
        # 5. τ - π/2_y - π/2_x
        # 6. τ - π_-x
        # 7. loop to step 3
        # 8. τ - π/2_y

        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.tdds_offset_register.reset()  # reset the dds_offset adjustment
        self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.delay_register) # account for first pi

        self.n_pulsepol_register.reset()

        # 1. π/2_x
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # 2. skip past π/2_x - π/2_y
        self.condj(0, 0, "==", 0, "skip past π/2_y - π/2_x") 
        # NOTE: this instruction should always jumps, since register 0 == register 0. 
        # Check if there is an issue with it

        # Set the sequence loop back point
        self.label("LOOP_n_pulsepol")

        # 3. τ - π/2_y - π/2_x
        self.program_pulse(type="half_pi_yx") # NOTE: Phase=0
        self.pulse(ch=self.cfg.mw_channel)

        # Set the skip point
        self.label("skip past π/2_y - π/2_x")
        # 4. τ - π_y
        self.program_pulse(type="pi", phase=90)

        # 5. τ - π/2_x - π/2_y
        self.program_pulse(type="half_pi_xy") # NOTE: Phase=0
        
        # 6. τ - π_-x
        self.program_pulse(type="pi", phase=180)

        # 7. loop back (to 3.)
        self.loopnz(
            self.n_pulsepol_register.page,
            self.n_pulsepol_register.addr,
            'LOOP_n_pulsepol')
        
        # 8. τ - π/2_y
        self.program_pulse(type="half_pi", phase=90) 
        self.sync_all(self.cfg.pulse_seq_delay_treg)


    def program_pulse(self, type="pi", phase=0):
        """
        Method that computes the correct waveform address and fpga delay based on the current value of tau and the pulse to be played.
        Executes the delay and the correct pulse.
        Args:
            type (string):
                "half_pi": π/2 (default to x phase)
                "pi": π (default to x phase)
                "half_pi_xy": π/2_x - π/2_y
                "half_pi_yx": π/2_y - π/2_x
            phase (float/int):
                adjustment to make to phase. For example to play a Y pi pulse set to 90.
        """

        # Add total delay (adjusted)
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.delay_register)

        # Computes how long to stall the FPGA output in tproc cycles from the total delay. (coarse delay)
        self.bitwi(self.tdds_offset_register.page, self.treg_offset_register.addr, self.tdds_offset_register.addr, ">>", int(np.log2(self.samps_per_clk)))
        # Computes the remaining samples to delay the waveform by (fine delay)
        self.bitwi(self.tdds_offset_register.page, self.tdds_offset_register.addr, self.tdds_offset_register.addr, "&", self.samps_per_clk - 1)

        # NOTE: Must call set_pulse_registers() first, to avoid overwriting changes to address_register 
        # update pulse length & phase (using stand-in waveform of same length)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"{type}_0", phase=self.deg2reg(phase)) 

        # updating address register to select correct waveform based on the current offset
        self.address_register.set_to(self.tdds_offset_register, '*', self.half_pi_waveform_len_treg + self.pi_waveform_len_treg +
                                      self.pi_xy_waveform_len_treg + self.pi_xy_waveform_len_treg, physical_unit=False)

        # choose waveform address
        if type == "pi":
            self.address_register.set_to(self.address_register, '+', self.half_pi_waveform_len_treg, physical_unit=False)
        elif type == "half_pi_xy":
            self.address_register.set_to(self.address_register, '+', 
                                            self.half_pi_waveform_len_treg + self.pi_waveform_len_treg, physical_unit=False)
        elif type == "half_pi_yx":
            self.address_register.set_to(self.address_register, '+',
                                          self.half_pi_waveform_len_treg + self.pi_waveform_len_treg + self.pi_xy_waveform_len_treg, 
                                          physical_unit=False)
        
        # NOTE: For neatness delay and pulse calls are embedded here (cost: 1 additional line of instruction memory).
        # Coarse delay
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        # pulse
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

            
        
