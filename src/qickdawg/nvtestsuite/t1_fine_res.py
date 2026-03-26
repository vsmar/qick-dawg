'''
T1 Fine Resolution
=======================================================================
T1 sequence with MW pi pulse followed by a variable delay before readout.
Uses time sweep units (treg/tns/tus), not fine-time sample sweep units.

Author: Victor Marcenac
'''

from itemattribute import ItemAttribute
from qickdawg.util.apply_on_axis_0_n_times import apply_on_axis_0_n_times

from ..nvpulsing.nvaverageprogram import NVAveragerProgram
from ..nvpulsing.nvqicksweep import NVQickSweep
from .readout_helpers import ReadoutHelpers
import numpy as np


class T1FineRes(ReadoutHelpers, NVAveragerProgram):
    '''
    T1 sequence with fixed MW pi pulse and swept waiting delay before readout.
    '''
    required_cfg = [
        # Channels and pmods
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod", # should be 0 for PMOD0_0

        # MW pulse parameters
        "mw_pi_ftsamp",
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "mw_fMHz",

        # Sweep parameters
        "t1_delay_start_treg", # delay between end of MW pulse and laser RO
        "t1_delay_end_treg",
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
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pi_ftsamp / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_ftsamp = self.mw_pulse_waveform_len_treg * self.samps_per_clk
        # Create waveform with exact duration
        i_data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
        q_data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
        i_data[:self.cfg.mw_pi_ftsamp] = 1
        q_data[:self.cfg.mw_pi_ftsamp] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=i_data, qdata=q_data)
        # MW pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="pulse",
                                     phase=0)
        # Explicitly arm the pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel)
        
        # Setup T1-delay sweep (time-register units).
        self.t1_delay_register = self.new_gen_reg(self.cfg.mw_channel, "t1_delay")
        scaling_mode = getattr(self.cfg, 'scaling_mode', 'linear')
        scaling_factor = getattr(self.cfg, 'scaling_factor', '')

        if scaling_mode == 'exponential':
            self.add_sweep(NVQickSweep(
                self,
                self.t1_delay_register,
                self.cfg.t1_delay_start_treg,
                self.cfg.t1_delay_end_treg,
                expts=self.cfg.nsweep_points,
                scaling_mode=scaling_mode,
                scaling_factor=scaling_factor))

        elif scaling_mode == 'linear':
            self.add_sweep(NVQickSweep(
                self,
                self.t1_delay_register,
                self.cfg.t1_delay_start_treg,
                self.cfg.t1_delay_end_treg,
                self.cfg.nsweep_points))
        else:
            assert 0, 'cfg.scaling_mode must be "linear" or "exponential"'
        
        # start from a close to initialized state
        self.pre_init()

    def body(self):
        self.laser_init()
        self.program_pulse()
        self.signal_and_reference_readout(self.program_pulse)

    def program_pulse(self):
        self.pulse(ch=self.cfg.mw_channel)
        self.sync(self.t1_delay_register.page, self.t1_delay_register.addr)
        self.sync_all()

    
    def acquire(self, raw_data=False, *arg, **kwarg):
        # self.acquire --> ReadoutHelpers.acquire --> NVAveragerProgram.acquire
        data = super().acquire(raw_data=raw_data, sweep_param='t1_delay_treg', *arg, **kwarg)
        return data


# Backward-compatible alias while callers migrate to T1FineRes.
PODMRFineRes = T1FineRes