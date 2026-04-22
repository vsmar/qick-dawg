'''
Bootstrap sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for delay steps 
using fine control of waveform start address and phase.
Follows https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.077601?__cf_chl_tk=OJit3IH_f_.tMb6W6g0SHQM6.uV36H9ieFmu3en4wj4-1759769753-1.0.1.1-67pXH.nAif8JKOAf_MDe0Gjfad2FLsFhsTxxyD9G7lo 
'''
from itemattribute import ItemAttribute
from qickdawg.util.apply_on_axis_0_n_times import apply_on_axis_0_n_times

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from .readout_helpers import ReadoutHelpers
import numpy as np

class BootstrapFineRes(NVAveragerProgram, ReadoutHelpers):
    '''
    Bootstrap sub-nanosecond resolution pulsing program.
    '''
    required_cfg = [      
        # Channels and pmods
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod", # should be 0 for PMOD0_0

        # MW pulse parameters
        "mw_freg", # Microwave freq 
        "mw_nqz", # 1 at <2.457 GHz
        "mw_gain", # MW Gain

        # Bootstrap specific parameters
        "mw_pi2_ftsamp",
        "btwn_mw_delay_ftsamp", # delay between mw pulses
        "bootstrap_experiment_number",

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

    # helper function to make pulse envelope for bootstrap sequences
    # returns I and Q data arrays
    def make_pulse_envelope(self):

        if self.cfg.bootstrap_experiment_number not in range(1,13):
            print("Error: bootstrap_experiment_number must be an integer between 1 and 12")
            raise ValueError
        
        # Configure the waveforms for different fine resolution pulse steps
        # Waveforms must have at least a length of 3 treg units 
        if self.cfg.bootstrap_experiment_number in [1,2]:
            self.mw_pulse_seq_len_ftsamp = self.cfg.mw_pi2_ftsamp 
        elif self.cfg.bootstrap_experiment_number in [3,4,5,6]:
            self.mw_pulse_seq_len_ftsamp = 3*self.cfg.mw_pi2_ftsamp + self.cfg.btwn_mw_delay_ftsamp
        elif self.cfg.bootstrap_experiment_number in [7,8]:
            self.mw_pulse_seq_len_ftsamp = 2*self.cfg.mw_pi2_ftsamp + self.cfg.btwn_mw_delay_ftsamp
        elif self.cfg.bootstrap_experiment_number in [9,10,11,12]:
            self.mw_pulse_seq_len_ftsamp = 4*self.cfg.mw_pi2_ftsamp + 2*self.cfg.btwn_mw_delay_ftsamp
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.mw_pulse_seq_len_ftsamp / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_ftsamp = self.mw_pulse_waveform_len_treg * self.samps_per_clk  
        
        i_data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
        q_data = np.zeros(self.mw_pulse_waveform_len_ftsamp)

        axis = {"x": (1, 1), "y": (-1, 1)}
        pulse_type = {"pi2": self.cfg.mw_pi2_ftsamp, "pi": 2 * self.cfg.mw_pi2_ftsamp}
        seqs = {
            1: [("pi2", "x")],
            2: [("pi2", "y")],
            3: [("pi2", "x"), ("delay", None), ("pi", "x")],
            4: [("pi2", "y"), ("delay", None), ("pi", "y")],
            5: [("pi", "y"), ("delay", None), ("pi2", "x")],
            6: [("pi", "x"), ("delay", None), ("pi2", "y")],
            7: [("pi2", "y"), ("delay", None), ("pi2", "x")],
            8: [("pi2", "x"), ("delay", None), ("pi2", "y")],
            9: [("pi2", "x"), ("delay", None), ("pi", "x"), ("delay", None), ("pi2", "y")],
            10: [("pi2", "y"), ("delay", None), ("pi", "x"), ("delay", None), ("pi2", "x")],
            11: [("pi2", "x"), ("delay", None), ("pi", "y"), ("delay", None), ("pi2", "y")],
            12: [("pi2", "y"), ("delay", None), ("pi", "y"), ("delay", None), ("pi2", "x")],
        }

        # fill i/q arrays
        idx = 0
        for typ, ax in seqs[self.cfg.bootstrap_experiment_number]:
            if typ == "delay":
                idx += self.cfg.btwn_mw_delay_ftsamp
                continue
            length = pulse_type[typ]
            i_val, q_val = axis[ax]
            i_data[idx : idx + length] = i_val
            q_data[idx : idx + length] = q_val
            idx += length

        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        return i_data, q_data

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.setup_helper_registers(self.cfg.mw_channel)

        # Setup laser
        self.setup_readout()

        # Get samps per clk for later calculations. should be 16 for mw with current version rfsoc 11/14/2025
        # if this changes from 16 then need to change waveform generation part
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        i_data , q_data = self.make_pulse_envelope()
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse_seq", idata=i_data, qdata=q_data)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain,
                                     phase = self.deg2reg(0))
        
        self.pre_init()

    def body(self):
        self.initialize_spin()
        self.program_pulse()
        self.readout_and_reference(self.program_pulse)

    def program_pulse(self):
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse_seq")
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()


    def acquire(self, raw_data=False, *arg, **kwarg):

        data = super().acquire(readouts_per_experiment=4, *arg, **kwarg)

        if raw_data:
            return data

        data = np.reshape(data, self.data_shape)

        d = ItemAttribute()

        d.signal1 = data[..., 0]
        d.reference1 = data[..., 1]
        d.signal2 = data[..., 2]
        d.reference2 = data[..., 3]

        n = len(d.signal1.shape)

        if self.cfg.edge_counting is False:
            ret_type = float
            func = np.mean
        else:
            ret_type = int
            func = np.sum

        if self.cfg.edge_counting is False:
            d.contrast1 = ((d.signal1 - d.reference1) / d.reference1 * 100)
            d.contrast2 = ((d.signal2 - d.reference2) / d.reference2 * 100)
        else:
            d.contrast1 = d.signal1 - d.reference1
            d.contrast2 = d.signal2 - d.reference2

        d.contrast = d.signal1 / d.signal2

        d.contrast1 = apply_on_axis_0_n_times(d.contrast1.astype(ret_type), func, n)
        d.signal1 = apply_on_axis_0_n_times(d.signal1.astype(ret_type), func, n)
        d.reference1 = apply_on_axis_0_n_times(d.reference1.astype(ret_type), func, n)

        d.contrast2 = apply_on_axis_0_n_times(d.contrast2.astype(ret_type), func, n)
        d.signal2 = apply_on_axis_0_n_times(d.signal2.astype(ret_type), func, n)
        d.reference2 = apply_on_axis_0_n_times(d.reference2.astype(ret_type), func, n)

        d.contrast = apply_on_axis_0_n_times(d.contrast.astype(ret_type), func, n)

        norm_factor = self.cfg.readout_integration_tns * 1e-9 * self.cfg.reps
        for key in ['signal1', 'reference1', 'signal2', 'reference2']:
            d[key + '_cts_s'] = d[key] / norm_factor

        return d

