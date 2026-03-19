'''
Counting Duration
=======================================================================
Min resolution of 200ps for steps between pulses in Rabi sequence
using fine control of waveform start address and phase.

Modified from Tommy's RabiFineRes program. See his notebook for more details on the method.
'''

from qickdawg.util.apply_on_axis_0_n_times import apply_on_axis_0_n_times

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from .readout_helpers import ReadoutHelpers
import numpy as np
from itemattribute import ItemAttribute


class CountingDurationFineRes(NVAveragerProgram, ReadoutHelpers):
    '''
    Rabi sub-nanosecond resolution pulsing program
    '''
    required_cfg = [
        "mw_pi_tdds", # length of mw
        "mw_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "laser_gate_pmod", # should be 0 for PMOD0_0
        "relax_delay_treg", # delay between pulse seq end and trigger start of next seq

        # Readout and delays
        "adc_channel",
        "laser_on_treg",
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
        # Waveforms must have at least a length of 3 treg units 
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_duration_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk  # in tdds units
        
        i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        i_data[:self.cfg.mw_duration_tdds] = 1
        q_data[:self.cfg.mw_duration_tdds] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=i_data, qdata=q_data)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="pulse",
                                     phase = 0)
            
        self.set_pulse_registers(ch=self.cfg.mw_channel)

        # Setup laser
        self.setup_readout()

        self.pre_init()

    def body(self):
        # Initialize
        self.laser_init()

        # MW Pulse
        self.program_pulse()

        # Readout with optional reference
        self.signal_and_reference_readout(self.program_pulse)

    def program_pulse(self):
        """Program the MW pulse sequence"""
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

    # TODO: See to cleaning up acquire later

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
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

        d.contrast = d.contrast1 - d.contrast2

        d.contrast1 = apply_on_axis_0_n_times(d.contrast1.astype(ret_type), func, n)
        d.signal1 = apply_on_axis_0_n_times(d.signal1.astype(ret_type), func, n)
        d.reference1 = apply_on_axis_0_n_times(d.reference1.astype(ret_type), func, n)

        d.contrast2 = apply_on_axis_0_n_times(d.contrast2.astype(ret_type), func, n)
        d.signal2 = apply_on_axis_0_n_times(d.signal2.astype(ret_type), func, n)
        d.reference2 = apply_on_axis_0_n_times(d.reference2.astype(ret_type), func, n)

        d.contrast = apply_on_axis_0_n_times(d.contrast.astype(ret_type), func, n)

        return d

    def plot_sequence(cfg=None):
        '''
        Function that plots the pulse sequence generated by this program

        Parameters
        ----------
        cfg: `.NVConfiguration` or None(default None)
            If None, this plots the squence with configuration labels
            If a `.NVConfiguration` object is supplied, the configuraiton value are added to the plot
        '''
        graphics_folder = os.path.join(os.path.dirname(__file__), '../../graphics')
        image_path = os.path.join(graphics_folder, 'RABI.png')

        if cfg is None:
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(455, 510, "config.reps", fontsize=14)
            plt.text(350, 440, "config.laser_on", fontsize=14)
            plt.text(
                195,
                580,
                " Sweep pi/2 pulse time linearly from config.mw_start to config.mw_end in config.mw_delta sized steps",
                fontsize=12)
            plt.text(265, 355, "config.readout_integration", fontsize=14)
            plt.text(527, 355, " config.readout_integration", fontsize=14)
            plt.text(190, 368, " pi/2\npulse", fontsize=14)
            plt.text(735, 355, "config.relax_delay", fontsize=14)
            plt.text(220, 407, "config.laser_readout_offset", fontsize=14)
            plt.text(430, 407, "config.readout_reference_start", fontsize=14)
            plt.title("           Rabi Oscillation Pulse Sequence", fontsize=20)

        else:
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(420, 510, "Repeat {} times".format(cfg.reps), fontsize=14)
            plt.text(350, 440, "laser_on_tus = {} us".format(str(cfg.laser_on_tus)[:4]), fontsize=14)

            string = f"Sweep pi/2 pulse time linearly from {int(cfg.mw_start_treg)} time register"
            string += f" to {int(cfg.mw_end_treg)} time register in steps of "
            string += f"{str(cfg.mw_delta_treg)[:4]} time register"
            plt.text(195, 580, string, fontsize=12)

            plt.text(265, 370, "readout_integration  \n       = {} ns".format(
                int(cfg.readout_integration_tns)), fontsize=14)
            plt.text(527, 370, "readout_integration  \n      = {} ns".format(
                int(cfg.readout_integration_tns)), fontsize=14)
            plt.text(190, 368, " pi/2\npulse", fontsize=14)
            plt.text(735, 370, "relax_delay \n = {} ns".format(int(cfg.relax_delay_tns)), fontsize=14)
            plt.text(235, 407, "laser_offset = {} ns".format(int(cfg.laser_readout_offset_tns)), fontsize=14)
            plt.text(430, 407, "readout_reference_start = {} us".format(
                int(cfg.readout_reference_start_tus)), fontsize=14)
            plt.title("           Rabi Oscillation Pulse Sequence", fontsize=20)
