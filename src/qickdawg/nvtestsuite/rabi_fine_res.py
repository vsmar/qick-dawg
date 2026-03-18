'''
Rabi sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for steps between pulses in Rabi sequence
using fine control of waveform start address and phase.

Modified from Tommy's RabiFineRes program. See his notebook for more details on the method.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times



class RabiFineRes(NVAveragerProgram):
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
    ]

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

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
            self.trigger( # Laser
                pins=[self.cfg.laser_gate_pmod],
                adc_trig_offset=0,
                width=self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg,
                t=0)
            self.wait_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg)
            self.sync_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg + 200)
        else:
            self.sync_all(200)  # give processor some time to configure pulses

    def body(self):
        # Initialize
        # laser_init_treg = self.cfg.laser_init_treg - (self.cfg.laser_readout_offset_treg + self.cfg.readout_integration_treg)
        self.trigger(pins = [self.cfg.laser_gate_pmod], width = self.cfg.laser_init_treg)
        self.wait_all(self.cfg.laser_init_treg)
        self.sync_all(self.cfg.laser_init_treg + self.cfg.mw_to_laser_delay_treg)

        # MW Pulse
        self.program_pulses()
        self.sync_all()

        # Readout
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


    def acquire(self, raw_data=False, *arg, **kwarg):
        data = super().acquire(readouts_per_experiment=1, *arg, **kwarg)

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