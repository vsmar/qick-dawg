'''
Counting Duration
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

class CountingDurationFineRes(NVAveragerProgram):
    '''
    Rabi sub-nanosecond resolution pulsing program
    '''
    required_cfg = [
        "mw_duration_tdds", # length of mw
        "freq_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "laser_gate_pmod", # should be 0 for PMOD0_0
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq

        # Readout and delays
        "adc_channel",
        "laser_init_treg",
        "readout_integration_treg",
        "mw_to_laser_delay_treg", # Positive 
        "laser_readout_offset_treg",
    ]

    def initialize(self):
        if self.cfg.laser_init_treg < self.cfg.laser_readout_offset_treg + self.cfg.readout_integration_treg:
            raise ValueError("laser_init_treg must be long enough to account for the readout offset and integration time to ensure the system is initialized before readout starts.")

        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

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
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     phase = 0)
                
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
        # Compute duration of laser pulse directly before trigger based on the total initialization time
        # laser_init_accounted_treg = self.cfg.laser_init_treg - (self.cfg.laser_readout_offset_treg + self.cfg.readout_integration_treg)

        # self.trigger(pins = [self.cfg.laser_gate_pmod], width = laser_init_accounted_treg)
        # self.sync_all(self.cfg.mw_to_laser_delay_treg + laser_init_accounted_treg)

        # init
        self.trigger(pins = [self.cfg.laser_gate_pmod], width = self.cfg.laser_init_treg)
        self.wait_all(self.cfg.laser_init_treg)
        self.sync_all(self.cfg.laser_init_treg + self.cfg.mw_to_laser_delay_treg)

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse")
        self.pulse(ch=self.cfg.mw_channel) 
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



    def acquire(self, raw_data=False, *arg, **kwarg):
        data = super().acquire(readouts_per_experiment=1, *arg, **kwarg)

        if raw_data is False:
            data = self.analyze_results(data)

        return data
        
    def analyze_results(self, data):
        d = ItemAttribute()

        # Ensure it's a numpy scalar/array
        data = np.asarray(data)

        # Extract scalar value safely
        d.signal = data.item() if data.ndim == 0 else data

        # Normalize
        d.signal = d.signal / (
            self.cfg.readout_integration_treg * self.cfg.reps
        )

        return d