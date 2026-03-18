'''
Helper class for shared laser initialization and readout sequences
=======================================================================
Provides common laser_init(), readout(), and acquisition/analysis methods
for NV test suite programs.
'''

import numpy as np
from itemattribute import ItemAttribute


class ReadoutHelpers:
    """
    Mixin class that provides shared laser initialization, readout, and data analysis methods
    for NV pulse programs. Intended to be used with NVAveragerProgram subclasses.
    """
    
    def check_laser_init_timing(self):
        """Validate that laser initialization duration is sufficient"""
        laser_init_duration = self.cfg.laser_init_treg - (self.cfg.laser_readout_offset_treg + self.cfg.readout_integration_treg)
        if laser_init_duration <= 3:
            raise ValueError(f"laser_init_treg ({self.cfg.laser_init_treg}) must be long enough so that "
                           f"laser_init_treg - (laser_readout_offset_treg + readout_integration_treg) > 3. "
                           f"Current calculated duration: {laser_init_duration}")
    
    def setup_readout_registers(self, mw_channel):
        """Setup registers and validation for readout sequence"""
        self.check_laser_init_timing()
        self.mw_gain_register = self.get_gen_reg(mw_channel, "gain")
    
    def laser_init(self):
        """Initialize spin state with laser pulse"""
        laser_init_duration = self.cfg.laser_init_treg - (self.cfg.laser_readout_offset_treg + self.cfg.readout_integration_treg)
        self.trigger(pins=[self.cfg.laser_gate_pmod], width=laser_init_duration)
        self.wait_all(laser_init_duration)
        self.sync_all(laser_init_duration + self.cfg.mw_to_laser_delay_treg)

    def readout(self):
        """Readout spin state with laser and ADC"""
        # RO
        self.trigger_no_off(  # Laser
            pins=[self.cfg.laser_gate_pmod],
            t=0)
        self.trigger(  # Laser + ADC
            adcs=self.cfg.adcs,
            pins=[self.cfg.laser_gate_pmod],
            adc_trig_offset=0,
            width=self.cfg.readout_integration_treg,
            t=self.cfg.laser_readout_offset_treg)
        self.wait_all(self.cfg.readout_integration_treg)
        self.sync_all(self.cfg.readout_integration_treg + self.cfg.pulse_seq_delay_treg)

    def signal_and_reference_readout(self, program_pulse_fn):
        """
        Perform signal readout (with RF pulse), then optionally reference readout (without RF).
        
        The signal readout is performed immediately (pulse was already applied in body).
        If cfg.get_reference is True, reinitializes and takes a reference readout with MW gain set to 0.
        
        Parameters
        ----------
        program_pulse_fn : callable
            Function to call to program the pulse sequence for the reference measurement.
            This is typically self.program_pulses() or equivalent.
        """
        # Signal readout (pulse was already applied in body)
        self.readout()
        
        # Reference readout (if configured)
        if self.cfg.get_reference:
            # Set MW gain to 0 for reference
            self.mw_gain_register.set_to(0, physical_unit=False)
            self.laser_init()
            program_pulse_fn()
            self.readout()
            # Restore original gain
            self.mw_gain_register.set_to(self.cfg.mw_gain, physical_unit=False)

    def pre_init(self):
        """Pre-initialization pulse to bring system close to initialized state"""
        self.trigger(  # Laser
            pins=[self.cfg.laser_gate_pmod],
            adc_trig_offset=0,
            width=self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg,
            t=0)
        self.wait_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg)
        self.sync_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg)

    def acquire(self, raw_data=False, *arg, **kwarg):
        """
        Generic acquire method that handles reference readouts based on config.
        
        Parameters
        ----------
        raw_data : bool
            If False, analyzes results; if True, returns raw data
        
        Returns
        -------
        Analyzed or raw data depending on raw_data flag
        """
        readouts = 2 if self.cfg.get_reference else 1
        data = super().acquire(readouts_per_experiment=readouts, *arg, **kwarg)
        if not raw_data:
            data = self.analyze_results(data)
        return data

    def analyze_results(self, data):
        """
        Generic analysis for experiments with signal and optional reference readouts.
        
        Handles both swept and non-swept experiments. For swept experiments, subclasses
        should override to add the appropriate sweep axis (e.g., duration, frequencies).
        
        Parameters
        ----------
        data
            (1D or multi-D np.array) data returned from acquire()
        
        Returns
        -------
        ItemAttribute with signal, optional reference, contrast, and optional sweep_pts
        """
        data = np.reshape(data, self.data_shape)
        d = ItemAttribute()
        
        norm_factor = self.cfg.readout_integration_treg * self.cfg.reps
        
        if data.ndim == 0:
            # Scalar case (single readout, no sweep)
            d.signal = data.item() / norm_factor
        else:
            # Array case - extract signal (first readout)
            d.signal = data[..., 0]
            n = len(d.signal.shape) - 1
            from ..util import apply_on_axis_0_n_times
            d.signal = apply_on_axis_0_n_times(d.signal, np.sum, n)
            d.signal_cts_s = d.signal / norm_factor
            
            # Extract and process reference if available
            if self.cfg.get_reference and data.shape[-1] > 1:
                d.reference = data[..., 1]
                d.reference = apply_on_axis_0_n_times(d.reference, np.sum, n)
                d.reference_cts_s = d.reference / norm_factor
                d.contrast = d.signal_cts_s / d.reference_cts_s
                # maybe rename to signal/reference since contrast can mean something different
        
        # Add sweep points if available (subclass should override to rename as appropriate)
        if hasattr(self, 'qick_sweeps') and len(self.qick_sweeps) > 0:
            d.sweep_pts = self.qick_sweeps[0].get_sweep_pts()
        
        return d
