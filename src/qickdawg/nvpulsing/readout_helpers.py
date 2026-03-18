'''
Helper class for shared laser initialization and readout sequences
=======================================================================
Provides common laser_init(), readout(), and pre_init() methods
for NV test suite programs.
'''


class ReadoutHelpers:
    """
    Mixin class that provides shared laser initialization and readout methods
    for NV pulse programs. Intended to be used with NVAveragerProgram subclasses.
    """
    
    def laser_init(self):
        """Initialize spin state with laser pulse"""
        self.trigger(pins=[self.cfg.laser_gate_pmod], width=self.cfg.laser_init_treg)
        self.wait_all(self.cfg.laser_init_treg)
        self.sync_all(self.cfg.laser_init_treg + self.cfg.mw_to_laser_delay_treg)

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

    def pre_init(self):
        """Pre-initialization pulse to bring system close to initialized state"""
        self.trigger(  # Laser
            pins=[self.cfg.laser_gate_pmod],
            adc_trig_offset=0,
            width=self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg,
            t=0)
        self.wait_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg)
        self.sync_all(self.cfg.readout_integration_treg + self.cfg.laser_readout_offset_treg + 200)
