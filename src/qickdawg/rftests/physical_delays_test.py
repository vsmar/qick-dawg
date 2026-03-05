"""
Physical Delays Program
=======================================================================
Generates a synchronized RF and laser pulse at t=0 to enable measurement
of the physical delays (response times) of the system.
"""

from ..nvpulsing.nvaverageprogram import NVAveragerProgram


class PhysicalDelays(NVAveragerProgram):
    """Generates synchronized RF and laser pulses for measuring system physical delays.
    
    Configuration Parameters:
        laser_gate_pmod (int): PMOD pin for laser gate trigger (typically 0).
            User must provide.
        
        mw_channel (int): Microwave generator channel. User must provide.
        
        mw_gain (int): Microwave gain (0-30000). Be cautious not to overload
            oscilloscope. User must provide.
                
        mw_freg (int, optional): Microwave frequency in register units.
            Default: 200 MHz.
        
        pulse_len_treg (int, optional): Pulse duration in register units (cycles).
            Default: 0.5 μs.
        
        mw_nqz (int, optional): Nyquist zone (1 for freq < fdss/2 ≈ 2495 MHz).
            Default: 1.
    """
    
    required_cfg = [
        "laser_gate_pmod",
        #"adc_channel", # delete if not necessary
        "mw_channel",
        "mw_gain",
    ]
    
    def initialize(self):
        """Initialize the program with default preset values and validation."""
        # Set default values for optional configs
        if not hasattr(self.cfg, "mw_freg"):
            self.cfg.mw_freg = self.freq2reg(200)  # 200 MHz
            self.cfg.mw_nqz = 1
        if not hasattr(self.cfg, "pulse_len_treg"):
            self.cfg.pulse_len_treg = self.us2cycles(0.5)  # 0.5 μs

        self.check_cfg()

        # Validate microwave gain
        if self.cfg.mw_gain > 30000:
            raise ValueError("Microwave gain must not exceed 30000 to avoid oscilloscope overload")

        # Configure microwave channel
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # self.setup_readout() # delete if not necessary

        # Set microwave pulse parameters
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.pulse_len_treg,
            phase=0
        )
        
        self.synci(100)  # Allow processor time to configure pulses

    def body(self):
        """Execute single trial: simultaneous RF pulse and laser trigger at t=0."""
        # Trigger microwave pulse
        self.pulse(ch=self.cfg.mw_channel, t=0)

        # Trigger laser gate at same time with same duration
        self.trigger(
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.pulse_len_treg,
            t=0
        )

        self.wait_all()
        self.synci(self.us2cycles(3))