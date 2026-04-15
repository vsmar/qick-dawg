# Pulse Compiler
import numpy as np

class PulseCompiler:
    def __init__(self):
        self.defined_pulses = {}
        self.registers_used = 0
        self.instructions = []
        self.ftsamp_per_treg = 16 # FIXME: self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        self.last_pulse = None # to track the last pulse played for handling deadtime and waveform selection (pulse item so that the values can be compared directly)
        self.last_phase = None # to track the last phase for handling waveform selection
        self.last_delay = None # to track the last delay for delay computations (specifically for register value delays)

    def add_pulse(self, name, type, IQarrays=None, length=None):
        if name in self.defined_pulses:
            raise ValueError(f"Pulse with name {name} already defined.")
        pulse = DefinedPulse(name, type, IQarrays, length)
        self.defined_pulses[name] = pulse

    def play_pulse(self, pulse_name, delay, phase=0):
        # phase can take the argument of a list or a single value
        # If i use a list style then must implement careful structured looping...
        # could abstract away phase list handling eventually 
        # (so that it auto identifies and create the appropriate looping structure)

        # Delay (ft units):
        # - Register value or immediate value
        #  Register: if sweeping or to improve code looping (use same cpmg sequence with different delays)
        #  Immediate: if fixed delay





        self.last_pulse = self.defined_pulses[pulse_name]
        self.last_delay = delay
        self.last_phase = phase

        # I think it initially makes most sense to compile a long list of equivalent instructions and then to take a 
        # optimization pass to identify repeated structures and optimize with loops and register values. 
        # This is similar to how a compiler would first generate assembly and then optimize it, but it allows us to focus on correctness first and optimization second.

    def create_waveforms(self):
        curr_addr_offset = 0
        # Logic to create waveforms based on defined pulses
        for pulse_name, pulse in self.defined_pulses.items():
            pulse.address_offset = curr_addr_offset
            curr_addr_offset += pulse.length_treg * self.ftsamp_per_treg # add Waveform memory used by this pulse (indexed in treg)
            if pulse.type == 'arb':
                # Create waveform from IQ arrays
                for i in range(0, self.ftsamp_per_treg):
                    idata = np.zeros(pulse.length_treg * 16)
                    qdata = np.zeros(pulse.length_treg * 16)

                    idata[i:i+pulse.length] = pulse.IQarrays[0] 
                    qdata[i:i+pulse.length] = pulse.IQarrays[1]
                     #FIXME: inherit class
                    self.add_envelope(ch=self.cfg.mw_channel, name=f"{pulse_name}_{i}", idata=idata, qdata=qdata)
                    # TODO: Add a way to track array for debugging / visualizing via my visualizer

        # TODO: Eventaully we should implement a way to make arbitrary length constant amplitude pulses with 
        # lengths configurable at runtime (rabi.pu style)
        pass

class DefinedPulse:
    def __init__(self, name, type, IQarrays=None, length=None):
        # require either length or IQarrays, but not both
        if length is not None and IQarrays is not None:
            raise ValueError("DefinedPulse must have either length or IQarrays, but not both.")
        self.name = name
        self.type = type # constant, arb # FIXME: Terminology for pulse types
        #first row of IQarrays is I, second row is Q
        self.IQarrays = np.trim_zeros(IQarrays, axis=1) if IQarrays is not None else None
        self.length = length if length is not None else self.IQarrays.shape[1]
        self.length_treg = max(int(np.ceil((self.length + (self.ftsamp_per_treg-1)) / self.ftsamp_per_treg)), 3)
        self.deadtime = self.length_treg * self.ftsamp_per_treg - self.length
        self.address_offset = None # to be set by PulseCompiler when creating waveforms

# FIXME Goals:
# identify how many free registers we have at start, track number of registers used at any point
# validate total number of instructions
# Auto perform setup - creation of waveforms with offsets, and structure selection of correct corresponding waveform
# validate amount of waveform used

play(halfpi, delay=67, phase=-90)
for n in 32:
    play("pi_pulse", delay=67, phase=n%1 * 90)

play(halfpi, delay=67, phase=0)
