import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import dataclass, fields

FREQ_DAC = 4915.2e6 # Hz
NCO_WORD_SIZE = 32 # bits

SAMPLES_PER_TPROC = 16
SAMPLE_SIZE = 16 # 

DEBUG = False

# TODO:
# Display the RF signal of each channel
# Display the relative phase of each waveform on each channel (we can take a FFT over time)


# TODO: NCO state can only be updated at a treg boundary

# TODO: I should run simulation once the waveforms are decided, hence its better to have the phase adjustment done explicitly in NCO
@dataclass(frozen=True)
class EnvelopeIR:
    name: str
    idata: np.ndarray
    qdata: np.ndarray
    sample_rate: float = FREQ_DAC

@dataclass(frozen=True)
class PulseIR:
    ch: int
    start_ftsamp: int
    end_ftsamp: int
    duration_ftsamp: int
    envelope: EnvelopeIR
    freq: float                 # hz
    phase: float                # deg
    amp: float = 1.0
    phase_reset: bool = False


#####################################################
# NCO and Mixing
#####################################################

@dataclass
class NCOState:
    # Class to track the state of the NCO for each channel (predominantly for phase tracking)
    time_ftsamp: int
    fcw: int
    accum_phase: int

def calculate_fcw(freq_desired: float, freq_dac=FREQ_DAC, phase_word_len=NCO_WORD_SIZE):
    """Calculate the Frequency Control Word for a desired frequency.

    Args:
        freq_desired: Target frequency in Hz
        freq_dac: DAC clock frequency in Hz
        phase_word_len: Length of phase word in bits
    
    Returns:
        int: Frequency Control Word
    """
    fcw = int((freq_desired / freq_dac) * (2 ** phase_word_len))
    if DEBUG:
        actual_freq = (fcw / (2 ** phase_word_len)) * freq_dac
        error = actual_freq - freq_desired
        print(f"Desired Frequency: {freq_desired/1e6:.6f} MHz")
        print(f"FCW: {fcw} (0x{fcw:08X}, or {fcw/2**33:.3f} rad)") # FIXME: Check
        print(f"Actual Frequency: {actual_freq/1e6:.9f} MHz")
        print(f"Frequency Error: {error:.3f} Hz")
    return fcw

def advance_phase(state: NCOState, sample: int):
    elapsed = sample - state.time_ftsamp
    return (state.accum_phase + state.fcw * elapsed) % (2**NCO_WORD_SIZE)

def evolve_nco(state: NCOState, current_sample: int, new_fcw: int, phase_reset: bool):
    """Evolve the NCO phase for a given channel based on the last known state, current FCW, and time
    ...
    """
    if phase_reset:
        # TODO: Check if reset --> 0 or some other reference
        phase = 0 
    else:
        phase = advance_phase(state, current_sample)
    
    return NCOState(time_ftsamp=current_sample, fcw=new_fcw, accum_phase=phase)

def nco_signal(nco_state: NCOState, start_ftsamp: int, end_ftsamp: int):
    # Generate the NCO signal that gets mixed with the waveform
    samples = np.arange(start_ftsamp, end_ftsamp)

    phase_array = (nco_state.accum_phase + nco_state.fcw * (samples - nco_state.time_ftsamp)) % (2**NCO_WORD_SIZE)

    # NOTE: In the hardware there is clipping and interpolation that happens to map to phase
    phase = 2 * np.pi * (phase_array / 2**NCO_WORD_SIZE) 
    nco_cos = np.cos(phase)
    nco_sin = np.sin(phase)
    return nco_cos, nco_sin

def rotate_iq(I: np.ndarray, Q: np.ndarray, phase_deg: float):
    """
    Rotates the waveform to account for any rotation by the NCO or Waveform in assigning phase.
    TODO: Consider refactoring as the compiler approach gets more developed
    """
    theta = np.deg2rad(phase_deg)
    I_rot = I * np.cos(theta) - Q * np.sin(theta)
    Q_rot = I * np.sin(theta) + Q * np.cos(theta)
    return I_rot, Q_rot

def mix_pulse(pulse: PulseIR, nco: NCOState):
    # Mix the NCO signal with the waveform
    I = pulse.amp * pulse.envelope.idata
    Q = pulse.amp * pulse.envelope.qdata

    I, Q = rotate_iq(I, Q, pulse.phase)

    nco_cos, nco_sin = nco_signal(nco, pulse.start_ftsamp, pulse.end_ftsamp)

    # Real part of: (nco_I + j*nco_Q) * (waveform_I + j*waveform_Q)
    return nco_cos * I - nco_sin * Q


def simulate_channel(pulses):
    nco = NCOState(time_ftsamp=0, fcw=calculate_fcw(pulses[0].freq), accum_phase=0)

    rendered = []

    for pulse in pulses:
        new_fcw = calculate_fcw(pulse.freq)
        if new_fcw != nco.fcw:
            nco = evolve_nco(nco, current_sample=pulse.start_ftsamp, new_fcw=new_fcw, phase_reset=False) # TODO add compatibility for phase reset
        
        rendered.append(pulse, mix_pulse(pulse, nco))

    return rendered



# NOTE:
# Phase continuity is not maintained when changing frequency on a DAC tile
# However we have two DAC tiles on the 4x2, so we can maintain 2 frequencies across both channels
# TODO: Implement reference phase handling
# def change_frequency(self, ch, freq):
