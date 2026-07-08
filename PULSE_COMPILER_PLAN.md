# Pulse Compiler Architecture Plan

## Overview
Translate high-level pulse sequence descriptions into optimized hardware instructions. Start with constant amplitude pulses, then extend to arbitrary waveforms.

**Input:** `play(pulse_name, delay, phase)` commands
**Output:** Optimized instruction set with waveforms and timing

---

## Stage 1: Input & Validation

### What the user provides:
```python
play("halfpi", delay=67, phase=-90)
for n in 32:
    play("pi_pulse", delay=67, phase=n%1 * 90)
play("halfpi", delay=67, phase=0)
```

### Compiler responsibilities:
- Parse pulse sequence into a list of (pulse_name, delay, phase) tuples
- Validate that all referenced pulses are defined
- Track dependencies and constraints
- Handle loops and control flow (or flatten initially for correctness)

### Data structure:
```python
class SequenceInstruction:
    pulse_name: str      # which pulse to play
    delay: int | str     # ft samples or register name
    phase: int | list    # degrees or register sweep
    timestamp: int       # absolute position in sequence (ft samples)
```

---

## Stage 2: Waveform Strategy & Planning

### Key decisions for constant amplitude pulses:
1. **Duration alignment** - Pulses need alignment to sample boundaries (multiples of `ftsamp_per_treg`)
2. **Phase encoding** - How to apply phase?
   - Option A: Create distinct waveform for each phase variant
   - Option B: Use NCO phase register (single waveform, update phase register)
   - Option C: Hybrid (NCO for coarse, waveforms for fine if needed)
3. **Deadtime handling** - Pulses always have deadtime padding to alignment boundary

### Planning data structure:
```python
class WaveformPlan:
    pulse_name: str
    phase: int
    waveform_id: str           # "halfpi_-90", "pi_pulse_0", etc.
    address_offset: int        # location in waveform memory
    length_treg: int           # duration in treg units
    deadtime_ftsamp: int       # padding needed
    phase_method: str          # "nco" | "waveform" | "both"
    nco_phase_register: int    # if using NCO
    waveform_address: int      # if using waveform selection
```

### Compiler responsibilities:
- Determine which waveforms are actually needed (avoid creating redundant waveforms)
- Decide phase encoding strategy per pulse/phase combination
- Calculate waveform memory requirements
- Assign waveform addresses and memory layout

---

## Stage 3: Timeline & Timing Validation

### What gets validated:
1. **Absolute timing** - Each instruction has a known start time (ft sample)
2. **Delay feasibility** - Can the delay between pulses be achieved?
   - If delay < 2×`ftsamp_per_treg`: cannot use separate waveforms, must use NCO only
   - If delay ≥ 2×`ftsamp_per_treg`: can use separate waveforms or NCO
3. **Register resources** - Track register allocation for sweep variables
4. **Total sequence duration** - Validate it fits within hardware constraints

### Data structure:
```python
class TimingAnalysis:
    instructions: List[TimedInstruction]
    total_duration_ftsamp: int
    total_duration_us: float
    register_usage: Dict[str, int]      # register_name -> value
    waveform_memory_used: int            # in ft samples
    conflicts: List[str]                 # potential issues
```

---

## Stage 4: Instruction Generation

### What gets output:
For each play command, generate the actual hardware instructions:
- **Envelope setup** (if needed): Set waveform address, phase register
- **Timing control**: Set delay register or immediate delay
- **Play instruction**: Execute the pulse
- **Deadtime handling**: Account for pulse duration + deadtime

### Instruction format (pseudocode):
```python
# Example sequence
nco.set_phase(register=phase_reg)           # Set phase from register
envelope.select_waveform(address=wf_addr)   # Point to waveform
wait(delay=delay_value_ft)                  # Apply pre-pulse delay
play(envelope)                              # Execute pulse
# Pulse duration + deadtime automatically handled by waveform
```

### Data structure:
```python
class HardwareInstruction:
    instruction_type: str    # "nco_phase", "waveform_select", "wait", "play"
    parameters: Dict         # instruction-specific params
    timing_ft: int           # when this executes
    duration_ft: int         # how long it takes
```

---

## Stage 5: Optimization

### Optimization opportunities:
1. **Loop detection** - Identify repeated pulse sequences and create loops
2. **Register reuse** - Minimize register allocation by reusing where possible
3. **Waveform consolidation** - Merge similar waveforms if memory is tight
4. **Instruction compression** - Combine operations where possible

### For now: Focus on correctness, optimize later

---

## Stage 6: Backend Integration

### Final outputs:
- List of waveforms to create/upload to hardware
- List of hardware instructions in execution order
- Register initialization values
- Timing validation report

---

## Implementation Roadmap (Recommended Order)

### Phase 1: Core Structure
- [ ] `SequenceInstruction` class
- [ ] Input parsing and validation
- [ ] Pulse definition validation

### Phase 2: Waveform Planning
- [ ] `WaveformPlan` class
- [ ] Waveform deduplication logic
- [ ] Phase handling strategy selection

### Phase 3: Timing
- [ ] Timeline construction (absolute timing for each instruction)
- [ ] Delay feasibility checking
- [ ] Resource tracking

### Phase 4: Code Generation
- [ ] Basic instruction generation
- [ ] Envelope/waveform selection logic
- [ ] Phase encoding generation

### Phase 5: Integration & Testing
- [ ] End-to-end compilation
- [ ] Validation checks
- [ ] Example sequences

### Phase 6: Optimization (Later)
- [ ] Loop detection
- [ ] Memory optimization
- [ ] Instruction compression

---

## Key Design Questions to Answer

1. **Phase strategy**: How do you prefer to handle phase for constant pulses? (NCO vs waveforms?)
2. **Delay representation**: Can all delays be constants, or do you need register values for sweeps?
3. **Memory constraints**: How much waveform memory do you have?
4. **Deadtime model**: Is deadtime always `(aligned_length - actual_length)`, or more complex?
5. **Loop requirements**: Do you need compile-time loops or runtime-configurable ones?

---

## Example: Constant Amplitude Compilation

Input:
```python
play("halfpi", delay=100, phase=-90)
play("pi", delay=200, phase=0)
play("halfpi", delay=100, phase=90)
```

Expected flow:
1. **Stage 1**: Parse into 3 instructions with absolute timing
2. **Stage 2**: Plan waveforms: `halfpi_-90`, `pi_0`, `halfpi_+90`
3. **Stage 3**: Build timeline, validate delays are sufficient
4. **Stage 4**: Generate instructions: set phase register, select envelope, wait, play, repeat
5. **Stage 5**: Optimize (no loops here, but could consolidate similar waveforms)
6. **Stage 6**: Output: 3 waveforms + sequence of hardware instructions

