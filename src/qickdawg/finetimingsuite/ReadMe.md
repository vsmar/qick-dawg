# Fine Timing Suite

Subnanosecond pulse sequences for NV center spin control (~200 ps resolution).

## Available Sequences

| Sequence | Purpose | Key Sweep |
|----------|---------|-----------|
| CPMG-XY | Coherence & nuclear spectroscopy | τ (inter-pulse delay) |
| Ramsey | T2* dephasing time | τ (free precession) |
| Rabi | Pulse calibration | MW duration |
| T1 | Spin relaxation | MW-to-readout delay |
| Hahn Echo | Spin echo decay | τ |
| Bootstrap | Pulse shape characterization | Bootstrap experiment # |
| PODMR | Frequency sweep with pulses | MW frequency |
| Counting Duration | ADC integration duration | — |

## Key Features

- **Sub-nanosecond timing**: Fine-time sample (ftsamp) unit offsetting
- **Hardware efficient**: FPGA register arithmetic, no reinitialization
- **Configurable readout**: Optional MW-off reference readouts for normalization
- **Shared infrastructure**: `ReadoutHelpers` provides common spin init/readout logic

## Quick Start

See `CPMG-XY` for comprehensive documentation of the fine-timing approach,
as well as tips for structuring more complicated patterned sequences (phase-cycling).