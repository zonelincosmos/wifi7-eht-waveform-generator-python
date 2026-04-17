# wifi7-eht-waveform-generator-python

**IEEE 802.11be EHT SU waveform generator (Wi-Fi 7) in Python.**

Generates a complete 802.11be (Wi-Fi 7) Extremely-High-Throughput
single-user PPDU at baseband, as a 1-D complex-valued IQ array.

```python
from eht_waveform_gen import eht_waveform_gen

waveform, cfg, psdu = eht_waveform_gen(
    BW=320, MCS=13, GI=3.2, LTFType=4, PayloadBytes=5000,
)
# waveform: complex128 numpy array, 320 MHz sample rate, 30 720 samples
# cfg:      dict with every derived PHY parameter
# psdu:     uint8 numpy array, the A-MPDU bytes fed to the Data field
```
<img width="1837" height="614" alt="image" src="https://github.com/user-attachments/assets/33bde4db-069b-4bc2-8a60-a2c7b7b89514" />


## Table of contents

- [Feature highlights](#feature-highlights)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Supported configurations](#supported-configurations)
- [Full parameter reference](#full-parameter-reference)
- [Spec references](#spec-references)
- [Licence](#licence)

## Feature highlights

- **All five bandwidths**: 20, 40, 80, 160, 320 MHz.
- **All fourteen MCS**: MCS 0 (BPSK 1/2) through MCS 13 (4096-QAM 5/6).
- **Pure Python**: only `numpy` is required at runtime; `matplotlib`
  is optional and only used by `run_example.py` for plots.

## Installation

```sh
git clone <repo-url>
cd wifi7-eht-waveform-generator-python
pip install -r requirements.txt
```

Requires Python 3.9 or newer (tested on 3.9 and 3.14).  Dependencies:

| Package      | Min. version | Used for                     |
| ------------ | -----------: | ---------------------------- |
| `numpy`      |         1.20 | everything                   |
| `matplotlib` |          3.3 | `run_example.py` plots only  |

No C / Fortran build step.  No platform-specific wheels.  Runs the
same on Windows / macOS / Linux.

## Quick start

### Minimal example

```python
from eht_waveform_gen import eht_waveform_gen

wf, cfg, psdu = eht_waveform_gen()
print(f'{len(wf)} samples at {cfg["Fs"]/1e6:.0f} MHz')
# Default: BW=80, MCS=7, GI=0.8, LTF=2, 1000-byte payload
```

### Supply your own user payload

```python
import numpy as np
from eht_waveform_gen import eht_waveform_gen

user_bytes = np.frombuffer(b'Hello Wi-Fi 7 world!', dtype=np.uint8)

wf, cfg, psdu = eht_waveform_gen(
    BW=80, MCS=7, GI=0.8, LTFType=2,
    PayloadBytes=1000,
    PSDU=user_bytes,       # truncated / zero-padded to fit
)
```

Your `user_bytes` becomes the MAC payload of a single MPDU inside an
A-MPDU; everything else (delimiter, MAC header, FCS, EOF padding) is
added for you so the waveform is decodable by a compliant receiver.

### Run the bundled example script

```sh
python run_example.py
```

Generates the primary target config (BW = 320 MHz, MCS = 13,
4096-QAM 5/6, GI = 3.2 us, 4x EHT-LTF, 5000-byte PSDU), saves

- `eht_waveform.npz` - the complex IQ waveform and framed PSDU bytes
- `eht_waveform_time.png` - time-domain magnitude with every PPDU
  field shaded and labelled
- `eht_waveform_psd.png` - Welch-style power spectral density
- `eht_waveform_constellation.png` - Data-field constellation after
  CP removal and DFT (for MCS 13 you will see the full 4096-QAM grid)

and prints a field-by-field duration summary.  Works on a headless
machine (falls back to the Agg matplotlib backend automatically).

## Supported configurations

| Parameter       | Values                                |
| --------------- | ------------------------------------- |
| Bandwidth       | 20, 40, 80, 160, 320 MHz              |
| MCS             | 0..13 (BPSK 1/2 through 4096-QAM 5/6) |
| Guard interval  | 0.8, 1.6, 3.2 us                      |
| EHT-LTF type    | 2x, 4x                                |
| Coding          | BCC (BW = 20, MCS <= 9), else LDPC    |
| Spatial streams | 1 (SISO)                              |
| PayloadBytes    | 38 .. 16 383 bytes per subframe       |

Valid `(LTFType, GI)` pairs per IEEE 802.11be-2024 Table 36-36:
`(2, 0.8)`, `(2, 1.6)`, `(4, 0.8)`, `(4, 3.2)`.

## Full parameter reference

`eht_waveform_gen` accepts the following keyword arguments (defaults
in parentheses):

| Argument               | Meaning                                             |
| ---------------------- | --------------------------------------------------- |
| `BW`                   | Channel bandwidth, MHz                              |
| `MCS`                  | EHT modulation-coding scheme index                  |
| `GI`                   | Data-field guard interval, us                       |
| `LTFType`              | EHT-LTF density: `2` or `4`, or `'auto'` from GI    |
| `PayloadBytes`         | APEP_LENGTH in bytes (PSDU length field)            |
| `PSDU`                 | Optional uint8 user payload; framed via A-MPDU      |
| `ScramblerInit`        | 11-bit scrambler seed, 1..2047                      |
| `Coding`               | `'BCC'`, `'LDPC'`, or `'auto'`                      |
| `NominalPacketPadding` | PE duration in us: 0, 8, 16, or 20                  |
| `UL_DL`                | U-SIG-1 B6: 0 = downlink, 1 = uplink                |
| `BSS_Color`            | U-SIG-1 B7..B12, 0..63                              |
| `TXOP`                 | U-SIG-1 B13..B19, 0..127                            |
| `STA_ID`               | EHT-SIG user field B0..B10, 0..2047                 |
| `Beamformed`           | EHT-SIG user field B20: 0 or 1                      |
| `SpatialReuse`         | EHT-SIG common B0..B3, 0..15                        |
| `EHT_SIG_MCS`          | U-SIG-2 B9..B10, 0..3 (BPSK only implemented)       |
| `verbose`              | Print progress and summary lines                    |

## Spec references

- IEEE Std 802.11be-2024, *Enhancements for Extremely High Throughput (EHT)*
- IEEE Std 802.11-2024, Sections 9 (MAC), 10 (A-MPDU), 17 (legacy
  preamble), 19 (LDPC), 27 (HE), 36 (EHT)

## Licence

MIT - see [LICENSE](LICENSE).
