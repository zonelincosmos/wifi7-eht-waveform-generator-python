# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
CRC-4 computation for U-SIG / EHT-SIG fields.

IEEE 802.11-2024, Section 27.3.11.7.3 (p.4227, Figure 27-24):
  8-bit LFSR with G(x) = x^8 + x^2 + x + 1
  Init = all 1's.  Feedback from c7 (MSB).  Left-shift.
  Output = top 4 bits [c7 c6 c5 c4], bit-inverted.

Spec test vector (p.4227):
  Input:  42 bits {1 1 0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0
                   0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 1 0}
  Output: {B7 B6 B5 B4} = {0 1 1 1}
"""

import numpy as np


def crc4_usig(bits):
    """Compute 4-bit CRC for U-SIG / EHT-SIG fields.

    Parameters
    ----------
    bits : array_like
        1-D binary vector (0/1 values).

    Returns
    -------
    numpy.ndarray
        4-element array of 0/1 (dtype ``np.int8``), representing
        [c7, c6, c5, c4] after bit inversion per Figure 27-24.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()

    # 8-bit register [c7, c6, c5, c4, c3, c2, c1, c0], init all 1's
    reg = np.ones(8, dtype=np.int8)

    for i in range(len(bits)):
        # Feedback from MSB (c7) XOR input
        fb = bits[i] ^ reg[0]

        # Left-shift: [c6, c5, c4, c3, c2, c1, c0, 0]
        reg = np.concatenate([reg[1:], np.array([0], dtype=np.int8)])

        # XOR taps for G(x) = x^8 + x^2 + x + 1:
        reg[5] = reg[5] ^ fb   # x^2 tap (c2 position)
        reg[6] = reg[6] ^ fb   # x^1 tap (c1 position)
        reg[7] = reg[7] ^ fb   # x^0 tap (c0 position)

    # Output: top 4 bits [c7, c6, c5, c4] through an inverter
    crc = 1 - reg[0:4]
    return crc
