# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
802.11be EHT PHY DATA scrambler.

Polynomial: S(x) = x^11 + x^9 + 1  (Section 36.3.13.2, Eq. 36-46)
11-bit LFSR per Figure 36-50 (IEEE 802.11be-2024 p.813).

The shift register is clocked once per data bit.  At each clock:
  1. Output = x11 (rightmost register cell in Figure 36-50)
  2. Feedback = x9 XOR x11
  3. Shift right: x11<-x10, x10<-x9, ..., x2<-x1, x1<-feedback
  4. Scrambled bit = data(n) XOR output

The init_state sets the register to [x1, x2, ..., x11] BEFORE the
first clock.
"""

import numpy as np


def eht_scrambler(data, init_state):
    """802.11be EHT PHY DATA scrambler (11-bit Fibonacci LFSR).

    Parameters
    ----------
    data : array_like
        Binary vector (row or column) of 0/1 values.
    init_state : int or array_like
        If int (1..2047): 11-bit LFSR seed per Section 36.3.13.2.
            LSB of the integer maps to x1 (leftmost cell).
        If array_like: 11-element binary vector [x1, x2, ..., x11].

    Returns
    -------
    scrambled : numpy.ndarray
        Scrambled binary vector (same length as data), dtype ``np.int8``.
    state_out : numpy.ndarray
        Final 11-bit LFSR state [x1, x2, ..., x11], dtype ``np.int8``.
    """
    data = np.asarray(data, dtype=np.int8).ravel()

    # Convert integer init to binary vector [x1, x2, ..., x11]
    # Per spec Figure 36-50: x1 is the leftmost cell (feedback input),
    # x11 is the rightmost cell (output). LSB of the integer maps to x1.
    # MATLAB: de2bi(init_state, 11, 'right-msb') produces
    #   [LSB, ..., MSB] which is [x1, x2, ..., x11].
    # Accept either a scalar integer seed or an 11-element bit vector.
    # np.ndim(x) == 0 catches Python ints, numpy 0-d arrays and numpy
    # scalar types (e.g. np.int64) uniformly.
    is_scalar = np.ndim(init_state) == 0
    if is_scalar:
        val = int(init_state)
        if val < 1 or val > 2047:
            raise ValueError(
                "init_state must be 1..2047 (nonzero 11-bit LFSR seed)"
            )
        reg = np.zeros(11, dtype=np.int8)
        # 'right-msb' in MATLAB de2bi: index 0 = LSB = x1
        for i in range(11):
            reg[i] = (val >> i) & 1
    else:
        reg = np.asarray(init_state, dtype=np.int8).ravel().copy()
        if len(reg) != 11:
            raise ValueError('init_state must be 11 bits')

    N = len(data)
    scrambled = np.zeros(N, dtype=np.int8)

    # Shift-register loop per Figure 36-50:
    #   reg = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
    #   Output = reg[10] = x11           (0-indexed: position 10)
    #   Feedback = xor(reg[8], reg[10])   (x9 XOR x11, 0-indexed: 8 and 10)
    #   Shift: reg = [feedback, reg[0:10]]
    for n in range(N):
        output = reg[10]                                # x11 -> output
        fb = reg[8] ^ reg[10]                           # x9 XOR x11
        scrambled[n] = int(data[n]) ^ int(output)
        reg = np.concatenate([np.array([fb], dtype=np.int8), reg[0:10]])

    state_out = reg
    return scrambled, state_out
