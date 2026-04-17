# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
BCC (Binary Convolutional Code) rate-1/2 encoder.

Implements the rate-1/2 convolutional encoder per IEEE 802.11-2024
Section 17.3.5.5, with generator polynomials g0 = 133 (octal) and
g1 = 171 (octal), constraint length K = 7.

Output is interleaved [A0, B0, A1, B1, ...] (2x input length).
"""

import numpy as np


# Generator polynomials (MSB first)
_G0 = np.array([1, 0, 1, 1, 0, 1, 1], dtype=np.int8)  # 133 octal
_G1 = np.array([1, 1, 1, 1, 0, 0, 1], dtype=np.int8)  # 171 octal
_K = 7  # constraint length


def bcc_encoder(data):
    """Rate-1/2 binary convolutional encoder.

    Parameters
    ----------
    data : array_like
        Binary input vector (1-D).

    Returns
    -------
    numpy.ndarray
        Encoded binary vector of length ``2 * len(data)`` with
        dtype ``np.int8``, interleaved as [A0, B0, A1, B1, ...].
    """
    data = np.asarray(data, dtype=np.int8).ravel()
    n = len(data)

    # Shift register initialised to zeros
    sr = np.zeros(_K, dtype=np.int8)

    encoded = np.zeros(2 * n, dtype=np.int8)
    for i in range(n):
        # Shift in new bit (prepend to front, drop last)
        sr[1:] = sr[:-1]
        sr[0] = data[i]

        # Compute outputs via generator polynomials
        out_a = np.sum(sr * _G0) % 2
        out_b = np.sum(sr * _G1) % 2

        encoded[2 * i] = out_a
        encoded[2 * i + 1] = out_b

    return encoded
