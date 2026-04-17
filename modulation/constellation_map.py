# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
Gray-coded QAM constellation mapping per IEEE 802.11be-2024.

    symbols = constellation_map(bits, N_BPSCS)

Supports BPSK (1), QPSK (2), 16-QAM (4), 64-QAM (6), 256-QAM (8),
1024-QAM (10), 4096-QAM (12).

Section 27.3.12.9 (BPSK through 1024-QAM), Table 36-51 (4096-QAM).

For each I/Q half (independently):
  b0 = sign bit: 0 -> negative, 1 -> positive
  b1..b_{n-1} = magnitude bits, Gray-coded
  level = sign * (2 * gray_decode(b1..b_{n-1}) + 1)   ... but inverted:
  magnitude = 2 * (N_half - 1 - gray_decoded_index) + 1
"""

import numpy as np


def _qam_half_map(bits):
    """Map n bits to a PAM level per IEEE 802.11 Table 36-51.

    Parameters
    ----------
    bits : numpy.ndarray
        1-D array of 0/1 values.  ``bits[0]`` is the MSB (sign bit).

    Returns
    -------
    int
        Signed integer PAM level.

    Notes
    -----
    b0 = MSB = sign: 0 -> negative, 1 -> positive
    b1..b_{n-1} = magnitude (Gray-coded)

    Gray decode the magnitude bits, then:
      magnitude = 2 * (N_half - 1 - gray_decoded_index) + 1

    where N_half = 2^(n-1).
    """
    n = len(bits)
    sign_bit = int(bits[0])

    if n == 1:
        return 2 * sign_bit - 1

    # Magnitude bits: bits[1:]
    mag_bits = bits[1:]
    n_mag = len(mag_bits)

    # Gray-to-binary decode
    bin_bits = np.zeros(n_mag, dtype=np.int8)
    bin_bits[0] = int(mag_bits[0])
    for k in range(1, n_mag):
        bin_bits[k] = bin_bits[k - 1] ^ int(mag_bits[k])

    # Binary to decimal (MSB first)
    idx = 0
    for k in range(n_mag):
        idx = idx + int(bin_bits[k]) * (2 ** (n_mag - 1 - k))

    # Level: Gray index 0 -> max magnitude, index max -> magnitude 1
    N_half = 2 ** n_mag
    magnitude = 2 * (N_half - 1 - idx) + 1

    if sign_bit == 0:
        return -magnitude
    else:
        return magnitude


def constellation_map(bits, N_BPSCS):
    """Gray-coded QAM constellation mapping per IEEE 802.11.

    Parameters
    ----------
    bits : array_like
        Binary vector of 0/1 values.  Length must be a multiple of
        *N_BPSCS*.
    N_BPSCS : int
        Number of coded bits per subcarrier per spatial stream.
        1=BPSK, 2=QPSK, 4=16-QAM, 6=64-QAM, 8=256-QAM,
        10=1024-QAM, 12=4096-QAM.

    Returns
    -------
    numpy.ndarray
        1-D complex array of QAM symbols, length ``len(bits) / N_BPSCS``.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()
    N_bits = len(bits)

    if N_bits % N_BPSCS != 0:
        raise ValueError(
            f"Number of bits ({N_bits}) must be a multiple of "
            f"N_BPSCS ({N_BPSCS})"
        )

    N_sym = N_bits // N_BPSCS

    if N_BPSCS == 1:
        # BPSK: K_mod = 1, b=0 -> -1, b=1 -> +1
        symbols = (2.0 * bits.astype(np.float64) - 1.0) + 0j
        return symbols

    if N_BPSCS == 2:
        # QPSK: K_mod = 1/sqrt(2)
        K_mod = 1.0 / np.sqrt(2.0)
        symbols = np.zeros(N_sym, dtype=np.complex128)
        for i in range(N_sym):
            b = bits[i * 2: i * 2 + 2]
            I_val = 2.0 * b[0] - 1.0
            Q_val = 2.0 * b[1] - 1.0
            symbols[i] = K_mod * (I_val + 1j * Q_val)
        return symbols

    # 16-QAM through 4096-QAM
    n_half = N_BPSCS // 2  # bits per I or Q dimension

    # K_mod normalization factors per spec
    K_mod_table = {
        4:  1.0 / np.sqrt(10.0),     # 16-QAM
        6:  1.0 / np.sqrt(42.0),     # 64-QAM
        8:  1.0 / np.sqrt(170.0),    # 256-QAM
        10: 1.0 / np.sqrt(682.0),    # 1024-QAM
        12: 1.0 / np.sqrt(2730.0),   # 4096-QAM (Table 36-51)
    }
    if N_BPSCS not in K_mod_table:
        raise ValueError(f"Unsupported N_BPSCS={N_BPSCS}")

    K_mod = K_mod_table[N_BPSCS]

    symbols = np.zeros(N_sym, dtype=np.complex128)
    for i in range(N_sym):
        b = bits[i * N_BPSCS: (i + 1) * N_BPSCS]
        b_I = b[:n_half]
        b_Q = b[n_half:]

        I_val = _qam_half_map(b_I)
        Q_val = _qam_half_map(b_Q)

        symbols[i] = K_mod * (I_val + 1j * Q_val)

    return symbols
