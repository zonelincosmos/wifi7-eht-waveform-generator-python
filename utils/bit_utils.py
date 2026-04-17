# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
Bit manipulation utilities for 802.11be EHT waveform generation.

Conversions between integers, byte arrays, and bit vectors (numpy arrays
of 0/1 with dtype=np.int8).  All bit vectors use MSB-first ordering,
matching MATLAB ``de2bi(x, n, 'left-msb')``.
"""

import numpy as np


def int2bits(val, n_bits):
    """Convert a non-negative integer to an MSB-first bit vector.

    Equivalent to MATLAB ``de2bi(val, n_bits, 'left-msb')``.

    Parameters
    ----------
    val : int
        Non-negative integer value to convert.
    n_bits : int
        Number of output bits.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(n_bits,)`` with dtype ``np.int8``,
        MSB at index 0.
    """
    if val < 0:
        raise ValueError(f"val must be non-negative, got {val}")
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1, got {n_bits}")
    bits = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        bits[n_bits - 1 - i] = val & 1
        val >>= 1
    if val > 0:
        raise ValueError(
            f"Value requires more than {n_bits} bits to represent"
        )
    return bits


def bits2int(bits):
    """Convert an MSB-first bit vector to a non-negative integer.

    Parameters
    ----------
    bits : array_like
        1-D sequence of 0/1 values, MSB at index 0.

    Returns
    -------
    int
        Decoded integer value.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def bytes2bits(byte_array):
    """Convert a byte array to an MSB-first bit vector.

    Each byte is expanded to 8 bits with the MSB first, and the bytes
    are concatenated in order.

    Parameters
    ----------
    byte_array : array_like
        1-D sequence of uint8 values.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(8 * len(byte_array),)`` with dtype ``np.int8``.
    """
    byte_array = np.asarray(byte_array, dtype=np.uint8).ravel()
    n_bytes = len(byte_array)
    bits = np.zeros(n_bytes * 8, dtype=np.int8)
    for i, byte_val in enumerate(byte_array):
        for j in range(8):
            bits[i * 8 + j] = (byte_val >> (7 - j)) & 1
    return bits


def bits2bytes(bits):
    """Convert an MSB-first bit vector to a byte array.

    The bit vector length must be a multiple of 8.

    Parameters
    ----------
    bits : array_like
        1-D sequence of 0/1 values whose length is a multiple of 8.

    Returns
    -------
    numpy.ndarray
        1-D array of uint8 values.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) % 8 != 0:
        raise ValueError(
            f"Bit vector length ({len(bits)}) must be a multiple of 8"
        )
    n_bytes = len(bits) // 8
    result = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i * 8 + j])
        result[i] = val
    return result
