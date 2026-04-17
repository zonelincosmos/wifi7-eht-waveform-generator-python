# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
IEEE 802.11 MAC Frame Check Sequence (CRC-32) computation.

Port of eht_waveform_gen/crc32_fcs.m.

Polynomial: 0x04C11DB7 (reflected form 0xEDB88320)
Init:       0xFFFFFFFF
Final XOR:  0xFFFFFFFF
Output:     4 uint8 bytes, LSByte first per IEEE 802.11
"""

import numpy as np


# Precomputed 256-entry CRC-32 table (reflected polynomial 0xEDB88320).
_POLY = 0xEDB88320
_TABLE = np.empty(256, dtype=np.uint32)
for _i in range(256):
    _c = _i
    for _ in range(8):
        if _c & 1:
            _c = (_c >> 1) ^ _POLY
        else:
            _c >>= 1
    _TABLE[_i] = _c


def crc32_fcs(data_bytes):
    """Compute 4-byte IEEE 802.11 FCS for a byte sequence.

    Parameters
    ----------
    data_bytes : array-like of uint8
        MPDU octets excluding FCS.

    Returns
    -------
    numpy.ndarray
        1-D uint8 array of length 4 (LSByte first).
    """
    data = np.asarray(data_bytes, dtype=np.uint8).ravel()
    crc = 0xFFFFFFFF
    for b in data:
        idx = (crc ^ int(b)) & 0xFF
        crc = (crc >> 8) ^ int(_TABLE[idx])
    crc ^= 0xFFFFFFFF
    return np.array([
        crc        & 0xFF,
        (crc >>  8) & 0xFF,
        (crc >> 16) & 0xFF,
        (crc >> 24) & 0xFF,
    ], dtype=np.uint8)
