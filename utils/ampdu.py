# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
Build A-MPDU formatted PSDU for EHT PPDU (S-MPDU).

Port of eht_waveform_gen/build_ampdu.m.

Structure:
    [Delimiter(4)] [MAC Hdr(26) + Payload + FCS(4)] [Pad(0-3)] [EOF delims...]
"""

import numpy as np

from utils.crc32 import crc32_fcs


def _build_mac_header():
    """Build minimal QoS Data MAC header (26 bytes).

    Matches `build_mac_header()` inner function in build_ampdu.m.
    FC = 0x88, 0x00 (QoS Data, ToDS=0, FromDS=0).
    """
    fc       = np.array([0x88, 0x00], dtype=np.uint8)
    duration = np.array([0x00, 0x00], dtype=np.uint8)
    addr1    = np.array([0xFF] * 6,    dtype=np.uint8)
    addr2    = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    addr3    = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    seq_ctl  = np.array([0x00, 0x00], dtype=np.uint8)
    qos_ctl  = np.array([0x00, 0x00], dtype=np.uint8)
    hdr = np.concatenate([fc, duration, addr1, addr2, addr3, seq_ctl, qos_ctl])
    assert len(hdr) == 26, f"MAC header must be 26 bytes, got {len(hdr)}"
    return hdr


def _crc8_delimiter(bits):
    """CRC-8 for A-MPDU delimiter.

    Polynomial G(x) = x^8 + x^2 + x + 1 (0x07), init 0xFF, final XOR 0xFF.
    Input: iterable of bits (LSB first).
    """
    poly = 7
    m = 255
    for bit in bits:
        fb = ((m >> 7) ^ int(bit)) & 1
        m = (m << 1) & 0xFF
        if fb:
            m ^= poly
    return m ^ 0xFF


def _build_delimiter(mpdu_length, eof):
    """Build A-MPDU delimiter (4 bytes) per IEEE 802.11-2024 Figure 9-1329."""
    if mpdu_length > 16383:
        raise ValueError(
            f"MPDU length {mpdu_length} exceeds 14-bit maximum (16383)"
        )

    # 16-bit field: B0=EOF, B1=Reserved(0), B2-B15=MPDU Length
    word16 = 0
    if eof:
        word16 |= 1
    word16 |= (mpdu_length & 0x3FFF) << 2

    bits16 = [(word16 >> b) & 1 for b in range(16)]
    crc = _crc8_delimiter(bits16)

    byte0 = word16 & 0xFF
    byte1 = (word16 >> 8) & 0xFF
    byte2 = crc & 0xFF
    byte3 = 0x4E
    return np.array([byte0, byte1, byte2, byte3], dtype=np.uint8)


def _build_eof_delimiter():
    """EOF delimiter: MPDU Length = 0, EOF = 1."""
    return _build_delimiter(0, 1)


def build_ampdu(user_payload, psdu_length):
    """Build A-MPDU formatted PSDU for EHT PPDU.

    Parameters
    ----------
    user_payload : array-like of uint8
        User data bytes.
    psdu_length : int
        Total PSDU length in bytes (= PayloadBytes from eht_config).

    Returns
    -------
    numpy.ndarray
        uint8 1-D array of exactly ``psdu_length`` bytes.
    """
    user_payload = np.asarray(user_payload, dtype=np.uint8).ravel()

    # MPDU = MAC header + payload + FCS
    mac_hdr = _build_mac_header()
    mpdu_body = np.concatenate([mac_hdr, user_payload])
    fcs = crc32_fcs(mpdu_body)
    mpdu = np.concatenate([mpdu_body, fcs])
    mpdu_len = len(mpdu)

    # A-MPDU delimiter (S-MPDU: EOF=1)
    delimiter = _build_delimiter(mpdu_len, 1)

    # Subframe = delimiter + MPDU + pad to 4-byte boundary
    subframe_unpadded = np.concatenate([delimiter, mpdu])
    pad_needed = (4 - (len(subframe_unpadded) % 4)) % 4
    subframe = np.concatenate([
        subframe_unpadded,
        np.zeros(pad_needed, dtype=np.uint8),
    ])

    remaining = psdu_length - len(subframe)
    if remaining < 0:
        raise ValueError(
            f"build_ampdu: MPDU ({mpdu_len} bytes) + framing "
            f"({len(subframe)} bytes) exceeds psdu_length ({psdu_length}). "
            "Reduce payload or increase PayloadBytes."
        )

    eof_delim = _build_eof_delimiter()
    n_eof = remaining // 4
    if n_eof > 0:
        eof_padding = np.tile(eof_delim, n_eof)
    else:
        eof_padding = np.array([], dtype=np.uint8)

    leftover = remaining - n_eof * 4
    tail_pad = np.full(leftover, 0xFF, dtype=np.uint8)

    psdu = np.concatenate([subframe, eof_padding, tail_pad])
    assert len(psdu) == psdu_length, (
        f"build_ampdu: output {len(psdu)} != expected {psdu_length}"
    )
    return psdu
