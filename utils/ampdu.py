# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
Build A-MPDU formatted PSDU for EHT PPDU.

Supports both S-MPDU (single subframe, num_mpdus=1, the default) and
multi-MPDU A-MPDU (num_mpdus > 1) per IEEE 802.11-2024 Section 9.6 and
Section 10.12.

Structure for ``num_mpdus = N``:
    [Delim_1(EOF=0) + MAC_hdr + chunk_1 + FCS + pad(0-3)]   1st subframe
    [Delim_2(EOF=0) + MAC_hdr + chunk_2 + FCS + pad(0-3)]   2nd subframe
    ...
    [Delim_N(EOF=0) + MAC_hdr + chunk_N + FCS + pad(0-3)]   Nth subframe
    [EOF padding delim (Length=0, EOF=1)] x M               fills remainder
    [0xFF tail (0-3 bytes)]                                 final non-aligned tail

Per Section 10.12.7, EOF=0 is set on every real MPDU subframe and EOF=1
is reserved for the trailing zero-length padding delimiters.
"""

import numpy as np

from utils.crc32 import crc32_fcs


def _build_mac_header(seq_num=0):
    """Build the 3-address QoS Data MAC header (26 bytes).

    IEEE 802.11-2024 Section 9.3.2.1.  Layout (3-address, ToDS=1, FromDS=0,
    +HTC=0):

        FC(2) | Duration(2) | Addr1(6) | Addr2(6) | Addr3(6) | SeqCtl(2) | QoSCtl(2)

    Frame Control:
      Byte 0 (0x88) = Subtype[3:0] | Type[1:0] | ProtocolVersion[1:0]
                    = 1000 (QoS Data) | 10 (Data) | 00 (v0)
      Byte 1 (0x01) = Order|Protected|MoreData|PwrMgmt|Retry|MoreFrag|FromDS|ToDS
                    = 0|0|0|0|0|0|0|1  -> ToDS=1, all other flags=0

    Duration/ID:
      44 us NAV per Section 9.2.4.2 (16 us SIFS + 28 us legacy ACK at 24 Mbps).
      Little-endian 16-bit: 44 = 0x002C -> [0x2C, 0x00].

    Addresses (Table 9-60 for ToDS=1, FromDS=0):
      Addr1 = RA = BSSID
      Addr2 = TA = SA
      Addr3 = DA  (unicast locally-administered)

    Sequence Control:
      Fragment = 0 (bits 0-3), SeqNum = 12 bits (bits 4-15).

    Parameters
    ----------
    seq_num : int
        12-bit sequence number (0..4095).  Default 0.  A-MPDU subframes
        use consecutive sequence numbers per Section 10.3.2.14.2.

    Returns
    -------
    numpy.ndarray
        uint8 1-D array of exactly 26 bytes.
    """
    if not (0 <= seq_num <= 4095):
        raise ValueError(
            f"seq_num must be 0..4095 (12-bit). Got {seq_num}."
        )

    fc       = np.array([0x88, 0x01], dtype=np.uint8)            # QoS Data, ToDS=1
    duration = np.array([0x2C, 0x00], dtype=np.uint8)            # 44 us NAV (LE)
    addr1    = np.array([0x02, 0, 0, 0, 0, 0x01], dtype=np.uint8)  # BSSID
    addr2    = np.array([0x02, 0, 0, 0, 0, 0x02], dtype=np.uint8)  # SA
    addr3    = np.array([0x02, 0, 0, 0, 0, 0x03], dtype=np.uint8)  # DA

    seq_ctl_word = (seq_num & 0xFFF) << 4   # frag = 0 in low nibble
    seq_ctl = np.array(
        [seq_ctl_word & 0xFF, (seq_ctl_word >> 8) & 0xFF],
        dtype=np.uint8,
    )
    qos_ctl = np.array([0, 0], dtype=np.uint8)

    hdr = np.concatenate([fc, duration, addr1, addr2, addr3, seq_ctl, qos_ctl])
    assert len(hdr) == 26, f"MAC header must be 26 bytes, got {len(hdr)}"
    return hdr


def _crc8_delimiter(bits):
    """CRC-8 for the A-MPDU subframe delimiter.

    Polynomial G(x) = x^8 + x^2 + x + 1 (= 0x07), init 0xFF, final XOR 0xFF.
    Input is a bit list (LSB-first within the delimiter word).

    Output convention: 802.11 transmits CRC bits MSB-first -- bit 7 of the
    CRC register value is sent first and ends up at bit 0 of the on-air
    byte.  So the stored CRC byte is BIT-REVERSED relative to the standard
    CRC register value (a frequent source of mismatch with off-the-shelf
    CRC libraries that don't reflect the output byte).
    """
    poly = 0x07         # x^8 + x^2 + x + 1, with the implicit x^8 dropped
    m = 0xFF            # init all 1s
    for bit in bits:
        fb = ((m >> 7) ^ int(bit)) & 1   # MSB of register XOR input
        m = (m << 1) & 0xFF              # left-shift, keep 8 bits
        if fb:
            m ^= poly                    # XOR polynomial if feedback=1

    crc_reg = m ^ 0xFF                   # final inversion (standard CRC value)

    # Bit-reverse the byte for transmission order: CRC bit 7 (MSB) is sent
    # first, ending up at bit 0 of the stored byte.
    crc = 0
    for b in range(8):
        if (crc_reg >> b) & 1:
            crc |= 1 << (7 - b)
    return crc


def _build_delimiter(mpdu_length, eof):
    """Build an A-MPDU subframe delimiter (4 bytes).

    HE/EHT format per IEEE 802.11-2024 Section 9.6.2 / Figure 9-66:

        B0       EOF/Tag       (1 bit)
        B1-B3    Reserved      (3 bits, 0)
        B4-B15   MPDU Length   (12 bits, max 4095)
        B16-B23  Delimiter CRC-8 over B0-B15
        B24-B31  Delimiter Signature = 0x4E ('N')

    Note: 802.11n legacy used a 14-bit MPDU Length at B2-B15.  The HE/EHT
    amendment redefined this to 12-bit Length at B4-B15, inserting a 3-bit
    Reserved field at B1-B3.  This function uses the HE/EHT 12-bit layout.

    Parameters
    ----------
    mpdu_length : int
        MPDU length in bytes (0..4095).
    eof : int or bool
        EOF/Tag bit (1 = end-of-aggregation marker for padding delimiters,
        0 = real MPDU subframe).

    Returns
    -------
    numpy.ndarray
        uint8 1-D array of exactly 4 bytes.
    """
    if mpdu_length > 4095:
        raise ValueError(
            f"MPDU length {mpdu_length} exceeds 12-bit maximum (4095) for "
            "HE/EHT A-MPDU subframe delimiter.  Increase NumMPDUs to split "
            "the aggregated payload into multiple smaller subframes."
        )

    # 16-bit field: B0=EOF, B1-B3=Reserved(0), B4-B15=MPDU Length.
    word16 = 0
    if eof:
        word16 |= 1
    word16 |= (mpdu_length & 0xFFF) << 4

    bits16 = [(word16 >> b) & 1 for b in range(16)]
    crc = _crc8_delimiter(bits16)

    byte0 = word16 & 0xFF                 # B0-B7
    byte1 = (word16 >> 8) & 0xFF          # B8-B15
    byte2 = crc & 0xFF                    # B16-B23: CRC-8
    byte3 = 0x4E                          # B24-B31: Signature 'N'
    return np.array([byte0, byte1, byte2, byte3], dtype=np.uint8)


def build_ampdu(user_payload, psdu_length, num_mpdus=1):
    """Build A-MPDU formatted PSDU for EHT PPDU.

    Constructs ``num_mpdus`` real MPDU subframes, each 4-byte aligned, with
    consecutive sequence numbers 0..N-1, followed by EOF padding delimiters
    filling the remaining pre-FEC bytes per IEEE 802.11-2024 Section 10.12.7.

    Parameters
    ----------
    user_payload : array-like of uint8
        User data bytes.  Will be split across ``num_mpdus`` chunks: the
        first ``len(user_payload) % num_mpdus`` chunks get one extra byte,
        the rest get ``len(user_payload) // num_mpdus`` bytes.
    psdu_length : int
        Total PSDU length in bytes (= PayloadBytes from eht_config).
    num_mpdus : int, optional
        Number of real MPDU subframes (default 1 = S-MPDU).

    Returns
    -------
    numpy.ndarray
        uint8 1-D array of exactly ``psdu_length`` bytes.
    """
    if (not isinstance(num_mpdus, (int, np.integer))) or num_mpdus < 1:
        raise ValueError(
            f"num_mpdus must be a positive integer scalar. Got {num_mpdus}."
        )

    user_payload = np.asarray(user_payload, dtype=np.uint8).ravel()
    total_user = len(user_payload)

    # Split user_payload into num_mpdus chunks.  First chunk_rem chunks
    # get one extra byte; rest get floor(total_user / num_mpdus).  This
    # distributes the remainder so chunk sizes differ by at most 1 byte.
    chunk_base = total_user // num_mpdus
    chunk_rem  = total_user % num_mpdus

    subframes = []
    chunk_start = 0
    for i in range(num_mpdus):
        chunk_size = chunk_base + (1 if i < chunk_rem else 0)
        chunk = user_payload[chunk_start : chunk_start + chunk_size]
        chunk_start += chunk_size

        # MAC header with consecutive SeqNum (Section 10.3.2.14.2).
        mac_hdr = _build_mac_header(seq_num=i)

        # MPDU = MAC header + frame body + FCS.
        mpdu_body = np.concatenate([mac_hdr, chunk])
        fcs = crc32_fcs(mpdu_body)
        mpdu = np.concatenate([mpdu_body, fcs])
        mpdu_len = len(mpdu)

        # EOF=0 on every real MPDU subframe per Section 10.12.7.  EOF=1 is
        # reserved for the trailing zero-length padding delimiters below.
        delimiter = _build_delimiter(mpdu_len, eof=0)

        # Subframe = delimiter + MPDU, padded to 4-byte boundary.
        subframe_unpadded = np.concatenate([delimiter, mpdu])
        pad_needed = (4 - (len(subframe_unpadded) % 4)) % 4
        subframe = np.concatenate([
            subframe_unpadded,
            np.zeros(pad_needed, dtype=np.uint8),
        ])
        subframes.append(subframe)

    subframes_combined = (np.concatenate(subframes)
                          if subframes else np.array([], dtype=np.uint8))

    # Fill remaining PSDU bytes with EOF padding delimiters.
    remaining = psdu_length - len(subframes_combined)
    if remaining < 0:
        raise ValueError(
            f"build_ampdu: subframes total {len(subframes_combined)} bytes "
            f"exceed psdu_length {psdu_length}.  Reduce payload, increase "
            "PayloadBytes, or reduce num_mpdus."
        )

    # EOF delimiters are 4 bytes each (Length=0, EOF=1, CRC-8, Signature=0x4E).
    eof_delim = _build_delimiter(0, eof=1)
    n_eof = remaining // 4
    eof_padding = (np.tile(eof_delim, n_eof) if n_eof > 0
                   else np.array([], dtype=np.uint8))

    # Any remaining 0-3 bytes after EOF delimiters: fill with 0xFF.  EOF
    # padding is 4-byte aligned by spec; the leftover tail is implementation
    # defined.
    leftover = remaining - n_eof * 4
    tail_pad = np.full(leftover, 0xFF, dtype=np.uint8)

    psdu = np.concatenate([subframes_combined, eof_padding, tail_pad])
    assert len(psdu) == psdu_length, (
        f"build_ampdu: output {len(psdu)} != expected {psdu_length}"
    )
    return psdu
