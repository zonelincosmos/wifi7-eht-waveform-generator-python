# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
GEN_EHT_SIG  Generate EHT-SIG for EHT SU PPDU (non-OFDMA).

Section 36.3.12.8
Pre-EHT modulated field: 4 us per OFDM symbol
Same per-20 MHz replication as L-SIG/U-SIG

For EHT SU (non-OFDMA): single content channel, duplicated per 20 MHz
Structure: Common field (20 bits, Table 36-36) +
           User Specific field (22 bits, Table 36-40) = 42 data bits
  + CRC4 (over 42 bits) + Tail (6 bits) = 52 bits total
  BCC R=1/2 -> 104 coded bits -> 52 per OFDM symbol
  Mapped to 52 data SC per 20 MHz (same structure as U-SIG)
"""

import numpy as np

from eht_constants import eht_constants
from utils.crc4 import crc4_usig


# =========================================================================
# Helper: de2bi(val, n, 'right-msb') -- LSB-first bit vector
# =========================================================================
def _de2bi(val, n_bits):
    """Convert integer to LSB-first bit vector (MATLAB de2bi 'right-msb')."""
    bits = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        bits[i] = val & 1
        val >>= 1
    return bits


# =========================================================================
# BCC rate-1/2 encoder (Section 17.3.5.5)
# =========================================================================
def _bcc_encoder(data):
    """Rate-1/2 binary convolutional encoder."""
    data = np.asarray(data, dtype=np.int8).ravel()
    N = len(data)
    g0 = np.array([1, 0, 1, 1, 0, 1, 1], dtype=np.int8)
    g1 = np.array([1, 1, 1, 1, 0, 0, 1], dtype=np.int8)
    K = 7
    sr = np.zeros(K, dtype=np.int8)
    encoded = np.zeros(2 * N, dtype=np.int8)
    for i in range(N):
        sr = np.concatenate([[data[i]], sr[:-1]])
        out_A = np.sum(sr * g0) % 2
        out_B = np.sum(sr * g1) % 2
        encoded[2 * i] = out_A
        encoded[2 * i + 1] = out_B
    return encoded


# =========================================================================
# HE-SIG-A interleaver for EHT-SIG (Section 27.3.12.8)
# =========================================================================
def _hesiga_interleave(bits, N_CBPS, N_BPSCS):
    """Interleave coded bits per HE-SIG-A interleaver (N_COL=13)."""
    N_COL = 13
    N_ROW = N_CBPS // N_COL
    s = max(N_BPSCS // 2, 1)
    k = np.arange(N_CBPS)
    i_p = N_ROW * (k % N_COL) + k // N_COL
    j_p = s * (i_p // s) + (i_p + N_CBPS - N_COL * i_p // N_CBPS) % s
    out = np.zeros(N_CBPS, dtype=bits.dtype)
    for idx in range(N_CBPS):
        out[j_p[idx]] = bits[idx]
    return out


# =========================================================================
# BPSK constellation mapping
# =========================================================================
def _constellation_map_bpsk(bits):
    """BPSK: b=0 -> -1, b=1 -> +1."""
    return (2.0 * np.asarray(bits, dtype=np.float64) - 1.0).astype(np.complex128)


def gen_eht_sig(cfg):
    """Generate EHT-SIG for EHT SU PPDU.

    Parameters
    ----------
    cfg : dict
        Configuration from eht_config().

    Returns
    -------
    numpy.ndarray
        Time-domain samples (complex128).
    """
    c = eht_constants(cfg['BW'])
    NFFT = cfg['NFFT']
    Fs = cfg['Fs']
    gamma = c['gamma_preEHT']

    # ==================================================================
    # Common field (20 bits) -- Table 36-36
    # ==================================================================
    # B0-B3 Spatial Reuse (4 bits)
    assert 0 <= cfg['SpatialReuse'] <= 15, 'SpatialReuse must be 0..15'
    spatial_reuse = _de2bi(cfg['SpatialReuse'], 4)

    # B4-B5 GI+LTF Size (2 bits, Table 36-36)
    ltf_type = cfg['EHT_LTF_Type']
    gi = cfg['GI']
    if ltf_type == 2 and gi == 0.8:
        gi_val = 0
    elif ltf_type == 2 and gi == 1.6:
        gi_val = 1
    elif ltf_type == 4 and gi == 0.8:
        gi_val = 2
    elif ltf_type == 4 and gi == 3.2:
        gi_val = 3
    else:
        raise ValueError('Invalid (EHT_LTF_Type, GI) combo')
    gi_ltf = _de2bi(gi_val, 2)

    # B6-B8 Number Of EHT-LTF Symbols (3 bits, non-linear mapping)
    ltf_count_map = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}
    n_ltf_enc = ltf_count_map[cfg['N_EHT_LTF']]
    n_ltf_bits = _de2bi(n_ltf_enc, 3)

    # B9 LDPC Extra Symbol Segment
    if (cfg['Coding'] == 'LDPC' and 'ldpc' in cfg
            and cfg['ldpc']['has_extra_symbol']):
        ldpc_extra = np.array([1], dtype=np.int8)
    else:
        ldpc_extra = np.array([0], dtype=np.int8)

    # B10-B11 Pre-FEC Padding Factor
    if cfg['a'] == 4:
        pre_fec_pad_val = 0
    else:
        pre_fec_pad_val = cfg['a']
    pre_fec_pad = _de2bi(pre_fec_pad_val, 2)

    # B12 PE Disambiguity
    pe_disamb = np.array([cfg['PE_Disambiguity']], dtype=np.int8)

    # B13-B16 Disregard (4 bits, all 1s)
    disregard = np.ones(4, dtype=np.int8)

    # B17-B19 Number Of Non-OFDMA Users (3 bits, 0 for SU)
    n_non_ofdma_users = _de2bi(0, 3)

    common_field = np.concatenate([
        spatial_reuse, gi_ltf, n_ltf_bits, ldpc_extra,
        pre_fec_pad, pe_disamb, disregard, n_non_ofdma_users
    ])
    assert len(common_field) == 20, \
        f'Common field must be 20 bits, got {len(common_field)}'

    # ==================================================================
    # User field (22 bits) -- Table 36-40
    # ==================================================================
    # B0-B10 STA-ID (11 bits)
    assert 0 <= cfg['STA_ID'] <= 2047, 'STA_ID must be 0..2047'
    sta_id = _de2bi(cfg['STA_ID'], 11)

    # B11-B14 MCS (4 bits)
    mcs_bits = _de2bi(cfg['MCS'], 4)

    # B15 Reserved (set to 1)
    reserved = np.array([1], dtype=np.int8)

    # B16-B19 NSS-1 (4 bits)
    nss_bits = _de2bi(cfg['NSS'] - 1, 4)

    # B20 Beamformed
    beamformed = np.array([cfg['Beamformed']], dtype=np.int8)

    # B21 Coding: 0 = BCC, 1 = LDPC
    coding_bit = np.array([1 if cfg['Coding'] == 'LDPC' else 0], dtype=np.int8)

    user_field = np.concatenate([
        sta_id, mcs_bits, reserved, nss_bits, beamformed, coding_bit
    ])
    assert len(user_field) == 22, \
        f'User field must be 22 bits, got {len(user_field)}'

    # ==================================================================
    # Combine and add CRC + Tail
    # ==================================================================
    data_bits = np.concatenate([common_field, user_field])
    assert len(data_bits) == 42, 'EHT-SIG data must be 42 bits'

    crc = crc4_usig(data_bits)
    tail = np.zeros(6, dtype=np.int8)

    all_bits = np.concatenate([data_bits, crc, tail])
    assert len(all_bits) == 52, 'EHT-SIG total must be 52 bits'

    # ==================================================================
    # BCC encode R=1/2 -> 104 coded bits
    # ==================================================================
    encoded = _bcc_encoder(all_bits)
    assert len(encoded) == 104, 'BCC output must be 104 bits'

    # ==================================================================
    # Split into N_EHT_SIG OFDM symbols
    # ==================================================================
    sym_len = round(cfg['T_DFT_preEHT'] * Fs)
    CP_len = cfg['CP_preEHT']

    # Data and pilot subcarrier indices
    data_sc = np.concatenate([
        np.arange(-28, -21),   # -28:-22
        np.arange(-20, -7),    # -20:-8
        np.arange(-6, 0),      # -6:-1
        np.arange(1, 7),       # 1:6
        np.arange(8, 21),      # 8:20
        np.arange(22, 29),     # 22:28
    ])
    pilot_sc = np.array([-21, -7, 7, 21], dtype=np.int32)

    td_parts = []
    for sym_idx in range(cfg['N_EHT_SIG']):
        # Extract 52 coded bits for this symbol
        bit_start = sym_idx * 52
        bit_end = min((sym_idx + 1) * 52, len(encoded))
        sym_coded = encoded[bit_start:bit_end].copy()
        if len(sym_coded) < 52:
            sym_coded = np.concatenate([
                sym_coded, np.zeros(52 - len(sym_coded), dtype=np.int8)
            ])

        # Only MCS 0 (BPSK 1/2) supported
        assert cfg['EHT_SIG_MCS'] == 0, \
            f'gen_eht_sig supports EHT_SIG_MCS=0 only, got {cfg["EHT_SIG_MCS"]}'

        # Interleave
        sym_coded = _hesiga_interleave(sym_coded, 52, 1)

        # BPSK constellation mapping
        d_k = _constellation_map_bpsk(sym_coded)

        # Phase rotation Gamma per Eq. 36-24
        if cfg['EHT_SIG_MCS'] == 3:
            gamma_m = np.ones(52, dtype=np.float64)
        else:
            gamma_m = np.ones(52, dtype=np.float64)
            # Positions 26..51 (0-indexed): alternating sign
            for ii in range(26, 52):
                m_prime = ii   # 0-indexed M' in [26, 51]
                gamma_m[ii] = (-1) ** m_prime

        # Pilot polarity: continuing from U-SIG
        # L-SIG=p0, RL-SIG=p1, U-SIG-1=p2, U-SIG-2=p3, EHT-SIG-1=p4, ...
        p_idx = 4 + sym_idx   # p4 for first EHT-SIG symbol
        p_val = c['pilot_polarity'][p_idx % len(c['pilot_polarity'])]
        pilot_vals = p_val * np.array([1, 1, 1, -1], dtype=np.float64)

        # Build per-20 MHz structure
        legacy_sc = np.zeros(57, dtype=np.complex128)
        for i in range(52):
            k = data_sc[i]
            legacy_sc[k + 28] = gamma_m[i] * d_k[i]
        for i in range(4):
            k = pilot_sc[i]
            legacy_sc[k + 28] = pilot_vals[i]

        # Replicate to full BW
        freq = np.zeros(NFFT, dtype=np.complex128)
        for seg in range(c['N_20MHz']):
            Ks = c['K_Shift'][seg]
            g = gamma[seg]
            for k_local in range(-28, 29):
                val = legacy_sc[k_local + 28]
                if val == 0:
                    continue
                fft_idx = (k_local - Ks) * c['SC_RATIO']
                fft_bin = fft_idx % NFFT
                freq[fft_bin] = g * val

        # Normalization per Eq. 36-24
        assert cfg['NSS'] == 1, \
            f'gen_eht_sig: Eq 36-24 assumes N_TX=1; got NSS={cfg["NSS"]}'
        freq = freq / np.sqrt(c['N_tone_LSIG'])

        # IFFT + CP
        td_full = np.fft.ifft(freq, NFFT) * np.sqrt(NFFT)
        one_sym = td_full[:sym_len]
        cp = one_sym[-CP_len:]
        td_parts.append(cp)
        td_parts.append(one_sym)

    return np.concatenate(td_parts)
