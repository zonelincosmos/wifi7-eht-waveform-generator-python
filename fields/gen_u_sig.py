# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
GEN_U_SIG  Generate U-SIG field for EHT PPDU.

Section 36.3.12.7, Table 36-28
Duration: 8 us = 2 OFDM symbols (each 4 us: 0.8 us CP + 3.2 us)

U-SIG is pre-EHT modulated. Per Section 36.3.12.7.3:
  Each of U-SIG-1 and U-SIG-2 contains 26 bits.
  BCC encoded at R=1/2 -> 52 coded bits per symbol.
  Mapped to 52 data subcarriers per 20 MHz (HE-SIG-A structure).
  Interleaved per 27.3.12.8 (HE-SIG-A/B interleaver).

52 data SC: [-28:-22, -20:-8, -6:-1, 1:6, 8:20, 22:28]
4 pilot SC: [-21, -7, 7, 21]
"""

import numpy as np

from eht_constants import eht_constants
from utils.crc4 import crc4_usig


# =========================================================================
# Helper: de2bi(val, n, 'right-msb') -- LSB-first bit vector
# MATLAB 'right-msb' places MSB at the right and LSB at index 0.
# =========================================================================
def _de2bi(val, n_bits):
    """Convert integer to LSB-first bit vector (matches MATLAB de2bi 'right-msb')."""
    bits = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        bits[i] = val & 1
        val >>= 1
    return bits


# =========================================================================
# BCC rate-1/2 encoder (Section 17.3.5.5)
# g0 = 133 octal, g1 = 171 octal, K = 7
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
# HE-SIG-A / U-SIG interleaver (Section 27.3.12.8)
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


def gen_u_sig(cfg):
    """Generate U-SIG field for EHT PPDU.

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
    # U-SIG-1 bits (26 bits, Table 36-28)
    # Integer fields are transmitted LSB-first per spec 36.3.12.7.1.
    # ==================================================================
    # B0-B2 PHY Version Identifier: 0 for EHT
    phy_version = _de2bi(0, 3)
    # B3-B5 Bandwidth
    bw_val_map = {20: 0, 40: 1, 80: 2, 160: 3, 320: 4}
    bw_bits = _de2bi(bw_val_map[cfg['BW']], 3)
    # B6 UL/DL
    ul_dl = np.array([cfg['UL_DL']], dtype=np.int8)
    # B7-B12 BSS Color (6 bits)
    assert 0 <= cfg['BSS_Color'] <= 63, 'BSS_Color must be 0..63'
    bss_color = _de2bi(cfg['BSS_Color'], 6)
    # B13-B19 TXOP (7 bits)
    assert 0 <= cfg['TXOP'] <= 127, 'TXOP must be 0..127'
    txop = _de2bi(cfg['TXOP'], 7)
    # B20-B24 Disregard: 5 bits, set to all 1s
    disregard = np.ones(5, dtype=np.int8)
    # B25 Validate
    validate1 = np.array([1], dtype=np.int8)

    usig1_bits = np.concatenate([phy_version, bw_bits, ul_dl, bss_color,
                                  txop, disregard, validate1])
    assert len(usig1_bits) == 26, 'U-SIG-1 must be 26 bits'

    # ==================================================================
    # U-SIG-2 bits (26 bits: 16 data + 4 CRC + 6 tail)
    # ==================================================================
    # B0-B1 PPDU Type And Compression Mode = 1 (EHT SU)
    ppdu_type = _de2bi(1, 2)
    # B2 Validate
    validate2 = np.array([1], dtype=np.int8)
    # B3-B7 Punctured Channel Information (0 for non-punctured)
    punct_info = _de2bi(0, 5)
    # B8 Validate
    validate3 = np.array([1], dtype=np.int8)
    # B9-B10 EHT-SIG MCS
    eht_sig_mcs = _de2bi(cfg['EHT_SIG_MCS'], 2)
    # B11-B15 Number Of EHT-SIG Symbols field (5 bits)
    mcs_arr = c['eht_sig_mcs_table']['mcs']
    mcs_row = np.where(mcs_arr == cfg['EHT_SIG_MCS'])[0]
    assert len(mcs_row) > 0, f"EHT_SIG_MCS={cfg['EHT_SIG_MCS']} not in table"
    n_eht_sig_sym_val = int(c['eht_sig_mcs_table']['field_value'][mcs_row[0]])
    n_eht_sig_sym = _de2bi(n_eht_sig_sym_val, 5)
    assert n_eht_sig_sym_val == cfg['N_EHT_SIG'] - 1, \
        f"U-SIG N_EHT_SIG field (={n_eht_sig_sym_val}) != cfg N_EHT_SIG-1 (={cfg['N_EHT_SIG'] - 1})"

    usig2_info = np.concatenate([ppdu_type, validate2, punct_info,
                                  validate3, eht_sig_mcs, n_eht_sig_sym])
    assert len(usig2_info) == 16, 'U-SIG-2 info must be 16 bits'

    # CRC over bits 0-41 (U-SIG-1[0:25] + U-SIG-2[0:15])
    crc_input = np.concatenate([usig1_bits, usig2_info])
    assert len(crc_input) == 42, 'CRC input must be 42 bits'
    crc = crc4_usig(crc_input)
    tail = np.zeros(6, dtype=np.int8)
    usig2_bits = np.concatenate([usig2_info, crc, tail])
    assert len(usig2_bits) == 26, 'U-SIG-2 must be 26 bits'

    # ==================================================================
    # BCC encode all 52 bits as single pass -> 104 coded bits
    # ==================================================================
    all_bits = np.concatenate([usig1_bits, usig2_bits])
    assert len(all_bits) == 52, 'U-SIG pre-FEC stream must be 52 bits'
    encoded_all = _bcc_encoder(all_bits)
    assert len(encoded_all) == 104, 'BCC output must be 104 bits'

    # ==================================================================
    # Generate 2 OFDM symbols
    # ==================================================================
    sym_len = round(cfg['T_DFT_preEHT'] * Fs)
    CP_len = cfg['CP_preEHT']

    # Data and pilot subcarrier indices (per 20 MHz)
    data_sc_usig = np.concatenate([
        np.arange(-28, -21),   # -28:-22
        np.arange(-20, -7),    # -20:-8
        np.arange(-6, 0),      # -6:-1
        np.arange(1, 7),       # 1:6
        np.arange(8, 21),      # 8:20
        np.arange(22, 29),     # 22:28
    ])
    pilot_sc = np.array([-21, -7, 7, 21], dtype=np.int32)

    td_parts = []
    for sym_idx in range(2):
        # Split 104 coded bits: symbol 0 gets bits 0:52, symbol 1 gets 52:104
        sym_coded = encoded_all[sym_idx * 52: (sym_idx + 1) * 52].copy()
        assert len(sym_coded) == 52

        # Interleave (HE-SIG-A, N_CBPS=52, BPSK)
        sym_coded = _hesiga_interleave(sym_coded, 52, 1)

        # BPSK constellation mapping
        d_k = _constellation_map_bpsk(sym_coded)

        # Pilot polarity: L-SIG=p0, RL-SIG=p1, U-SIG-1=p2, U-SIG-2=p3
        # sym_idx=0 -> p_idx=2, sym_idx=1 -> p_idx=3
        p_idx = sym_idx + 2
        p_val = c['pilot_polarity'][p_idx % len(c['pilot_polarity'])]
        pilot_vals = p_val * np.array([1, 1, 1, -1], dtype=np.float64)

        # Build per-20 MHz legacy SC vector: k=-28..28 -> array[k+28]
        legacy_sc = np.zeros(57, dtype=np.complex128)
        for i in range(52):
            k = data_sc_usig[i]
            legacy_sc[k + 28] = d_k[i]
        for i in range(4):
            k = pilot_sc[i]
            legacy_sc[k + 28] = pilot_vals[i]

        # Replicate to full BW with FFT bin mapping
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

        # Normalization per Eq. 36-20
        assert cfg['NSS'] == 1, \
            f'gen_u_sig: Eq 36-20 assumes N_TX=1; got NSS={cfg["NSS"]}'
        N_tone_USIG = c['N_tone_LSIG']
        freq = freq / np.sqrt(N_tone_USIG)

        # IFFT + CP (pre-EHT: 3.2 us symbol + 0.8 us CP)
        td_full = np.fft.ifft(freq, NFFT) * np.sqrt(NFFT)
        one_sym = td_full[:sym_len]
        cp = one_sym[-CP_len:]
        td_parts.append(cp)
        td_parts.append(one_sym)

    return np.concatenate(td_parts)
