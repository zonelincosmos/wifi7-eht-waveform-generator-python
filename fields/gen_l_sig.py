# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
GEN_L_SIG  Generate L-SIG field for EHT PPDU.

Section 36.3.12.5, Equation (36-18)
Duration: 4 us = 0.8 us CP + 3.2 us symbol

RATE=6Mb/s, BCC rate 1/2, BPSK, 48 data + 4 pilot subcarriers.
FFT bin mapping: (k_local - K_Shift) * 4
"""

import numpy as np

from eht_constants import eht_constants


# =====================================================================
# Embedded helper: BCC rate-1/2 encoder
# Generator polynomials (Section 17.3.5.5):
#   g0 = 133 octal = [1 0 1 1 0 1 1]
#   g1 = 171 octal = [1 1 1 1 0 0 1]
#   Constraint length K = 7
# =====================================================================
def _bcc_encode_half(data):
    """Rate-1/2 BCC encoder.

    Parameters
    ----------
    data : numpy.ndarray
        Binary input vector (1-D).

    Returns
    -------
    numpy.ndarray
        Encoded binary vector, length = 2 * len(data).
    """
    data = np.asarray(data, dtype=np.int8).ravel()
    N = len(data)

    # Generator polynomials (MSB first)
    g0 = np.array([1, 0, 1, 1, 0, 1, 1], dtype=np.int8)   # 133 octal
    g1 = np.array([1, 1, 1, 1, 0, 0, 1], dtype=np.int8)   # 171 octal
    K = 7

    # Initialize shift register
    sr = np.zeros(K, dtype=np.int8)

    encoded = np.zeros(2 * N, dtype=np.int8)
    for i in range(N):
        # Shift in new bit (prepend)
        sr = np.concatenate([np.array([data[i]], dtype=np.int8), sr[:-1]])

        # Compute outputs
        out_A = np.sum(sr * g0) % 2
        out_B = np.sum(sr * g1) % 2

        encoded[2 * i] = out_A
        encoded[2 * i + 1] = out_B

    return encoded


# =====================================================================
# Embedded helper: Legacy interleaver (N_CBPS=48, BPSK)
# Section 17.3.5.7 interleaving
# =====================================================================
def _legacy_interleave(bits, N_CBPS, N_BPSCS):
    """Legacy 802.11a/g interleaver.

    Parameters
    ----------
    bits : numpy.ndarray
        Input coded bit vector of length N_CBPS.
    N_CBPS : int
        Number of coded bits per symbol.
    N_BPSCS : int
        Number of coded bits per subcarrier per stream.

    Returns
    -------
    numpy.ndarray
        Interleaved bit vector of length N_CBPS.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()
    N_COL = 16
    s = max(N_BPSCS // 2, 1)

    k = np.arange(N_CBPS)

    # First permutation: Eq. (17-19)
    i_p = (N_CBPS // N_COL) * (k % N_COL) + (k // N_COL)

    # Second permutation: Eq. (17-20)
    j_p = s * (i_p // s) + (i_p + N_CBPS - (N_COL * i_p // N_CBPS)) % s

    out = np.zeros(N_CBPS, dtype=np.int8)
    for idx in range(N_CBPS):
        out[j_p[idx]] = bits[idx]

    return out


# =====================================================================
# Embedded helper: BPSK constellation mapping
# =====================================================================
def _bpsk_map(bits):
    """Map binary bits to BPSK constellation: 0 -> -1, 1 -> +1.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary input vector.

    Returns
    -------
    numpy.ndarray
        BPSK symbols (float64).
    """
    bits = np.asarray(bits, dtype=np.float64).ravel()
    return 2.0 * bits - 1.0


# =====================================================================
# Integer to bit vector, LSB-first (matches MATLAB de2bi 'right-msb')
# =====================================================================
def _int2bits_lsb(val, n_bits):
    """Convert integer to LSB-first bit vector.

    Equivalent to MATLAB ``de2bi(val, n_bits, 'right-msb')``:
    index 0 = LSB, index n_bits-1 = MSB.

    Parameters
    ----------
    val : int
        Non-negative integer value.
    n_bits : int
        Number of output bits.

    Returns
    -------
    numpy.ndarray
        1-D array of shape (n_bits,) with dtype np.int8, LSB at index 0.
    """
    bits = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        bits[i] = val & 1
        val >>= 1
    return bits


def gen_l_sig(cfg):
    """Generate L-SIG time-domain waveform.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from eht_config().

    Returns
    -------
    td : numpy.ndarray
        Time-domain L-SIG waveform (complex128), CP + 1 symbol (4 us).
    lsig_freq_20 : numpy.ndarray
        Per-20MHz frequency-domain content (57 elements, k=-28..28),
        for reuse by RL-SIG.
    """
    c = eht_constants(cfg['BW'])
    NFFT = cfg['NFFT']
    Fs = cfg['Fs']

    # --- L-SIG bits (24 bits) ---
    rate_bits = c['LSIG_RATE_bits'].copy()            # [1,1,0,1] = 6 Mb/s
    reserved = np.array([0], dtype=np.int8)

    # MATLAB: de2bi(cfg.LSIG_LENGTH, 12, 'right-msb') -> LSB-first
    length_bits = _int2bits_lsb(cfg['LSIG_LENGTH'], 12)

    parity = np.array(
        [np.sum(np.concatenate([rate_bits, reserved, length_bits])) % 2],
        dtype=np.int8
    )
    tail_bits = np.zeros(6, dtype=np.int8)

    lsig_bits = np.concatenate([
        rate_bits, reserved, length_bits, parity, tail_bits
    ])
    # lsig_bits is 4 + 1 + 12 + 1 + 6 = 24 bits

    # --- BCC encode rate 1/2 -> 48 coded bits ---
    encoded = _bcc_encode_half(lsig_bits)

    # --- Legacy interleaver (N_CBPS=48, BPSK) ---
    encoded = _legacy_interleave(encoded, 48, 1)

    # --- BPSK constellation mapping -> 48 complex symbols ---
    d_k = _bpsk_map(encoded)

    # --- Build per-20MHz frequency vector (legacy 64-pt structure) ---
    # Data SC: {-26:-22, -20:-8, -6:-1, 1:6, 8:20, 22:26} = 48 SCs
    # Pilot SC: {-21, -7, 7, 21}
    # Extra SC: {-28, -27, 27, 28}
    data_sc = np.concatenate([
        np.arange(-26, -21),     # -26:-22
        np.arange(-20, -7),      # -20:-8
        np.arange(-6, 0),        # -6:-1
        np.arange(1, 7),         # 1:6
        np.arange(8, 21),        # 8:20
        np.arange(22, 27),       # 22:26
    ])
    pilot_sc = c['legacy_pilot_sc']

    # Pilot values: p0 * [1, 1, 1, -1]
    p0 = c['pilot_polarity'][0]
    pilot_vals = p0 * np.array([1, 1, 1, -1], dtype=np.float64)

    # Map to per-20MHz legacy subcarrier structure (k = -28 to 28)
    # 57 elements: index offset: array[k+28] for k=-28..28
    legacy_sc = np.zeros(57, dtype=np.complex128)

    # Data symbols
    for i in range(48):
        k = data_sc[i]
        legacy_sc[k + 28] = d_k[i]

    # Pilots
    for i in range(4):
        k = pilot_sc[i]
        legacy_sc[k + 28] = pilot_vals[i]

    # Extra subcarriers
    for i in range(4):
        k = c['LSIG_extra_sc_indices'][i]
        legacy_sc[k + 28] = c['LSIG_extra_sc_values'][i]

    lsig_freq_20 = legacy_sc.copy()   # return for RL-SIG reuse

    # --- Replicate to full bandwidth with phase rotation ---
    gamma = c['gamma_preEHT']
    freq = np.zeros(NFFT, dtype=np.complex128)

    for seg in range(c['N_20MHz']):
        Ks = c['K_Shift'][seg]
        g = gamma[seg]           # per-segment; spec-correct for all BWs

        for k_local in range(-28, 29):   # -28..28
            val = legacy_sc[k_local + 28]
            if val == 0:
                continue

            k_global = k_local - Ks
            fft_idx = k_global * c['SC_RATIO']
            bin_idx = fft_idx % NFFT
            freq[bin_idx] = g * val

    # Normalization (Eq 36-18): 1/sqrt(N_TX * N_tone_LSIG * |Omega|/N_20MHz)
    freq = freq / np.sqrt(c['N_tone_LSIG'])

    # IFFT + CP. Timing per Table 36-18.
    td_full = np.fft.ifft(freq, NFFT) * np.sqrt(NFFT)
    sym_len = round(cfg['T_DFT_preEHT'] * Fs)
    CP_len = cfg['CP_preEHT']

    one_sym = td_full[:sym_len]
    cp = one_sym[sym_len - CP_len:]    # last CP_len samples

    td = np.concatenate([cp, one_sym])
    return td, lsig_freq_20
