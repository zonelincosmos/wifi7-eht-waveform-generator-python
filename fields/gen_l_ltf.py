# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
GEN_L_LTF  Generate L-LTF for EHT PPDU.

Section 36.3.12.4, Equation (36-16)
Duration: 8 us = 1.6 us CP + 2 x 3.2 us long training symbols

Pre-EHT field: legacy L_k replicated per 20 MHz segment.
FFT bin mapping: (k_local - K_Shift) * 4
"""

import numpy as np

from eht_constants import eht_constants


def gen_l_ltf(cfg):
    """Generate L-LTF time-domain waveform.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from eht_config().

    Returns
    -------
    numpy.ndarray
        Time-domain L-LTF waveform (complex128), CP + 2 symbols (8 us).
    """
    c = eht_constants(cfg['BW'])
    NFFT = cfg['NFFT']
    Fs = cfg['Fs']

    L_k = c['L_26_26']          # L(-26) to L(26)
    gamma = c['gamma_preEHT']

    freq = np.zeros(NFFT, dtype=np.complex128)

    for seg in range(c['N_20MHz']):
        Ks = c['K_Shift'][seg]
        g = gamma[seg]           # per-segment; spec-correct for all BWs

        for k_local in range(-26, 27):   # -26..26
            val = L_k[k_local + 26]      # Python 0-indexed
            if val == 0:
                continue

            k_global = k_local - Ks
            fft_idx = k_global * c['SC_RATIO']
            bin_idx = fft_idx % NFFT
            freq[bin_idx] = g * val

    # Normalization per Eq.(36-16) with Eq.(36-11):
    #   beta = epsilon / sqrt(N_TX * N_LLTF^Tone * |Omega|/N_20MHz)
    #   epsilon = sqrt(N_LLTF^Tone / N_LSIG^Tone) for L-LTF (page 751)
    #   For SISO non-punctured: => beta = 1/sqrt(N_LSIG)
    norm = 1.0 / np.sqrt(c['N_tone_LSIG'])
    freq = freq * norm

    td_full = np.fft.ifft(freq, NFFT) * np.sqrt(NFFT)

    # L-LTF symbol: T_DFT_preEHT period (3.2us) at sample rate Fs,
    # with double-length GI T_GI_LLTF (1.6us) per Table 36-18.
    sym_len = round(cfg['T_DFT_preEHT'] * Fs)
    CP_len = cfg['CP_LLTF']

    one_sym = td_full[:sym_len]
    cp = one_sym[sym_len - CP_len:]    # last CP_len samples of one_sym

    # L-LTF = CP(1.6us) + Symbol(3.2us) + Symbol(3.2us) = 8 us
    td = np.concatenate([cp, one_sym, one_sym])
    return td
