# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
GEN_L_STF  Generate L-STF for EHT PPDU.

Section 36.3.12.3, Equation (36-15)
Duration: 8 us = 10 short training symbols of 0.8 us each

Pre-EHT field: uses 312.5 kHz subcarrier spacing.
Legacy S_k replicated per 20 MHz with K_Shift and phase rotation.
In full-BW NFFT, legacy SC k maps to FFT bin (k - K_Shift) * 4.
"""

import numpy as np

from eht_constants import eht_constants


def gen_l_stf(cfg):
    """Generate L-STF time-domain waveform.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from eht_config().

    Returns
    -------
    numpy.ndarray
        Time-domain L-STF waveform (complex128), 10 short symbols (8 us).
    """
    c = eht_constants(cfg['BW'])
    NFFT = cfg['NFFT']
    S_k = c['S_26_26']           # S(-26..26) from Eq.(19-8), scale 1/sqrt(2)
    gamma = c['gamma_preEHT']

    # Build frequency domain in NFFT-size FFT
    freq = np.zeros(NFFT, dtype=np.complex128)

    for seg in range(c['N_20MHz']):
        Ks = c['K_Shift'][seg]
        g = gamma[seg]           # per-segment gamma

        for k_local in range(-26, 27):   # -26..26
            val = S_k[k_local + 26]      # Python 0-indexed (MATLAB used k+27)
            if val == 0:
                continue

            # Global subcarrier (legacy spacing) = k_local - K_Shift
            # FFT bin = global_sc * 4 (ratio of pre-EHT to EHT spacing)
            k_global = k_local - Ks
            fft_idx = k_global * c['SC_RATIO']

            bin_idx = fft_idx % NFFT     # Python mod (MATLAB mod + 1)
            freq[bin_idx] = g * val

    # Normalization per Eq.(36-15) with Eq.(36-11):
    #   beta_L-STF = epsilon_L-STF / sqrt(N_TX * N_L-STF^Tone * |Omega|/N_20)
    #              = sqrt(N_LLTF/N_LSIG) / sqrt(N_LSTF)   (SISO, non-punctured)
    epsilon = np.sqrt(c['N_tone_LLTF'] / c['N_tone_LSIG'])
    norm = epsilon / np.sqrt(c['N_tone_LSTF'])
    freq = freq * norm

    # IFFT to time domain
    td_full = np.fft.ifft(freq, NFFT) * np.sqrt(NFFT)

    # L-STF: 10 repetitions of 0.8 us short symbol = 8 us total
    # STF uses every 4th legacy SC -> effective spacing = 4 x 312.5 = 1250 kHz
    # In the full-BW FFT (78.125 kHz bins): every 4th legacy SC = every 16th FFT bin
    # Period = NFFT/16 samples = T_FFT_preEHT/4 = 3.2/4 = 0.8 us
    short_sym_len = NFFT // 16   # 0.8 us period

    td = np.tile(td_full[:short_sym_len], 10)
    return td
