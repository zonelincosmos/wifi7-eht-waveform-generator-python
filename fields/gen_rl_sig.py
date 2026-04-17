# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
GEN_RL_SIG  Generate RL-SIG for EHT PPDU.

Section 36.3.12.6, Equation (36-19)
Identical to L-SIG but uses pilot polarity p1 instead of p0.
Duration: 4 us
"""

import numpy as np

from eht_constants import eht_constants


def gen_rl_sig(cfg, lsig_freq_20):
    """Generate RL-SIG time-domain waveform.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from eht_config().
    lsig_freq_20 : numpy.ndarray
        Per-20MHz frequency-domain content from gen_l_sig() (57 elements,
        k=-28..28, indexed as array[k+28]).

    Returns
    -------
    numpy.ndarray
        Time-domain RL-SIG waveform (complex128), CP + 1 symbol (4 us).
    """
    c = eht_constants(cfg['BW'])
    NFFT = cfg['NFFT']
    Fs = cfg['Fs']

    # Make a local copy to avoid modifying the caller's array
    lsig_freq_20 = lsig_freq_20.copy()

    # Replace pilot values with p1 instead of p0
    p1 = c['pilot_polarity'][1]
    pilot_sc = c['legacy_pilot_sc']
    pilot_vals = p1 * np.array([1, 1, 1, -1], dtype=np.float64)
    for i in range(4):
        k = pilot_sc[i]
        lsig_freq_20[k + 28] = pilot_vals[i]   # Python 0-indexed offset

    # Replicate to full bandwidth
    gamma = c['gamma_preEHT']
    freq = np.zeros(NFFT, dtype=np.complex128)

    for seg in range(c['N_20MHz']):
        Ks = c['K_Shift'][seg]
        g = gamma[seg]           # per-segment; spec-correct for all BWs

        for k_local in range(-28, 29):   # -28..28
            val = lsig_freq_20[k_local + 28]
            if val == 0:
                continue

            k_global = k_local - Ks
            fft_idx = k_global * c['SC_RATIO']
            bin_idx = fft_idx % NFFT
            freq[bin_idx] = g * val

    freq = freq / np.sqrt(c['N_tone_LSIG'])

    td_full = np.fft.ifft(freq, NFFT) * np.sqrt(NFFT)

    # Timing per Table 36-18.
    sym_len = round(cfg['T_DFT_preEHT'] * Fs)
    CP_len = cfg['CP_preEHT']

    one_sym = td_full[:sym_len]
    cp = one_sym[sym_len - CP_len:]    # last CP_len samples

    td = np.concatenate([cp, one_sym])
    return td
