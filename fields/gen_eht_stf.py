# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
GEN_EHT_STF  Generate EHT-STF (EHT Short Training Field).

Section 36.3.12.9:  EHTS = HES (same as HE-STF)
Section 27.3.11.9:  HE-STF definition
Duration: 4 us for EHT MU PPDU (0.8 us x 5 periods)

Eq.(27-22):  M = {-1,-1,-1,1,1,1,-1,1,1,1,-1,1,1,-1,1}  (15 elements)
Eq.(27-23):  20 MHz  HES_{-112:16:112} = {M}*(1+j)/sqrt(2)
Eq.(27-24):  40 MHz  HES_{-240:16:240} = {M,0,-M}*(1+j)/sqrt(2)
Eq.(27-25):  80 MHz  HES_{-496:16:496} = {M,1,-M,0,-M,1,-M}*(1+j)/sqrt(2)
Eq.(27-26): 160 MHz  HES_{-1008:16:1008}
Eq.(36-29): 320 MHz  EHTS_{-2032:16:2032}

Eq.(36-11): For non-punctured PPDU, beta = epsilon/sqrt(N_Field^Tone)
            epsilon = 1 for EHT-STF
"""

import numpy as np


def gen_eht_stf(cfg):
    """Generate EHT-STF (EHT Short Training Field).

    Parameters
    ----------
    cfg : dict
        Configuration from eht_config().

    Returns
    -------
    numpy.ndarray
        Time-domain samples (complex128).
    """
    NFFT = cfg['NFFT']

    # M sequence -- Eq.(27-22), 15 elements
    M = np.array([-1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
                 dtype=np.float64)

    # Build HES coefficient vector per bandwidth
    BW = cfg['BW']
    if BW == 20:
        # Eq.(27-23): {M} -- 15 elements for -112:16:112
        coeff = M.copy()
        sc_range = np.arange(-112, 113, 16)  # 15 SCs

    elif BW == 40:
        # Eq.(27-24): {M, 0, -M} -- 31 elements for -240:16:240
        coeff = np.concatenate([M, [0], -M])
        sc_range = np.arange(-240, 241, 16)  # 31 SCs

    elif BW == 80:
        # Eq.(27-25): {M, 1, -M, 0, -M, 1, -M} -- 63 elements
        coeff = np.concatenate([M, [1], -M, [0], -M, [1], -M])
        sc_range = np.arange(-496, 497, 16)  # 63 SCs

    elif BW == 160:
        # Eq.(27-26): {M, 1, -M, 0, -M, 1, -M, 0, -M, -1, M, 0, -M, 1, -M}
        coeff = np.concatenate([
            M, [1], -M, [0], -M, [1], -M,   # lower 80 (63)
            [0],                               # gap (1)
            -M, [-1], M, [0], -M, [1], -M     # upper 80 (63)
        ])
        sc_range = np.arange(-1008, 1009, 16)  # 127 SCs

    elif BW == 320:
        # Eq.(36-29): 320 MHz sequence per 802.11be spec p.802
        coeff = np.concatenate([
            M, [1], -M, [0], -M, [1], -M, [0],     # pos 1..8
            M, [1], -M, [0], -M, [1], -M, [0],     # pos 9..16
            -M, [-1], M, [0], M, [-1], M, [0],     # pos 17..24
            -M, [-1], M, [0], M, [-1], M            # pos 25..31
        ])
        sc_range = np.arange(-2032, 2033, 16)  # 255 SCs
    else:
        raise ValueError(f'Unsupported BW={BW}')

    # Apply (1+j)/sqrt(2) scaling -- all HES patterns
    coeff = coeff * (1 + 1j) / np.sqrt(2)

    # HES_0 = 0 -- zero the DC subcarrier
    dc_mask = (sc_range == 0)
    coeff[dc_mask] = 0

    # Zero edge tones per spec (retained for forward compatibility with
    # step-8 EHT TB PPDU; no-op for step-16 MU grid)
    edge_zeros_all = [8, -8, 1016, -1016, 1032, -1032, 2040, -2040]
    for ez in edge_zeros_all:
        idx = np.where(sc_range == ez)[0]
        if len(idx) > 0:
            coeff[idx[0]] = 0

    # Verify length
    assert len(coeff) == len(sc_range), \
        f'EHT-STF: coeff length {len(coeff)} != sc_range length {len(sc_range)}'

    # Map to NFFT bins (EHT subcarrier spacing, direct bin mapping)
    freq = np.zeros(NFFT, dtype=np.complex128)
    for i in range(len(sc_range)):
        k = sc_range[i]
        if coeff[i] == 0:
            continue
        fft_bin = k % NFFT
        freq[fft_bin] = coeff[i]

    # Normalization per Eq.(36-35/36-11)
    # beta = 1/sqrt(|K_r^EHT-STF|) where |K_r| = number of nonzero SCs
    n_nonzero = np.sum(np.abs(freq) > 0)
    if n_nonzero > 0:
        freq = freq / np.sqrt(n_nonzero)

    # IFFT -> time domain
    td_full = np.fft.ifft(freq, NFFT) * np.sqrt(NFFT)

    # EHT-STF: 5 x 0.8 us periods = 4 us
    # Every 16th SC populated -> period = NFFT/16 samples
    period = NFFT // 16
    td = np.tile(td_full[:period], 5)

    return td
