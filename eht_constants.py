# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
EHT_CONSTANTS  Return 802.11be PHY constants for given bandwidth.

    c = eht_constants(80)

IEEE Std 802.11be-2024, Section 36.3
IEEE Std 802.11-2020, Section 17 (legacy sequences), Section 27 (HE)
"""

import numpy as np


def eht_constants(BW=80):
    """Return 802.11be PHY constants for the given bandwidth.

    Parameters
    ----------
    BW : int
        Channel bandwidth in MHz: 20, 40, 80, 160, or 320.

    Returns
    -------
    dict
        Dictionary containing all PHY constants and lookup tables.
    """
    if BW not in (20, 40, 80, 160, 320):
        raise ValueError(f"BW must be 20, 40, 80, 160, or 320. Got {BW}")

    c = {}

    # =====================================================================
    # L-STF sequence S_{-26,26} -- IEEE 802.11be-2024 Section 36.3.12.3 (p.751)
    # explicitly references IEEE 802.11-2024 Equation (19-8), NOT Eq.(17-6).
    #
    # Eq.(19-8) reads:  S_{-26,26} = sqrt(1/2) * {0, 0, 1+j, 0, 0, 0, -1-j, ...}
    # The sqrt(1/2) factor is the QPSK normalization per Eq.(19-8)
    # (IEEE 802.11-2024, p.3427).
    #
    # 12 nonzero subcarriers at k={-24,-20,-16,-12,-8,-4,4,8,12,16,20,24}
    # Per-tone |S_k|^2 = (1/sqrt(2))^2 * |+-1+-j|^2 = (1/2) * 2 = 1
    # sum_k |S_k|^2 = 12 * 1 = 12
    # =====================================================================
    # 53 elements: index offset: array[k+26] = S_k (Python 0-indexed,
    # MATLAB used k+27 because 1-indexed)
    S = np.zeros(53, dtype=np.complex128)
    sq = 1.0 / np.sqrt(2.0)    # Eq. 19-8 QPSK normalization
    # Nonzero values -- signs matching IEEE 802.11-2024 Eq.(19-8):
    S[-24 + 26] =  sq * (1 + 1j)    # k=-24
    S[-20 + 26] =  sq * (-1 - 1j)   # k=-20
    S[-16 + 26] =  sq * (1 + 1j)    # k=-16
    S[-12 + 26] =  sq * (-1 - 1j)   # k=-12
    S[ -8 + 26] =  sq * (-1 - 1j)   # k=-8
    S[ -4 + 26] =  sq * (1 + 1j)    # k=-4
    S[  4 + 26] =  sq * (-1 - 1j)   # k=4
    S[  8 + 26] =  sq * (-1 - 1j)   # k=8
    S[ 12 + 26] =  sq * (1 + 1j)    # k=12
    S[ 16 + 26] =  sq * (1 + 1j)    # k=16
    S[ 20 + 26] =  sq * (1 + 1j)    # k=20
    S[ 24 + 26] =  sq * (1 + 1j)    # k=24
    c['S_26_26'] = S

    # =====================================================================
    # L-LTF sequence L_{-26,26} (IEEE 802.11-2020, Equation 17-8)
    # 52 nonzero subcarriers, DC=0
    # =====================================================================
    c['L_26_26'] = np.array([
        1,  1, -1, -1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1,
       -1,  1,  1, -1,  1, -1,  1,  1,  1,  1,
        0,   # DC (k=0)
        1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1,
        1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1
    ], dtype=np.float64)
    # Indices: L(-26) to L(+26), array[k+26] = L_k

    # =====================================================================
    # N_20MHz and K_Shift
    # K_Shift(i) = (N_20MHz - 1 - 2*i) * 32   (Section 36.3.12.3)
    # =====================================================================
    bw_map = {20: 1, 40: 2, 80: 4, 160: 8, 320: 16}
    c['N_20MHz'] = bw_map[BW]
    c['K_Shift'] = np.array(
        [(c['N_20MHz'] - 1 - 2 * i) * 32 for i in range(c['N_20MHz'])],
        dtype=np.int32
    )

    # =====================================================================
    # FFT sizes
    # =====================================================================
    fft_map = {20: 256, 40: 512, 80: 1024, 160: 2048, 320: 4096}
    c['NFFT'] = fft_map[BW]

    # Ratio of pre-EHT to EHT subcarrier spacing = 312.5/78.125 = 4
    c['SC_RATIO'] = 4   # legacy SC maps to every 4th EHT FFT bin

    # =====================================================================
    # Phase rotation per 20 MHz segment for pre-EHT fields
    # (Section 21.3.7.5 of 802.11ax, extended for EHT)
    # gamma_{k,BW} applied per segment
    # =====================================================================
    if BW == 20:
        c['gamma_preEHT'] = np.array([1], dtype=np.complex128)
    elif BW == 40:
        c['gamma_preEHT'] = np.array([1, 1j], dtype=np.complex128)
    elif BW == 80:
        c['gamma_preEHT'] = np.array([1, -1, -1, -1], dtype=np.complex128)
    elif BW == 160:
        c['gamma_preEHT'] = np.array([1, -1, -1, -1, 1, -1, -1, -1],
                                     dtype=np.complex128)
    elif BW == 320:
        # Eq. 36-14 with phi1=1, phi2=1, phi3=-1.
        # Segments ordered by K_Shift(i) = (15-2i)*32, i=0..15.
        c['gamma_preEHT'] = np.array(
            [1, -1, -1, -1,  1, -1, -1, -1,
             1, -1, -1, -1, -1,  1,  1,  1],
            dtype=np.complex128
        )

    # =====================================================================
    # Modulated subcarrier counts per field (Table 36-26)
    # =====================================================================
    tone_LSTF_map  = {20: 12, 40: 24, 80: 48, 160: 96, 320: 192}
    tone_LLTF_map  = {20: 52, 40: 104, 80: 208, 160: 416, 320: 832}
    tone_LSIG_map  = {20: 56, 40: 112, 80: 224, 160: 448, 320: 896}
    c['N_tone_LSTF'] = tone_LSTF_map[BW]
    c['N_tone_LLTF'] = tone_LLTF_map[BW]
    c['N_tone_LSIG'] = tone_LSIG_map[BW]

    # =====================================================================
    # Pilot subcarrier indices for Data field (per IEEE 802.11be-2024
    # Section 36.3.13.11, p.832):
    #   * 20 MHz / 40 MHz PPDU -> use HE tables from Section 27.3.12.13
    #   * 80 MHz / 160 MHz / 320 MHz PPDU -> use EHT tables
    #       - 996-tone RU: IEEE 802.11be-2024 Table 36-58, p.834
    # =====================================================================
    p996 = np.array([-468, -400, -334, -266, -220, -152, -86, -18,
                       18,   86,  152,  220,  266,  334,  400,  468],
                    dtype=np.int32)

    if BW == 20:
        # Table 27-43 (242-tone RU, 20 MHz PPDU): 8 pilots
        c['pilot_indices'] = np.array(
            [-116, -90, -48, -22, 22, 48, 90, 116], dtype=np.int32)
    elif BW == 40:
        # Table 27-45 (484-tone RU, 40 MHz PPDU): 16 pilots
        c['pilot_indices'] = np.array(
            [-238, -212, -170, -144, -104, -78, -36, -10,
               10,   36,   78,  104,  144, 170, 212, 238], dtype=np.int32)
    elif BW == 80:
        # Table 36-58 (996-tone RU, 80 MHz PPDU): 16 pilots
        c['pilot_indices'] = p996.copy()
    elif BW == 160:
        # Table 36-58 (996-tone RU, 160 MHz PPDU row): 32 pilots
        c['pilot_indices'] = np.concatenate([p996 - 512, p996 + 512])
    elif BW == 320:
        # Table 36-58 (996-tone RU, 320 MHz PPDU row): 64 pilots
        c['pilot_indices'] = np.concatenate(
            [p996 - 1536, p996 - 512, p996 + 512, p996 + 1536])

    # =====================================================================
    # Data subcarrier indices for EHT data field (non-OFDMA full channel)
    # =====================================================================
    sr_map = {20: 122, 40: 244, 80: 500, 160: 1012, 320: 2036}
    N_SR = sr_map[BW]
    all_sc = np.arange(-N_SR, N_SR + 1, dtype=np.int32)

    if BW == 20:
        dc_null = np.array([-1, 0, 1], dtype=np.int32)
    elif BW == 40:
        dc_null = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    elif BW == 80:
        dc_null = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    elif BW == 160:
        # Table 36-20: 23 centre DC nulls + intra-160 nulls
        dc_null = np.unique(np.concatenate([
            np.arange(-11, 12, dtype=np.int32),        # -11:11
            np.arange(-514, -509, dtype=np.int32),     # -514:-510
            np.arange(510, 515, dtype=np.int32)         # 510:514
        ]))
    elif BW == 320:
        # 23 DC nulls + intra-160 nulls + inter-160 nulls
        dc_null = np.unique(np.concatenate([
            np.arange(-11, 12, dtype=np.int32),          # 23 DC nulls
            np.arange(-514, -509, dtype=np.int32),       # -514:-510
            np.arange(510, 515, dtype=np.int32),          # 510:514
            np.arange(-1538, -1533, dtype=np.int32),     # -1538:-1534
            np.arange(1534, 1539, dtype=np.int32),        # 1534:1538
            np.arange(-1035, -1012, dtype=np.int32),     # -1035:-1013
            np.arange(1013, 1036, dtype=np.int32)         # 1013:1035
        ]))

    # occupied = setdiff(all_sc, dc_null)
    occupied = np.setdiff1d(all_sc, dc_null)
    # data_indices = setdiff(occupied, pilot_indices)
    c['data_indices'] = np.setdiff1d(occupied, c['pilot_indices'])
    c['all_occupied'] = np.sort(occupied)
    c['N_SR'] = N_SR

    # =====================================================================
    # Sampling rate
    # =====================================================================
    c['Fs'] = BW * 1e6

    # =====================================================================
    # Pilot polarity sequence p_n (127 elements, Section 17.3.5.10)
    # =====================================================================
    c['pilot_polarity'] = np.array([
         1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1,
        -1,-1, 1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1,
         1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1,
        -1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
        -1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,
        -1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,
        -1,-1,-1,-1,-1, 1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,
        -1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1
    ], dtype=np.int8)

    # =====================================================================
    # Legacy pilot / L-SIG constants
    # =====================================================================
    c['legacy_pilot_sc'] = np.array([-21, -7, 7, 21], dtype=np.int32)
    c['LSIG_RATE_bits'] = np.array([1, 1, 0, 1], dtype=np.int8)
    c['LSIG_extra_sc_indices'] = np.array([-28, -27, 27, 28], dtype=np.int32)
    c['LSIG_extra_sc_values'] = np.array([-1, -1, -1, 1], dtype=np.float64)

    # =====================================================================
    # EHT-SIG per-MCS symbol count + U-SIG field encoding
    # IEEE 802.11be-2024 Table 36-28 (p.758) for SU PPDU, B11-B15:
    #   EHT_SIG_MCS    N_EHT_SIG (symbols actually emitted)   FIELD value
    #   0 (BPSK 1/2)             2                               1
    #   1 (QPSK 1/2)             1                               0
    #   2 (16-QAM 1/2)           1                               0
    #   3 (BPSK DCM)             4                               3
    # Spec rule: FIELD = actual_symbols - 1 (verified column-wise above).
    #
    # Single source of truth for both eht_config.py and gen_u_sig.py.
    c['eht_sig_mcs_table'] = {
        'mcs':         np.array([0, 1, 2, 3], dtype=np.int32),
        'n_sym':       np.array([2, 1, 1, 4], dtype=np.int32),
        'field_value': np.array([1, 0, 0, 3], dtype=np.int32),
    }

    return c
