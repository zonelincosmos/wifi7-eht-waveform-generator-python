# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
EHT_CONFIG  Create configuration for 802.11be EHT SU PPDU (SISO 1SS).

    cfg = eht_config()                 -- Default: 80 MHz, MCS 7, 0.8us GI
    cfg = eht_config(BW=80, MCS=7, GI=0.8, PayloadBytes=1000)

IEEE Std 802.11be-2024, Section 36
"""

import math
import numpy as np

from eht_constants import eht_constants


def _ldpc_params(N_CBPS, R_num, R_den, psdu_bytes, N_service,
                 N_CBPS_short=None, a_init_in=None):
    """Compute LDPC encoding parameters per Section 19.3.11.7.5
    + IEEE 802.11be-2024 Section 36.3.13.3.5 extra-symbol update.

    Implements the LDPC PPDU encoding parameter selection from
    IEEE 802.11-2024 Section 19.3.11.7.5, steps (a)-(e), with the
    EHT-specific extra-symbol update per IEEE 802.11be-2024 Eq. 36-56
    and Eq. 36-58.

    Adapted for EHT SU PPDU (SISO 1SS, m_STBC=1).

    Returns
    -------
    dict
        Dictionary with fields: L_LDPC, N_CW, N_shrt, N_punc, N_rep,
        N_SYM, N_avbits, N_pld, has_extra_symbol, a_init_used.
    """
    R = R_num / R_den
    m_STBC = 1   # No STBC for EHT SU SISO

    # --- Step a) raw N_pld, initial N_SYM, then EHT N_pld override ---
    # Raw payload: 8*PSDU + SERVICE (Eq. 36-47 input, no tail for LDPC)
    N_pld_raw = 8 * psdu_bytes + N_service
    N_DBPS = int(math.floor(N_CBPS * R))

    # Initial number of OFDM symbols (Eq. 36-49)
    N_SYM = int(math.ceil(N_pld_raw / (N_DBPS * m_STBC))) * m_STBC
    N_SYM_init = N_SYM

    # Start with raw N_pld; will be overridden by Eq. 36-54 below.
    N_pld = N_pld_raw

    # --- Resolve a_init for Eq. 36-56/58 handling ---
    if a_init_in is not None:
        a_init = a_init_in
    elif N_CBPS_short is not None:
        N_DBPS_short = int(math.floor(N_CBPS_short * R))
        N_Excess = N_pld % N_DBPS    # N_tail = 0 for LDPC
        if N_Excess == 0:
            a_init = 4
        else:
            a_init = min(int(math.ceil(N_Excess / N_DBPS_short)), 4)
    else:
        a_init = None   # unknown -> HT-style fallback

    # --- EHT override: N_pld (Eq. 36-54) and N_avbits (Eq. 36-55) ---
    if a_init is not None and N_CBPS_short is not None:
        N_DBPS_short_local = int(math.floor(N_CBPS_short * R))
        if a_init == 4:
            N_DBPS_last_init = N_DBPS                                   # Eq. 36-52 top
            N_CBPS_last_init = N_CBPS                                   # Eq. 36-53 top
        else:
            N_DBPS_last_init = a_init * N_DBPS_short_local              # Eq. 36-52 bottom
            N_CBPS_last_init = a_init * N_CBPS_short                    # Eq. 36-53 bottom
        N_pld    = (N_SYM - m_STBC) * N_DBPS + m_STBC * N_DBPS_last_init  # Eq. 36-54
        N_avbits = (N_SYM - m_STBC) * N_CBPS + m_STBC * N_CBPS_last_init  # Eq. 36-55
    else:
        N_avbits = N_CBPS * N_SYM

    # --- Step b) Select N_CW and L_LDPC per Table 19-16 ---
    if N_avbits <= 648:
        N_CW = 1
        if N_avbits >= N_pld + 912 * (1 - R):
            L_LDPC = 1296
        else:
            L_LDPC = 648
    elif N_avbits <= 1296:
        N_CW = 1
        if N_avbits >= N_pld + 1464 * (1 - R):
            L_LDPC = 1944
        else:
            L_LDPC = 1296
    elif N_avbits <= 1944:
        N_CW = 1
        L_LDPC = 1944
    elif N_avbits <= 2592:
        N_CW = 2
        if N_avbits >= N_pld + 2916 * (1 - R):
            L_LDPC = 1944
        else:
            L_LDPC = 1296
    else:
        # N_avbits > 2592
        N_CW = int(math.ceil(N_pld / (1944 * R)))
        L_LDPC = 1944

    # --- Step c) Compute shortening bits (Eq. 19-37) ---
    # Cast to int: round() returns a float; downstream consumers expect ints.
    N_shrt = int(max(0, round(N_CW * L_LDPC * R) - N_pld))

    # --- Step d) Compute puncturing bits (Eq. 19-38) ---
    N_punc = int(max(0, round(N_CW * L_LDPC) - N_avbits - N_shrt))

    # Check for excessive puncturing (step d condition)
    parity_total = N_CW * L_LDPC * (1 - R)
    cond1 = (N_punc > 0.1 * parity_total) and \
            (N_shrt < 1.2 * N_punc * R / (1 - R))
    cond2 = (N_punc > 0.3 * parity_total)

    has_extra = cond1 or cond2

    if has_extra:
        # --- Extra-symbol update ---
        if N_CBPS_short is None or a_init is None:
            # --- Fallback: HT-style Eq. 19-39 / 19-40 ---
            N_avbits = N_avbits + N_CBPS * m_STBC
            N_SYM = N_SYM + m_STBC
        else:
            # --- Spec-compliant path (Eq. 36-56 + 36-58) ---
            if a_init == 3:
                # Eq. 36-56, a_init = 3 branch
                N_avbits = N_avbits + N_CBPS - 3 * N_CBPS_short
            else:
                # Eq. 36-56, "otherwise" branch (a_init in {1,2,4})
                N_avbits = N_avbits + N_CBPS_short

            if a_init == 4:
                # Eq. 36-58, a_init = 4 branch
                N_SYM = N_SYM_init + 1
            else:
                # Eq. 36-58, "otherwise" branch
                N_SYM = N_SYM_init

        # Eq. 36-57 / Eq. 19-40: recompute N_punc from updated N_avbits
        N_punc = int(max(0, round(N_CW * L_LDPC) - N_avbits - N_shrt))

    # --- Step e) Compute repeated coded bits (Eq. 19-42) ---
    N_rep = int(max(0, N_avbits - round(N_CW * L_LDPC * (1 - R)) - N_pld))

    # --- Pack output ---
    return {
        'L_LDPC':           L_LDPC,
        'N_CW':             N_CW,
        'N_shrt':           N_shrt,
        'N_punc':           N_punc,
        'N_rep':            N_rep,
        'N_SYM':            N_SYM,
        'N_avbits':         N_avbits,
        'N_pld':            N_pld,
        'N_pld_raw':        N_pld_raw,
        'has_extra_symbol': has_extra,
        'a_init_used':      a_init,
    }


def eht_config(BW=80, MCS=7, GI=0.8, LTFType='auto', PayloadBytes=1000,
               ScramblerInit=1, Coding='auto', NominalPacketPadding=16,
               UL_DL=0, BSS_Color=0, TXOP=127, STA_ID=0, Beamformed=0,
               SpatialReuse=0, EHT_SIG_MCS=0):
    """Create configuration for 802.11be EHT SU PPDU (SISO 1SS).

    Parameters
    ----------
    BW : int
        Channel bandwidth in MHz: 20, 40, 80, 160, 320.
    MCS : int
        EHT-MCS index 0-13.
    GI : float
        Guard interval in us: 0.8, 1.6, 3.2.
    LTFType : int or str
        EHT-LTF type: 2 or 4 (or 'auto' to derive from GI).
    PayloadBytes : int
        PSDU length in bytes.
    ScramblerInit : int
        11-bit scrambler seed, 1-2047.
    Coding : str
        'BCC', 'LDPC', or 'auto'.
    NominalPacketPadding : int
        PE duration in us: 0, 8, 16, or 20.
    UL_DL : int
        U-SIG-1 B6: 0=DL, 1=UL.
    BSS_Color : int
        U-SIG-1 B7-B12, 0..63.
    TXOP : int
        U-SIG-1 B13-B19, 0..127.
    STA_ID : int
        EHT-SIG user field B0-B10, 0..2047.
    Beamformed : int
        EHT-SIG user field B20: 0 or 1.
    SpatialReuse : int
        EHT-SIG common B0-B3, 0..15.
    EHT_SIG_MCS : int
        U-SIG-2 B9-B10, 0..3.

    Returns
    -------
    dict
        Configuration dictionary with all derived parameters.
    """
    cfg = {}

    cfg['NominalPacketPadding'] = NominalPacketPadding
    cfg['UL_DL']                = UL_DL
    cfg['BSS_Color']            = BSS_Color
    cfg['TXOP']                 = TXOP
    cfg['STA_ID']               = STA_ID
    cfg['Beamformed']           = Beamformed
    cfg['SpatialReuse']         = SpatialReuse

    cfg['BW']             = BW
    cfg['MCS']            = MCS
    cfg['GI']             = GI
    cfg['PayloadBytes']   = PayloadBytes
    cfg['ScramblerInit']  = ScramblerInit
    cfg['NSS']            = 1   # SISO 1SS

    # Per Section 36.3.13.2 (p.813): the scrambler seed must be a nonzero
    # 11-bit value.
    if ScramblerInit < 1 or ScramblerInit > 2047:
        raise ValueError(
            "ScramblerInit must be 1..2047 (nonzero 11-bit LFSR seed "
            "per Section 36.3.13.2)"
        )

    # Validate PayloadBytes
    if not isinstance(PayloadBytes, (int, np.integer)):
        raise ValueError("PayloadBytes must be a non-negative integer scalar")
    if PayloadBytes < 0:
        raise ValueError("PayloadBytes must be a non-negative integer scalar")
    if PayloadBytes > 1_000_000:
        raise ValueError(
            f"PayloadBytes={PayloadBytes} exceeds a soft 1 MB guard. "
            "Use a smaller PSDU or adjust this limit explicitly."
        )

    # Validate UL_DL
    if UL_DL not in (0, 1):
        raise ValueError("UL_DL must be 0 (DL) or 1 (UL) per Table 36-28")
    # Validate Beamformed
    if Beamformed not in (0, 1):
        raise ValueError("Beamformed must be 0 or 1 per Table 36-40")

    # --- MCS Table (Table 36-76 ~ 36-86) ---
    # [MCS, Modulation order, N_BPSCS, CodeRate_num, CodeRate_den]
    mcs_table = [
        (0,    2,  1,  1, 2),   # BPSK   1/2
        (1,    4,  2,  1, 2),   # QPSK   1/2
        (2,    4,  2,  3, 4),   # QPSK   3/4
        (3,   16,  4,  1, 2),   # 16QAM  1/2
        (4,   16,  4,  3, 4),   # 16QAM  3/4
        (5,   64,  6,  2, 3),   # 64QAM  2/3
        (6,   64,  6,  3, 4),   # 64QAM  3/4
        (7,   64,  6,  5, 6),   # 64QAM  5/6
        (8,  256,  8,  3, 4),   # 256QAM 3/4
        (9,  256,  8,  5, 6),   # 256QAM 5/6
        (10, 1024, 10, 3, 4),   # 1024QAM 3/4
        (11, 1024, 10, 5, 6),   # 1024QAM 5/6
        (12, 4096, 12, 3, 4),   # 4096QAM 3/4
        (13, 4096, 12, 5, 6),   # 4096QAM 5/6
    ]

    if MCS < 0 or MCS > 13:
        raise ValueError("MCS must be 0-13")

    row = mcs_table[MCS]
    cfg['ModOrder'] = row[1]
    cfg['N_BPSCS']  = row[2]
    cfg['R_num']    = row[3]
    cfg['R_den']    = row[4]
    cfg['R']        = row[3] / row[4]

    # --- FFT size and subcarrier allocation (Table 36-19/20/21) ---
    bw_params = {
        20:  {'NFFT': 256,  'N_SD': 234,  'N_SP': 8,  'N_SR': 122},
        40:  {'NFFT': 512,  'N_SD': 468,  'N_SP': 16, 'N_SR': 244},
        80:  {'NFFT': 1024, 'N_SD': 980,  'N_SP': 16, 'N_SR': 500},
        160: {'NFFT': 2048, 'N_SD': 1960, 'N_SP': 32, 'N_SR': 1012},
        320: {'NFFT': 4096, 'N_SD': 3920, 'N_SP': 64, 'N_SR': 2036},
    }
    if BW not in bw_params:
        raise ValueError("BW must be 20, 40, 80, 160, or 320")

    bp = bw_params[BW]
    cfg['NFFT'] = bp['NFFT']
    cfg['N_SD'] = bp['N_SD']
    cfg['N_SP'] = bp['N_SP']
    cfg['N_SR'] = bp['N_SR']

    cfg['N_ST'] = cfg['N_SD'] + cfg['N_SP']
    cfg['Fs']   = BW * 1e6   # Sample rate

    # --- N_20MHz ---
    bw_to_n20 = {20: 1, 40: 2, 80: 4, 160: 8, 320: 16}
    cfg['N_20MHz'] = bw_to_n20[BW]

    # --- Coding selection (Section 36.3.13.3) ---
    if not isinstance(Coding, str):
        raise ValueError("Coding must be a string: 'auto', 'BCC', or 'LDPC'")
    coding_upper = Coding.upper()
    if coding_upper not in ('AUTO', 'BCC', 'LDPC'):
        raise ValueError(
            f"Coding must be one of {{'auto', 'BCC', 'LDPC'}} "
            f"(case-insensitive). Got '{Coding}'."
        )
    if coding_upper == 'AUTO':
        if BW == 20 and MCS <= 9:
            cfg['Coding'] = 'BCC'
        else:
            cfg['Coding'] = 'LDPC'
    else:
        cfg['Coding'] = coding_upper

    # Validate BCC constraint
    if cfg['Coding'] == 'BCC':
        if BW > 20:
            raise ValueError(
                "BCC only supported for 20 MHz (242-tone RU). "
                "Use LDPC for BW > 20 MHz."
            )
        if MCS > 9:
            raise ValueError(
                "BCC only supported for MCS 0-9. Use LDPC for MCS 10-13."
            )

    # --- Derived parameters ---
    cfg['N_CBPS'] = cfg['N_SD'] * cfg['N_BPSCS'] * cfg['NSS']
    cfg['N_DBPS'] = int(math.floor(cfg['N_CBPS'] * cfg['R']))

    # Tail bits
    if cfg['Coding'] == 'BCC':
        cfg['N_tail'] = 6
    else:
        cfg['N_tail'] = 0

    cfg['N_service'] = 16

    # --- Pre-compute N_SD_short / N_CBPS_short / a_init ---
    nsd_short_map = {20: 60, 40: 120, 80: 240, 160: 492, 320: 984}
    cfg['N_SD_short'] = nsd_short_map[BW]
    cfg['N_CBPS_short'] = cfg['N_SD_short'] * cfg['NSS'] * cfg['N_BPSCS']
    cfg['N_DBPS_short'] = int(math.floor(cfg['N_CBPS_short'] * cfg['R']))

    # N_pld for SU = 8*PSDU + N_tail + N_service (Eq. 36-47)
    N_pld_eht = 8 * PayloadBytes + cfg['N_tail'] + cfg['N_service']
    cfg['N_Excess'] = N_pld_eht % cfg['N_DBPS']    # Eq. 36-47
    if cfg['N_Excess'] == 0:
        cfg['a_init'] = 4                            # Eq. 36-48 top branch
    else:
        cfg['a_init'] = min(
            int(math.ceil(cfg['N_Excess'] / cfg['N_DBPS_short'])), 4
        )

    # Number of OFDM symbols
    if cfg['Coding'] == 'LDPC':
        # LDPC: use ldpc_params() for spec-compliant N_SYM / codeword
        cfg['ldpc'] = _ldpc_params(
            cfg['N_CBPS'], cfg['R_num'], cfg['R_den'],
            PayloadBytes, cfg['N_service'],
            cfg['N_CBPS_short'], cfg['a_init']
        )
        cfg['N_SYM'] = cfg['ldpc']['N_SYM']
        # Pre-FEC padding for LDPC per Eq. 36-63:
        # N_PAD = N_pld (Eq. 36-54) - N_pld_raw (8*APEP + N_service)
        cfg['N_PAD'] = cfg['ldpc']['N_pld'] - cfg['ldpc']['N_pld_raw']
        # Split pre-FEC padding between MAC (byte-aligned EOF delimiters,
        # Eq. 36-66) and PHY (remaining bits, Eq. 36-67).  The MAC-layer
        # portion is added to the A-MPDU as EOF padding delimiters
        # (signature 0x4E) per IEEE 802.11-2024 Section 10.12.7 so that a
        # receiver parsing the A-MPDU finds valid delimiters all the way
        # through the pre-FEC region.  Only the sub-byte remainder is
        # zero-padded at the PHY.
        cfg['N_PAD_MAC_bytes'] = cfg['N_PAD'] // 8           # Eq. 36-66
        cfg['N_PAD_PHY_bits']  = cfg['N_PAD'] % 8            # Eq. 36-67
    else:
        # BCC: Eq. 36-49 simplified for SU 1SS
        total_bits = 8 * PayloadBytes + cfg['N_service'] + cfg['N_tail']
        cfg['N_SYM'] = int(math.ceil(total_bits / cfg['N_DBPS']))
        cfg['N_PAD'] = cfg['N_SYM'] * cfg['N_DBPS'] - total_bits
        # BCC has no A-MPDU pre-FEC split; all padding stays at the PHY.
        cfg['N_PAD_MAC_bytes'] = 0
        cfg['N_PAD_PHY_bits']  = cfg['N_PAD']

    # --- Timing parameters (Table 36-18) ---
    cfg['T_DFT_preEHT'] = 3.2e-6
    cfg['T_GI_preEHT']  = 0.8e-6
    cfg['T_GI_LLTF']    = 1.6e-6
    cfg['T_LSTF']       = 8e-6
    cfg['T_LLTF']       = 8e-6
    cfg['T_LSIG']       = 4e-6
    cfg['T_RLSIG']      = 4e-6
    cfg['T_USIG']       = 8e-6
    cfg['T_EHTSIG']     = 4e-6
    cfg['T_EHTSTF']     = 4e-6   # for EHT MU/SU PPDU (non-TB)

    if GI == 0.8:
        cfg['T_GI_Data'] = 0.8e-6
        cfg['T_SYM']     = 13.6e-6
    elif GI == 1.6:
        cfg['T_GI_Data'] = 1.6e-6
        cfg['T_SYM']     = 14.4e-6
    elif GI == 3.2:
        cfg['T_GI_Data'] = 3.2e-6
        cfg['T_SYM']     = 16.0e-6
    else:
        raise ValueError("GI must be 0.8, 1.6, or 3.2")

    # CP length in samples for data symbols
    cfg['CP_Data']   = round(cfg['T_GI_Data'] * cfg['Fs'])
    # CP for pre-EHT symbols
    cfg['CP_preEHT'] = round(cfg['T_GI_preEHT'] * cfg['Fs'])
    # CP for L-LTF
    cfg['CP_LLTF']   = round(cfg['T_GI_LLTF'] * cfg['Fs'])

    # EHT-LTF parameters -- must match GI+LTF Size encoding in Table 36-36
    cfg['N_EHT_LTF'] = 1   # 1SS SU minimum

    if isinstance(LTFType, str) and LTFType.lower() == 'auto':
        # Derive LTF type from GI (pick minimum legal LTF size)
        if GI == 0.8:
            cfg['EHT_LTF_Type'] = 2    # gi_ltf = 0
        elif GI == 1.6:
            cfg['EHT_LTF_Type'] = 2    # gi_ltf = 1
        elif GI == 3.2:
            cfg['EHT_LTF_Type'] = 4    # gi_ltf = 3 (only legal)
    else:
        cfg['EHT_LTF_Type'] = int(LTFType)

    # Validate combination (Table 36-36)
    valid_combo = (
        (cfg['EHT_LTF_Type'] == 2 and GI == 0.8) or
        (cfg['EHT_LTF_Type'] == 2 and GI == 1.6) or
        (cfg['EHT_LTF_Type'] == 4 and GI == 0.8) or
        (cfg['EHT_LTF_Type'] == 4 and GI == 3.2)
    )
    if not valid_combo:
        raise ValueError(
            f"Invalid (EHT_LTF_Type, GI) combo = ({cfg['EHT_LTF_Type']}, "
            f"{GI} us). Valid combinations per Table 36-36: "
            "(2, 0.8), (2, 1.6), (4, 0.8), (4, 3.2)."
        )

    # Derive DFT period per LTF type
    if cfg['EHT_LTF_Type'] == 2:
        cfg['T_EHT_LTF'] = 6.4e-6     # 2x: NFFT/2 IFFT -> 6.4 us DFT period
    elif cfg['EHT_LTF_Type'] == 4:
        cfg['T_EHT_LTF'] = 12.8e-6    # 4x: NFFT IFFT -> 12.8 us DFT period

    # GI for EHT-LTF (and data field) follows user's GI selection
    cfg['T_GI_EHT_LTF']  = GI * 1e-6
    cfg['T_EHT_LTF_SYM'] = cfg['T_EHT_LTF'] + cfg['T_GI_EHT_LTF']

    # --- EHT-SIG parameters ---
    cfg['EHT_SIG_MCS'] = EHT_SIG_MCS
    if EHT_SIG_MCS < 0 or EHT_SIG_MCS > 3:
        raise ValueError("EHT_SIG_MCS must be 0..3")

    # Look up from eht_constants table
    c_sym_table = eht_constants(BW)
    mcs_arr = c_sym_table['eht_sig_mcs_table']['mcs']
    mcs_row_indices = np.where(mcs_arr == EHT_SIG_MCS)[0]
    if len(mcs_row_indices) == 0:
        raise ValueError(
            f"EHT_SIG_MCS={EHT_SIG_MCS} not in eht_sig_mcs_table"
        )
    mcs_row = mcs_row_indices[0]
    cfg['N_EHT_SIG'] = int(
        c_sym_table['eht_sig_mcs_table']['n_sym'][mcs_row]
    )

    # Only EHT_SIG_MCS=0 (BPSK 1/2) is fully exercised.
    if EHT_SIG_MCS != 0:
        raise ValueError(
            f"EHT_SIG_MCS = {EHT_SIG_MCS} not implemented in gen_eht_sig "
            "(only MCS 0 = BPSK 1/2 is currently supported). "
            "The non-zero MCS values would require QPSK/16-QAM/BPSK-DCM "
            "constellation mapping and adjusted symbol counts."
        )

    # --- Pre-FEC padding factor a (IEEE 802.11be-2024 Eq.(36-58)/(36-59)) ---
    if (cfg['Coding'] == 'LDPC' and 'ldpc' in cfg
            and cfg['ldpc']['has_extra_symbol']):
        # LDPC extra symbol added -> update a per Eq.(36-58)
        if cfg['a_init'] == 4:
            cfg['a'] = 1
        else:
            cfg['a'] = cfg['a_init'] + 1
    else:
        cfg['a'] = cfg['a_init']

    # --- T_PE per Table 36-61 (NOMINAL_PACKET_PADDING vs a) ---
    # Rows: a=1..4, Cols: NOM_PE = 0/8/16/20 us
    # Values are TPE in microseconds
    pe_table = [
        [0,  0,  4,  8],    # a=1
        [0,  0,  8, 12],    # a=2
        [0,  4, 12, 16],    # a=3
        [0,  8, 16, 20],    # a=4
    ]
    nom_pe_to_col = {0: 0, 8: 1, 16: 2, 20: 3}
    if NominalPacketPadding not in nom_pe_to_col:
        raise ValueError("NominalPacketPadding must be 0, 8, 16, or 20 us")
    # a is 1-based in the table (row index = a-1)
    cfg['T_PE'] = pe_table[cfg['a'] - 1][nom_pe_to_col[NominalPacketPadding]] * 1e-6

    # --- TXTIME calculation (Section 36.4.3) ---
    cfg['SignalExtension'] = 0   # 0 for 5 GHz / 6 GHz
    T_preamble = (cfg['T_LSTF'] + cfg['T_LLTF'] + cfg['T_LSIG'] +
                  cfg['T_RLSIG'] + cfg['T_USIG'] +
                  cfg['N_EHT_SIG'] * cfg['T_EHTSIG'] +
                  cfg['T_EHTSTF'] +
                  cfg['N_EHT_LTF'] * cfg['T_EHT_LTF_SYM'])
    cfg['TXTIME'] = (T_preamble + cfg['N_SYM'] * cfg['T_SYM'] +
                     cfg['T_PE'] + cfg['SignalExtension'])

    # --- L-SIG LENGTH (Eq. 36-17, page 752) ---
    # Snap to 1-ns precision to avoid FP rounding errors
    TXTIME_us = round(cfg['TXTIME'] * 1e9) / 1e3
    SE_us = round(cfg['SignalExtension'] * 1e9) / 1e3
    x_us = TXTIME_us - SE_us - 20.0
    cfg['LSIG_LENGTH'] = int(math.ceil(x_us / 4.0)) * 3 - 3

    # L-SIG LENGTH is a 12-bit field
    if cfg['LSIG_LENGTH'] > 4095:
        raise ValueError(
            f"TXTIME {cfg['TXTIME']*1e6:.1f} us (LSIG_LENGTH="
            f"{cfg['LSIG_LENGTH']}) exceeds the maximum representable "
            "by the 12-bit L-SIG LENGTH field (4095). "
            f"BW={BW} MCS={MCS} LTFType={cfg['EHT_LTF_Type']} "
            f"GI={GI} PayloadBytes={PayloadBytes}. "
            "Reduce PayloadBytes, lower GI, or use a higher MCS."
        )

    # --- PE Disambiguity (Eq. 36-94) ---
    T_PE_us  = round(cfg['T_PE']  * 1e9) / 1e3
    T_SYM_us = round(cfg['T_SYM'] * 1e9) / 1e3
    lhs_us = T_PE_us + 4.0 * math.ceil(x_us / 4.0) - x_us
    if lhs_us >= T_SYM_us - 1e-9:
        cfg['PE_Disambiguity'] = 1
    else:
        cfg['PE_Disambiguity'] = 0

    return cfg
