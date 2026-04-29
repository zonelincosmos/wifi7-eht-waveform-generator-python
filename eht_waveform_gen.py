# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
EHT_WAVEFORM_GEN  Generate 802.11be EHT SU PPDU waveform (SISO 1SS).

    waveform, cfg, psdu = eht_waveform_gen(BW=80, MCS=7, GI=0.8)
    waveform, cfg, psdu = eht_waveform_gen(BW=320, MCS=13, PSDU=my_psdu)

Generates a complete EHT SU PPDU with the following fields:
    L-STF -> L-LTF -> L-SIG -> RL-SIG -> U-SIG -> EHT-SIG
    -> EHT-STF -> EHT-LTF -> Data -> PE

Optional PSDU parameter accepts a user-supplied PSDU (uint8 byte
array of length PayloadBytes). If omitted, a reproducible random
PSDU is generated using numpy.random with seed 0.

IEEE Std 802.11be-2024, Section 36

Outputs:
    waveform  - complex baseband IQ samples (1-D numpy array)
    cfg       - configuration dict with all parameters
    psdu_out  - the PSDU bytes actually used in the Data field
"""

import numpy as np

from eht_config import eht_config
from fields.gen_l_stf import gen_l_stf
from fields.gen_l_ltf import gen_l_ltf
from fields.gen_l_sig import gen_l_sig
from fields.gen_rl_sig import gen_rl_sig
from fields.gen_u_sig import gen_u_sig
from fields.gen_eht_sig import gen_eht_sig
from fields.gen_eht_stf import gen_eht_stf
from fields.gen_eht_ltf import gen_eht_ltf
from fields.gen_data_field import gen_data_field
from utils.ampdu import build_ampdu


def eht_waveform_gen(BW=80, MCS=7, GI=0.8, LTFType='auto', PayloadBytes=1000,
                     PSDU=None, ScramblerInit=1, NumMPDUs=1, Coding='auto',
                     verbose=True, **kwargs):
    """Generate IEEE 802.11be (WiFi 7) EHT SU PPDU waveform.

    Parameters
    ----------
    BW : int
        Channel bandwidth in MHz: 20, 40, 80, 160, 320.
    MCS : int
        EHT-MCS index 0-13.
    GI : float
        Guard interval in us: 0.8, 1.6, 3.2.
    LTFType : int or str
        EHT-LTF type: 2 or 4 (or 'auto').
    PayloadBytes : int
        PSDU length in bytes.
    PSDU : numpy.ndarray or None
        User-supplied PSDU bytes (uint8 vector).  Will be split across
        ``NumMPDUs`` chunks, padded/truncated to fit the available
        user-data space.  If None, a deterministic cycling 0..255 byte
        pattern is used (platform-portable across reference
        implementations).
    ScramblerInit : int
        11-bit scrambler seed, 1-2047.
    NumMPDUs : int
        Number of real MPDU subframes in the A-MPDU (default 1 = S-MPDU).
        Auto-increased if the per-subframe MPDU would exceed the HE/EHT
        12-bit Length cap of 4095 bytes.
    Coding : str
        'BCC', 'LDPC', or 'auto'.
    verbose : bool
        If True (default), print progress and summary information.
    **kwargs
        Additional keyword arguments passed to eht_config()
        (e.g., NominalPacketPadding, UL_DL, BSS_Color, TXOP, STA_ID,
         Beamformed, SpatialReuse, EHT_SIG_MCS).

    Returns
    -------
    waveform : numpy.ndarray
        Complex baseband IQ samples (1-D, complex128).
    cfg : dict
        Configuration dictionary with all parameters and field lengths.
    psdu_out : numpy.ndarray
        The PSDU bytes (uint8) actually used in the Data field.
    """
    # --- Configuration ---
    cfg = eht_config(BW=BW, MCS=MCS, GI=GI, LTFType=LTFType,
                     PayloadBytes=PayloadBytes, ScramblerInit=ScramblerInit,
                     NumMPDUs=NumMPDUs, Coding=Coding, **kwargs)

    if verbose:
        print('=== 802.11be EHT SU PPDU Waveform Generator ===')
        print(f'Bandwidth:     {cfg["BW"]} MHz')
        print(f'MCS:           {cfg["MCS"]}')
        print(f'Modulation:    {cfg["ModOrder"]}-QAM')
        print(f'Code rate:     {cfg["R_num"]}/{cfg["R_den"]}')
        print(f'Coding:        {cfg["Coding"]}')
        print(f'GI:            {cfg["GI"]:.1f} us')
        print(f'EHT-LTF Type:  {cfg["EHT_LTF_Type"]}x '
              f'({cfg["T_GI_EHT_LTF"]*1e6:.1f} us GI)')
        print(f'FFT size:      {cfg["NFFT"]}')
        print(f'Sample rate:   {cfg["Fs"]/1e6:.1f} MHz')
        print(f'Data SC:       {cfg["N_SD"]}')
        print(f'Pilot SC:      {cfg["N_SP"]}')
        print(f'N_CBPS:        {cfg["N_CBPS"]}')
        print(f'N_DBPS:        {cfg["N_DBPS"]}')
        print(f'Payload:       {cfg["PayloadBytes"]} bytes')
        print(f'N_SYM:         {cfg["N_SYM"]}')
        print(f'L-SIG LENGTH:  {cfg["LSIG_LENGTH"]}')
        print(f'TXTIME:        {cfg["TXTIME"]*1e6:.1f} us')
        print()

    # --- Generate PSDU in A-MPDU format ---
    # Per IEEE 802.11-2024 Section 10.12, EHT PPDUs must carry A-MPDU format.
    # PayloadBytes = total PSDU length delivered to the PHY.  Structure for
    # num_mpdus = N real subframes:
    #   [delim_1 + MAC_hdr + chunk_1 + FCS + pad(0-3)]
    #   [delim_2 + MAC_hdr + chunk_2 + FCS + pad(0-3)]
    #   ...
    #   [delim_N + MAC_hdr + chunk_N + FCS + pad(0-3)]   (EOF=0 on each)
    #   [EOF padding delim x M]                          (fills remainder)
    # Per-subframe overhead (before alignment pad) = 4 (delim) + 26 (MAC)
    # + 4 (FCS) = 34 bytes.
    num_mpdus = cfg['NumMPDUs']
    per_mpdu_overhead = 34
    max_align_pad     = 3
    min_tail          = 4

    # HE/EHT A-MPDU subframe delimiter Length is 12 bits (max 4095 bytes).
    # Per-subframe MPDU = 26 + chunk + 4 <= 4095, so max chunk per subframe
    # is 4065 bytes.  If the user's NumMPDUs is too small for the requested
    # PayloadBytes, the per-subframe MPDU would exceed 4095 ->
    # _build_delimiter would raise.  Auto-increase NumMPDUs to fit.
    MPDU_LENGTH_LIMIT  = 4095
    max_chunk_per_mpdu = MPDU_LENGTH_LIMIT - per_mpdu_overhead + 4   # = 4065
    denom              = (max_chunk_per_mpdu + per_mpdu_overhead - 4
                          + max_align_pad)                           # = 4098
    min_num_mpdus_for_12bit = max(
        1, -(-(PayloadBytes - min_tail) // denom)   # ceil division
    )
    if num_mpdus < min_num_mpdus_for_12bit:
        if verbose:
            print(
                f'NOTE: Auto-increasing NumMPDUs {num_mpdus} -> '
                f'{min_num_mpdus_for_12bit} to fit per-subframe MPDU within '
                f'HE/EHT 12-bit length limit ({MPDU_LENGTH_LIMIT} B).'
            )
        num_mpdus = min_num_mpdus_for_12bit
        cfg['NumMPDUs'] = num_mpdus

    ampdu_overhead_min = num_mpdus * per_mpdu_overhead + min_tail
    if PayloadBytes < ampdu_overhead_min + num_mpdus:
        raise ValueError(
            f'PayloadBytes={PayloadBytes} too small for num_mpdus={num_mpdus}. '
            f'Minimum = {ampdu_overhead_min + num_mpdus} '
            f'({num_mpdus} * 34-byte overhead + {min_tail} min tail + '
            f'{num_mpdus} min 1 byte/chunk).'
        )

    # Compute maximum total user-data bytes such that, after each subframe
    # is 4-byte aligned, the total subframe bytes leave at least 4 bytes
    # (multiple of 4) for EOF padding at the end.
    max_user = (PayloadBytes - num_mpdus * per_mpdu_overhead
                             - num_mpdus * max_align_pad
                             - min_tail)
    user_data_len = 0
    for user_try in range(max_user, -1, -1):
        chunk_base = user_try // num_mpdus
        chunk_rem  = user_try % num_mpdus
        total_subframes = 0
        for i in range(num_mpdus):
            cs = chunk_base + (1 if i < chunk_rem else 0)
            sf = per_mpdu_overhead + cs                          # unpadded
            pad = (4 - (sf % 4)) % 4
            total_subframes += sf + pad
        remaining = PayloadBytes - total_subframes
        if remaining >= min_tail and remaining % 4 == 0:
            user_data_len = user_try
            break

    if PSDU is None:
        # Reproducible default user data.  Cycling 0..255 pattern is
        # platform-portable (Mersenne Twister streams differ between
        # numpy and other implementations even with the same seed, so
        # an explicit deterministic pattern is preferred for fixtures).
        user_data = np.arange(user_data_len, dtype=np.uint8)
    else:
        user_psdu = np.asarray(PSDU, dtype=np.uint8).ravel()
        if len(user_psdu) <= user_data_len:
            pad_bytes = np.zeros(user_data_len - len(user_psdu), dtype=np.uint8)
            user_data = np.concatenate([user_psdu, pad_bytes])
        else:
            user_data = user_psdu[:user_data_len].copy()

    # Build A-MPDU formatted PSDU with pre-FEC MAC padding per Section 10.12.7.
    # Total PSDU delivered to the PHY = APEP_LENGTH + N_PAD_MAC_bytes
    # (Eq. 36-66).  The extra bytes are EOF padding delimiters (signature
    # 0x4E) so the receiver sees a valid A-MPDU all the way through the
    # pre-FEC region.
    total_psdu_bytes = PayloadBytes + cfg.get('N_PAD_MAC_bytes', 0)
    psdu = build_ampdu(user_data, total_psdu_bytes, num_mpdus)
    psdu_out = psdu.copy()
    if verbose:
        print(f'A-MPDU: {num_mpdus} real MPDU subframe(s), '
              f'user_data = {user_data_len} bytes total')

    # --- Generate each field ---
    if verbose:
        print('Generating L-STF...')
    td_lstf = gen_l_stf(cfg)

    if verbose:
        print('Generating L-LTF...')
    td_lltf = gen_l_ltf(cfg)

    if verbose:
        print('Generating L-SIG...')
    td_lsig, lsig_freq_20 = gen_l_sig(cfg)

    if verbose:
        print('Generating RL-SIG...')
    td_rlsig = gen_rl_sig(cfg, lsig_freq_20)

    if verbose:
        print('Generating U-SIG...')
    td_usig = gen_u_sig(cfg)

    if verbose:
        print('Generating EHT-SIG...')
    td_ehtsig = gen_eht_sig(cfg)

    if verbose:
        print('Generating EHT-STF...')
    td_ehtstf = gen_eht_stf(cfg)

    if verbose:
        print('Generating EHT-LTF...')
    td_ehtltf = gen_eht_ltf(cfg)

    if verbose:
        print(f'Generating Data field ({cfg["N_SYM"]} OFDM symbols)...')
    td_data = gen_data_field(cfg, psdu)

    # --- Packet Extension field (Section 36.3.14) ---
    # Per Section 36.3.14: PE field must have the same average power as Data,
    # content is implementation-defined. We repeat the last Data OFDM symbol
    # (or a circular prefix of it if PE is longer than one symbol).
    # Length = T_PE * Fs samples (T_PE may be 0; PE field is omitted then).
    n_pe_samples = round(cfg['T_PE'] * cfg['Fs'])
    if n_pe_samples > 0:
        if verbose:
            print(f'Generating PE field ({n_pe_samples} samples = '
                  f'{cfg["T_PE"]*1e6:.1f} us)...')
        sym_len = cfg['NFFT'] + cfg['CP_Data']
        if len(td_data) >= sym_len:
            last_sym = td_data[-sym_len:]
        else:
            last_sym = td_data.copy()
        reps = int(np.ceil(n_pe_samples / max(1, len(last_sym))))
        td_pe = np.tile(last_sym, reps)
        td_pe = td_pe[:n_pe_samples]
    else:
        td_pe = np.array([], dtype=np.complex128)

    # --- Assemble PPDU ---
    # Note on field-boundary windowing (spec Section 36.3.12 Eq. 36-15..36-19):
    # The spec specifies a time-domain windowing function w_T(t) at each
    # field boundary (typ. 100 ns raised-cosine) to shape the spectral
    # mask. This is an OOB-emission / spectral-mask concern and does not
    # affect the decoded bits or the in-band sample values. For a DPD/
    # test-waveform generator (this tool's purpose) the OOB mask is the
    # responsibility of the downstream RF chain / pulse-shaping filter,
    # so we concatenate fields without windowing.
    waveform = np.concatenate([
        td_lstf, td_lltf, td_lsig, td_rlsig,
        td_usig, td_ehtsig, td_ehtstf, td_ehtltf, td_data, td_pe
    ])

    # --- Store field lengths for analysis ---
    cfg['FieldLengths'] = {
        'LSTF':   len(td_lstf),
        'LLTF':   len(td_lltf),
        'LSIG':   len(td_lsig),
        'RLSIG':  len(td_rlsig),
        'USIG':   len(td_usig),
        'EHTSIG': len(td_ehtsig),
        'EHTSTF': len(td_ehtstf),
        'EHTLTF': len(td_ehtltf),
        'Data':   len(td_data),
        'PE':     len(td_pe),
    }

    if verbose:
        print()
        print('=== Waveform Generated ===')
        print(f'Total samples: {len(waveform)}')
        print(f'Duration:      {len(waveform)/cfg["Fs"]*1e6:.2f} us')
        print()
        print('Field durations (us):')
        for name, key in [('L-STF', 'LSTF'), ('L-LTF', 'LLTF'),
                          ('L-SIG', 'LSIG'), ('RL-SIG', 'RLSIG'),
                          ('U-SIG', 'USIG'), ('EHT-SIG', 'EHTSIG'),
                          ('EHT-STF', 'EHTSTF'), ('EHT-LTF', 'EHTLTF'),
                          ('Data', 'Data'), ('PE', 'PE')]:
            dur = cfg['FieldLengths'][key] / cfg['Fs'] * 1e6
            print(f'  {name:10s} {dur:6.1f}')

    return waveform, cfg, psdu_out
