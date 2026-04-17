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


def eht_waveform_gen(BW=80, MCS=7, GI=0.8, LTFType=2, PayloadBytes=1000,
                     PSDU=None, ScramblerInit=1, Coding='auto',
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
        User-supplied PSDU (uint8 byte vector of length PayloadBytes).
        If None, a reproducible random PSDU is generated with seed 0.
    ScramblerInit : int
        11-bit scrambler seed, 1-2047.
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
                     Coding=Coding, **kwargs)

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
    # PayloadBytes = total PSDU length delivered to the PHY. Structure:
    #   [delim(4)] [MAC_hdr(26) + user_data + FCS(4)] [pad(0-3)] [EOF delims...]
    # Minimum overhead: 4 + 26 + 4 + 4 (at least one EOF delim) = 38 bytes.
    ampdu_overhead = 38
    if PayloadBytes < ampdu_overhead:
        raise ValueError(
            f'PayloadBytes must be >= {ampdu_overhead} (A-MPDU overhead: '
            '4 delimiter + 26 MAC header + 4 FCS + 4 EOF)'
        )

    # Determine the largest user-data length that fits with 4-byte alignment
    # and at least one EOF delimiter (matches build_ampdu.m descending loop).
    raw_user_space = PayloadBytes - 34
    user_data_len = 0
    for user_try in range(raw_user_space, -1, -1):
        subframe_unpad = 34 + user_try
        pad = (4 - (subframe_unpad % 4)) % 4
        subframe_len = subframe_unpad + pad
        remaining = PayloadBytes - subframe_len
        if remaining >= 4 and remaining % 4 == 0:
            user_data_len = user_try
            break

    if PSDU is None:
        rng = np.random.RandomState(0)  # reproducible default
        user_data = rng.randint(0, 256, size=user_data_len).astype(np.uint8)
    else:
        user_psdu = np.asarray(PSDU, dtype=np.uint8).ravel()
        if len(user_psdu) <= user_data_len:
            pad_bytes = np.zeros(user_data_len - len(user_psdu), dtype=np.uint8)
            user_data = np.concatenate([user_psdu, pad_bytes])
        else:
            user_data = user_psdu[:user_data_len].copy()

    # Build A-MPDU formatted PSDU with pre-FEC MAC padding per Sec.10.12.7.
    # Total PSDU delivered to the PHY = APEP_LENGTH + N_PAD_MAC_bytes
    # (Eq 36-66).  The extra bytes are EOF padding delimiters (signature
    # 0x4E) so the receiver sees a valid A-MPDU all the way through the
    # pre-FEC region.
    total_psdu_bytes = PayloadBytes + cfg.get('N_PAD_MAC_bytes', 0)
    psdu = build_ampdu(user_data, total_psdu_bytes)
    psdu_out = psdu.copy()

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
