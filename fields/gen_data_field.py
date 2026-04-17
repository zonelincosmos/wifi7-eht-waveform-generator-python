# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
Generate EHT Data field OFDM symbols.

    td = gen_data_field(cfg, psdu)

Section 36.3.13 of IEEE 802.11be-2024.

Pipeline: SERVICE + PSDU + tail + padding -> scramble -> FEC encode
          -> interleave (BCC) -> segment parser -> constellation map
          -> LDPC tone mapping -> pilot insertion -> normalization
          -> OFDM modulate -> concatenate symbols.
"""

import numpy as np

from modulation.scrambler import eht_scrambler
from coding.bcc_encoder import bcc_encoder
from coding.bcc_puncture import bcc_puncture
from coding.bcc_interleaver import bcc_interleaver
from coding.ldpc_encoder import ldpc_encoder
from modulation.constellation_map import constellation_map
from modulation.ofdm_mod import ofdm_mod
from eht_constants import eht_constants


def _ldpc_tone_map(sb_in, BW):
    """Per-subblock LDPC tone permutation per Eq. 36-72.

    Applies the LDPC tone mapper to a single frequency subblock of
    complex symbols.

    Parameters
    ----------
    sb_in : numpy.ndarray
        1-D complex array of N_SD_l symbols for one subblock.
    BW : int
        Channel bandwidth in MHz.

    Returns
    -------
    numpy.ndarray
        1-D complex array of permuted symbols, same length as *sb_in*.

    Notes
    -----
    D_TM is selected from Table 36-52 based on BW:
      BW 20  -> 242-tone RU, D_TM = 9,   N_SD_l = 234
      BW 40  -> 484-tone RU, D_TM = 12,  N_SD_l = 468
      BW>=80 -> 996-tone RU, D_TM = 20,  N_SD_l = 980

    The permutation moves symbol k to physical position:
      t(k) = D_TM * (k mod (N_SD_l / D_TM)) + floor(k * D_TM / N_SD_l)
    """
    if BW == 20:
        D_TM = 9
    elif BW == 40:
        D_TM = 12
    else:
        D_TM = 20    # BW >= 80 uses 996-tone subblocks

    N_SD_l = len(sb_in)
    N_SD_per_DTM = N_SD_l // D_TM
    if N_SD_l % D_TM != 0:
        raise ValueError(
            f"ldpc_tone_map: N_SD_l={N_SD_l} not divisible by D_TM={D_TM}"
        )

    sb_out = np.zeros(N_SD_l, dtype=np.complex128)
    for k in range(N_SD_l):
        t_k = D_TM * (k % N_SD_per_DTM) + (k * D_TM) // N_SD_l
        sb_out[t_k] = sb_in[k]

    return sb_out


def gen_data_field(cfg, psdu):
    """Generate EHT Data field OFDM symbols.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary from ``eht_config()``.
    psdu : numpy.ndarray
        PSDU as either:
        - uint8 byte array: each byte expanded to bits LSB-first per
          Section 36.3.13.2 ("bit 0 first and bit 7 last").
        - int8/int/float array of 0/1: treated as pre-formed bit vector.

    Returns
    -------
    numpy.ndarray
        1-D complex array of time-domain samples for all Data OFDM
        symbols (length = N_SYM * (NFFT + CP_Data)).
    """
    c = eht_constants(cfg['BW'])
    NFFT = cfg['NFFT']

    # ------------------------------------------------------------------
    # Convert PSDU to bits
    # ------------------------------------------------------------------
    psdu = np.asarray(psdu).ravel()

    if psdu.dtype == np.uint8:
        # Byte vector -> bit vector, LSB first per byte
        # (Section 36.3.13.2: "bit 0 first and bit 7 last")
        # MATLAB: bitget(double(psdu(i)), 1:8) extracts LSB first
        n_bytes = len(psdu)
        psdu_bits = np.zeros(n_bytes * 8, dtype=np.int8)
        for i in range(n_bytes):
            byte_val = int(psdu[i])
            for j in range(8):
                psdu_bits[i * 8 + j] = (byte_val >> j) & 1
    else:
        # Numeric / logical: must be 0/1 bits already
        if np.any(psdu > 1) or np.any(psdu < 0):
            raise ValueError(
                "PSDU as numeric must contain only 0/1 bits. "
                "Use uint8 dtype for byte input."
            )
        psdu_bits = psdu.astype(np.int8)

    # ------------------------------------------------------------------
    # Construct DATA field bit stream
    # ------------------------------------------------------------------
    # SERVICE field (cfg['N_service'] bits, normally 16)
    service_bits = np.zeros(cfg['N_service'], dtype=np.int8)

    # Tail bits (6 for BCC, 0 for LDPC)
    tail_bits = np.zeros(cfg['N_tail'], dtype=np.int8)

    # Pre-FEC PHY padding bits (Eq. 36-67).  The MAC-layer portion of the
    # pre-FEC padding (Eq. 36-66) was already inserted as EOF delimiters
    # inside the A-MPDU by build_ampdu; only the sub-byte remainder is
    # appended here at the PHY layer.  Falls back to the full N_PAD for
    # configurations built by an older eht_config without the split.
    n_pad_phy = cfg.get('N_PAD_PHY_bits', cfg['N_PAD'])
    pad_bits = np.zeros(n_pad_phy, dtype=np.int8)

    # Assemble: SERVICE + PSDU + Tail + Pad
    data_bits = np.concatenate([service_bits, psdu_bits, tail_bits, pad_bits])

    # ------------------------------------------------------------------
    # Scramble
    # ------------------------------------------------------------------
    scrambled, _ = eht_scrambler(data_bits, cfg['ScramblerInit'])

    # Zero out tail bits after scrambling (for BCC)
    if cfg['Coding'] == 'BCC':
        tail_start = 16 + len(psdu_bits)       # 0-indexed start of tail
        tail_end = tail_start + 6              # exclusive end
        if tail_end <= len(scrambled):
            scrambled[tail_start:tail_end] = 0

    # ------------------------------------------------------------------
    # FEC Encode
    # ------------------------------------------------------------------
    if cfg['Coding'] == 'BCC':
        # BCC encode at rate 1/2
        encoded = bcc_encoder(scrambled)
        # Puncture to desired rate
        encoded = bcc_puncture(encoded, cfg['R_num'], cfg['R_den'])

        # Verify coded length matches expected N_SYM * N_CBPS
        total_coded = cfg['N_SYM'] * cfg['N_CBPS']
        if len(encoded) != total_coded:
            raise ValueError(
                f"BCC-coded length {len(encoded)} != expected {total_coded} "
                f"(cfg.N_SYM={cfg['N_SYM']} * N_CBPS={cfg['N_CBPS']}). "
                "Check bcc_puncture and cfg.N_PAD."
            )
    else:
        # LDPC encode with shortening/puncturing/repetition
        # cfg['ldpc'] is a dict - convert to LdpcParams dataclass
        from coding.ldpc_params import LdpcParams
        ldpc_dict = cfg['ldpc']
        lp = LdpcParams(
            L_LDPC=ldpc_dict['L_LDPC'],
            N_CW=ldpc_dict['N_CW'],
            N_shrt=ldpc_dict['N_shrt'],
            N_punc=ldpc_dict['N_punc'],
            N_rep=ldpc_dict['N_rep'],
            N_SYM=ldpc_dict['N_SYM'],
            N_avbits=ldpc_dict['N_avbits'],
            N_pld=ldpc_dict['N_pld'],
            N_pld_raw=ldpc_dict.get('N_pld_raw', ldpc_dict['N_pld']),
            has_extra_symbol=ldpc_dict.get('has_extra_symbol', False),
            a_init_used=ldpc_dict.get('a_init_used', None),
        )
        encoded = ldpc_encoder(
            scrambled,
            [cfg['R_num'], cfg['R_den']],
            lp
        )
        # Post-FEC padding per Sec.36.3.13.3.5: when a_init < 4 we have
        # N_avbits < N_SYM * N_CBPS. Spec leaves content to implementation;
        # match MATLAB reference (gen_data_field.m:87-102) which fills with
        # PN11 scrambler output init = ScramblerInit XOR 1387 (fallback to 1
        # if 0) to avoid structured constellation patterns.
        total_coded = cfg['N_SYM'] * cfg['N_CBPS']
        n_pad_postfec = total_coded - len(encoded)
        if n_pad_postfec > 0:
            pad_init = cfg['ScramblerInit'] ^ 1387
            if pad_init == 0:
                pad_init = 1
            pad_bits, _ = eht_scrambler(
                np.zeros(n_pad_postfec, dtype=np.int8), pad_init
            )
            encoded = np.concatenate([encoded, pad_bits.astype(np.int8)])

    # ------------------------------------------------------------------
    # Generate OFDM symbols
    # ------------------------------------------------------------------
    sym_out_len = NFFT + cfg['CP_Data']
    td = np.zeros(cfg['N_SYM'] * sym_out_len, dtype=np.complex128)

    # Pilot base pattern (Table 27-44, Eq. 27-104)
    Psi = np.array([1, 1, 1, -1, -1, 1, 1, 1], dtype=np.float64)

    # Pilot polarity offset per Eq. 36-87:
    #   index = n + 2 + N_{U-SIG} + N_{EHT-SIG}
    #   N_{U-SIG} = 2 (fixed), so offset = 2 + 2 + N_EHT_SIG = 4 + N_EHT_SIG
    pilot_pol_offset = 2 + 2 + cfg['N_EHT_SIG']

    # Pilot polarity sequence (127 elements)
    pilot_polarity = c['pilot_polarity']
    n_pol = len(pilot_polarity)

    # Precompute subcarrier indices
    data_idx = c['data_indices']       # sorted subcarrier indices
    pilot_idx = c['pilot_indices']     # pilot subcarrier indices
    n_pilots = len(pilot_idx)

    # Determine segment parser parameters
    BW = cfg['BW']
    if BW in (20, 40, 80):
        N_sb = 1
    elif BW == 160:
        N_sb = 2
    elif BW == 320:
        N_sb = 4
    else:
        raise ValueError(
            f"gen_data_field: unsupported BW={BW} for segment parser"
        )
    N_SD_l = cfg['N_SD'] // N_sb
    if cfg['N_SD'] % N_sb != 0:
        raise ValueError(
            f"gen_data_field: cfg.N_SD={cfg['N_SD']} not divisible by "
            f"N_sb={N_sb}"
        )

    for sym_idx in range(cfg['N_SYM']):
        # Extract coded bits for this symbol
        sym_bits = encoded[
            sym_idx * cfg['N_CBPS']: (sym_idx + 1) * cfg['N_CBPS']
        ]

        # BCC interleaver (only for BCC coding)
        if cfg['Coding'] == 'BCC':
            sym_bits = bcc_interleaver(
                sym_bits, cfg['N_CBPS'], cfg['N_BPSCS'], cfg['BW']
            )

        # --------------------------------------------------------------
        # Segment parser + constellation map + LDPC tone mapping
        # --------------------------------------------------------------
        if N_sb == 1:
            # Trivial parser: whole stream -> single subblock
            data_syms = constellation_map(sym_bits, cfg['N_BPSCS'])
        else:
            # LDPC-only path (BCC is BW=20 only -> N_sb=1 branch)
            assert cfg['Coding'] == 'LDPC', (
                f"gen_data_field: N_sb>1 expected only with LDPC "
                f"(BW>=160), got {cfg['Coding']}"
            )
            # Segment parser per Eq. 36-70: bit-level round-robin
            # m_l = s_l = max(1, N_BPSCS/2)
            s_l = max(1, cfg['N_BPSCS'] // 2)
            N_CBPSS_l = cfg['N_CBPS'] // N_sb   # bits per subblock

            assert N_CBPSS_l % s_l == 0, (
                f"gen_data_field: N_CBPSS_l={N_CBPSS_l} not a multiple "
                f"of s_l={s_l}"
            )
            assert N_CBPSS_l % cfg['N_BPSCS'] == 0, (
                f"gen_data_field: N_CBPSS_l={N_CBPSS_l} not a multiple "
                f"of N_BPSCS={cfg['N_BPSCS']}"
            )

            # Round-robin dispatch: for each round of s_l bits,
            # subblock l=0 gets s_l bits, subblock 1 gets the next s_l,
            # etc.
            sb_bits = np.zeros((N_sb, N_CBPSS_l), dtype=np.int8)
            n_rounds = N_CBPSS_l // s_l
            for round_idx in range(n_rounds):
                for l in range(N_sb):
                    src_start = round_idx * N_sb * s_l + l * s_l
                    src_end = src_start + s_l
                    dst_start = round_idx * s_l
                    dst_end = dst_start + s_l
                    sb_bits[l, dst_start:dst_end] = (
                        sym_bits[src_start:src_end]
                    )

            # Per-subblock constellation map + LDPC tone map, then
            # concatenate in subblock order (segment deparser, Eq. 36-76)
            data_syms = np.zeros(cfg['N_SD'], dtype=np.complex128)
            for l in range(N_sb):
                sb_syms = constellation_map(sb_bits[l, :], cfg['N_BPSCS'])
                assert len(sb_syms) == N_SD_l, (
                    f"gen_data_field: subblock {l} mapped to "
                    f"{len(sb_syms)} symbols, expected {N_SD_l}"
                )
                sb_mapped = _ldpc_tone_map(sb_syms, BW)
                data_syms[l * N_SD_l: (l + 1) * N_SD_l] = sb_mapped

        # Apply LDPC tone mapping for the N_sb==1 LDPC path (BW 20/40/80)
        # BCC uses identity tone mapping per Eq. 36-74, so skip.
        if N_sb == 1 and cfg['Coding'] == 'LDPC':
            data_syms = _ldpc_tone_map(data_syms, BW)

        # --------------------------------------------------------------
        # Build frequency-domain vector
        # --------------------------------------------------------------
        freq = np.zeros(NFFT, dtype=np.complex128)

        # Map data symbols to data subcarrier indices
        assert len(data_syms) == len(data_idx), (
            f"gen_data_field: data_syms length {len(data_syms)} != "
            f"c.data_indices length {len(data_idx)}. "
            "Check constellation_map output or LDPC tone mapper."
        )
        for i in range(len(data_idx)):
            k = data_idx[i]
            fft_bin = k % NFFT           # MATLAB: mod(k, NFFT) + 1 (1-idx)
            freq[fft_bin] = data_syms[i]

        # --------------------------------------------------------------
        # Insert pilots
        # --------------------------------------------------------------
        # Pilot polarity for symbol sym_idx of the Data field
        p_n = pilot_polarity[
            (sym_idx + pilot_pol_offset) % n_pol
        ]

        # Per-subcarrier pilot base values
        # Psi = [1,1,1,-1,-1,1,1,1] cycling with (sym_idx + pilot_k) mod 8
        # MATLAB: Psi(mod(sym_idx + pp - 1, 8) + 1) with pp 1-indexed
        # Python: Psi[(sym_idx + pp) % 8] with pp 0-indexed
        base_pilots = np.zeros(n_pilots, dtype=np.float64)
        for pp in range(n_pilots):
            base_pilots[pp] = Psi[(sym_idx + pp) % 8]

        for i in range(n_pilots):
            k = pilot_idx[i]
            fft_bin = k % NFFT
            freq[fft_bin] = p_n * base_pilots[i]

        # --------------------------------------------------------------
        # Normalization per Eq. 36-87: divide by sqrt(N_ST)
        # For SU full-BW SISO, the per-tone amplitude factor is
        # 1/sqrt(N_ST) (alpha_r cancels out).
        # --------------------------------------------------------------
        freq = freq / np.sqrt(cfg['N_ST'])

        # --------------------------------------------------------------
        # OFDM modulate
        # --------------------------------------------------------------
        td_sym = ofdm_mod(freq, NFFT, cfg['CP_Data'])
        td[sym_idx * sym_out_len: (sym_idx + 1) * sym_out_len] = td_sym

    return td
