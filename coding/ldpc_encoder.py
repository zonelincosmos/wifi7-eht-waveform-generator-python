# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
IEEE 802.11 LDPC encoder per Annex F and Section 19.3.11.7.

Supports two modes:
  - **Simple mode**: pass a scalar codeword length (648/1296/1944).
    Pads info bits to fill codewords, encodes, and concatenates.
  - **Full mode**: pass an ``LdpcParams`` dataclass from ``ldpc_params``.
    Applies shortening, puncturing, and repetition per Section 19.3.11.7.5.

For each codeword the encoder computes
``parity = P @ info (mod 2)`` using the encoding matrix from
``ldpc_matrices.load_ldpc_enc_matrix``.
"""

import numpy as np

from .ldpc_matrices import load_ldpc_enc_matrix
from .ldpc_params import LdpcParams


def ldpc_encoder(info_bits, code_rate, cw_len_or_params):
    """LDPC encoder with shortening, puncturing, and repetition.

    Parameters
    ----------
    info_bits : array_like
        Binary information bits (1-D).
    code_rate : tuple of (int, int)
        ``(R_num, R_den)`` -- code rate as numerator/denominator.
    cw_len_or_params : int or LdpcParams
        - ``int``: codeword length (648, 1296, or 1944) for simple mode.
        - ``LdpcParams``: full-mode with shortening/puncturing/repetition.

    Returns
    -------
    numpy.ndarray
        Encoded bit vector with dtype ``np.int8``.

    Raises
    ------
    ValueError
        If ``cw_len_or_params`` is neither an int nor an LdpcParams instance.
    """
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    r_num, r_den = code_rate
    r = r_num / r_den

    # --- Parse third argument ---
    if isinstance(cw_len_or_params, LdpcParams):
        lp = cw_len_or_params
        codeword_len = lp.L_LDPC
        use_full_mode = True
    elif isinstance(cw_len_or_params, (int, np.integer)):
        codeword_len = int(cw_len_or_params)
        use_full_mode = False
    else:
        raise ValueError(
            "Third argument must be a codeword length (int) or LdpcParams "
            "dataclass from ldpc_params()."
        )

    # Number of info / parity bits per codeword
    k = round(codeword_len * r)
    n = codeword_len
    m = n - k  # parity bits

    # Load encoding matrix P (M x K)
    p_enc = load_ldpc_enc_matrix(n, r_num, r_den)

    if not use_full_mode:
        # --- Simple mode ---
        n_info = len(info_bits)
        n_cw = max(1, -(-n_info // k))  # ceil division
        padded = np.zeros(n_cw * k, dtype=np.int8)
        padded[:n_info] = info_bits

        parts = []
        for cw in range(n_cw):
            s = padded[cw * k:(cw + 1) * k]
            p = (p_enc @ s.astype(np.uint8)) % 2
            parts.append(s)
            parts.append(p.astype(np.int8))

        return np.concatenate(parts)

    else:
        # --- Full mode: shortening + puncturing + repetition ---
        n_cw = lp.N_CW
        n_shrt = lp.N_shrt
        n_punc = lp.N_punc
        n_rep = lp.N_rep

        # Distribute shortening evenly across codewords (step c)
        shrt_base = n_shrt // n_cw
        shrt_extra = n_shrt % n_cw

        # Distribute puncturing evenly (step d)
        punc_base = n_punc // n_cw
        punc_extra = n_punc % n_cw

        # Distribute repetition evenly (step e)
        rep_base = n_rep // n_cw
        rep_extra = n_rep % n_cw

        parts = []
        info_pos = 0

        for cw in range(n_cw):
            # Per-codeword counts (1-indexed in MATLAB, 0-indexed here)
            s_shrt = shrt_base + (1 if (cw + 1) <= shrt_extra else 0)
            s_punc = punc_base + (1 if (cw + 1) <= punc_extra else 0)
            s_rep = rep_base + (1 if (cw + 1) <= rep_extra else 0)

            # Number of actual info bits in this codeword
            k_actual = k - s_shrt

            # Extract info bits
            if info_pos + k_actual <= len(info_bits):
                s_info = info_bits[info_pos:info_pos + k_actual].copy()
            else:
                remaining = max(0, len(info_bits) - info_pos)
                s_info = np.zeros(k_actual, dtype=np.int8)
                if remaining > 0:
                    s_info[:remaining] = info_bits[info_pos:info_pos + remaining]
            info_pos += k_actual

            # Append shortening zeros, encode full K-bit block
            s_full = np.zeros(k, dtype=np.int8)
            s_full[:k_actual] = s_info
            p = (p_enc @ s_full.astype(np.uint8)) % 2

            # Build output: [info (without shortened zeros) | parity]
            cw_out = np.concatenate([s_info, p.astype(np.int8)])

            # Puncture: remove last s_punc parity bits
            if s_punc > 0:
                cw_out = cw_out[:len(cw_out) - s_punc]

            # Repeat: cyclically copy the shortened codeword (info+parity)
            if s_rep > 0:
                cw_len = len(cw_out)
                n_full_rep = s_rep // cw_len
                n_rem = s_rep % cw_len
                rep_parts = [cw_out]
                if n_full_rep > 0:
                    rep_parts.append(np.tile(cw_out, n_full_rep))
                if n_rem > 0:
                    rep_parts.append(cw_out[:n_rem])
                cw_out = np.concatenate(rep_parts)

            parts.append(cw_out)

        return np.concatenate(parts)
