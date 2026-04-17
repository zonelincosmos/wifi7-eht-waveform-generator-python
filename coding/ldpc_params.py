# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
LDPC encoding parameter computation per IEEE 802.11-2024 Section 19.3.11.7.5
with EHT-specific extra-symbol handling per IEEE 802.11be-2024 Section 36.3.13.3.5.

Computes: L_LDPC, N_CW, N_shrt, N_punc, N_rep, N_SYM, N_avbits, N_pld.

Adapted for EHT SU PPDU (SISO, 1 spatial stream, m_STBC = 1).
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class LdpcParams:
    """Container for LDPC encoding parameters."""
    L_LDPC: int
    N_CW: int
    N_shrt: int
    N_punc: int
    N_rep: int
    N_SYM: int
    N_avbits: int
    N_pld: int
    N_pld_raw: int
    has_extra_symbol: bool
    a_init_used: Optional[int]


def ldpc_params(n_cbps, r_num, r_den, psdu_bytes, n_service,
                n_cbps_short=None, a_init_in=None):
    """Compute LDPC encoding parameters per Sections 19.3.11.7.5 and 36.3.13.3.5.

    Parameters
    ----------
    n_cbps : int
        Number of coded bits per symbol (N_CBPS).
    r_num : int
        Code rate numerator.
    r_den : int
        Code rate denominator.
    psdu_bytes : int
        PSDU length in bytes.
    n_service : int
        Number of SERVICE bits (typically 16).
    n_cbps_short : int, optional
        N_CBPS,short,u per Section 36.3.13.3.5.  Required for
        spec-compliant EHT extra-symbol handling when a_init < 4.
    a_init_in : int, optional
        Pre-FEC padding factor a_init per Eq. 36-48.  If omitted but
        ``n_cbps_short`` is supplied, a_init is computed internally.

    Returns
    -------
    LdpcParams
        Dataclass with all computed LDPC parameters.
    """
    r = r_num / r_den
    m_stbc = 1  # No STBC for EHT SU SISO

    # --- Step (a): raw N_pld, initial N_SYM, then EHT N_pld override ---
    # Raw payload: 8*PSDU + SERVICE (Eq. 36-47 input, no tail for LDPC)
    n_pld_raw = 8 * psdu_bytes + n_service

    n_dbps = int(n_cbps * r)  # floor(N_CBPS * R)

    # Initial number of OFDM symbols (Eq. 36-49)
    n_sym = math.ceil(n_pld_raw / (n_dbps * m_stbc)) * m_stbc
    n_sym_init = n_sym

    # Start with raw N_pld; will be overridden by Eq. 36-54 below when
    # EHT parameters (a_init, N_CBPS_short) are available.
    n_pld = n_pld_raw

    # --- Resolve a_init for Eq. 36-56/58 ---
    if a_init_in is not None:
        a_init = a_init_in
    elif n_cbps_short is not None:
        n_dbps_short = int(n_cbps_short * r)
        n_excess = n_pld % n_dbps  # N_tail = 0 for LDPC
        if n_excess == 0:
            a_init = 4
        else:
            a_init = min(math.ceil(n_excess / n_dbps_short), 4)
    else:
        a_init = None

    # --- EHT override: N_pld (Eq. 36-54) and N_avbits (Eq. 36-55) ---
    # For EHT, N_pld INCLUDES pre-FEC padding (Eq. 36-63) and is LARGER
    # than the raw payload.  gen_data_field must prepend matching zero
    # pre-FEC padding so the LDPC encoder receives exactly N_pld info bits.
    if a_init is not None and n_cbps_short is not None:
        n_dbps_short_local = int(n_cbps_short * r)
        if a_init == 4:
            n_dbps_last_init = n_dbps                           # Eq. 36-52 top
            n_cbps_last_init = n_cbps                           # Eq. 36-53 top
        else:
            n_dbps_last_init = a_init * n_dbps_short_local      # Eq. 36-52 bottom
            n_cbps_last_init = a_init * n_cbps_short            # Eq. 36-53 bottom
        n_pld = (n_sym - m_stbc) * n_dbps + m_stbc * n_dbps_last_init      # Eq. 36-54
        n_avbits = (n_sym - m_stbc) * n_cbps + m_stbc * n_cbps_last_init   # Eq. 36-55
    else:
        # HT fallback (spec-exact when a_init = 4)
        n_avbits = n_cbps * n_sym

    # --- Step (b): Select N_CW and L_LDPC per Table 19-16 ---
    if n_avbits <= 648:
        n_cw = 1
        if n_avbits >= n_pld + 912 * (1 - r):
            l_ldpc = 1296
        else:
            l_ldpc = 648

    elif n_avbits <= 1296:
        n_cw = 1
        if n_avbits >= n_pld + 1464 * (1 - r):
            l_ldpc = 1944
        else:
            l_ldpc = 1296

    elif n_avbits <= 1944:
        n_cw = 1
        l_ldpc = 1944

    elif n_avbits <= 2592:
        n_cw = 2
        if n_avbits >= n_pld + 2916 * (1 - r):
            l_ldpc = 1944
        else:
            l_ldpc = 1296

    else:
        # n_avbits > 2592
        n_cw = math.ceil(n_pld / (1944 * r))
        l_ldpc = 1944

    # --- Step (c): shortening bits (Eq. 19-37) ---
    # Cast explicitly to int: round() returns a float, but downstream LDPC
    # loops index arrays with n_shrt / n_punc / n_rep, and the emitted
    # params dict is compared against MATLAB integer cfg fields.
    n_shrt = int(max(0, round(n_cw * l_ldpc * r) - n_pld))

    # --- Step (d): puncturing bits (Eq. 19-38) ---
    n_punc = int(max(0, round(n_cw * l_ldpc) - n_avbits - n_shrt))

    # Check for excessive puncturing
    parity_total = n_cw * l_ldpc * (1 - r)
    cond1 = (n_punc > 0.1 * parity_total) and \
            (n_shrt < 1.2 * n_punc * r / (1 - r))
    cond2 = (n_punc > 0.3 * parity_total)

    has_extra_symbol = cond1 or cond2

    if has_extra_symbol:
        if n_cbps_short is None or a_init is None:
            # HT-style fallback (Eq. 19-39 / 19-40)
            n_avbits = n_avbits + n_cbps * m_stbc
            n_sym = n_sym + m_stbc
        else:
            # Spec-compliant path (Eq. 36-56 + 36-58)
            if a_init == 3:
                n_avbits = n_avbits + n_cbps - 3 * n_cbps_short
            else:
                n_avbits = n_avbits + n_cbps_short

            if a_init == 4:
                n_sym = n_sym_init + 1
            else:
                n_sym = n_sym_init

        # Recompute N_punc (Eq. 36-57 / Eq. 19-40)
        n_punc = int(max(0, round(n_cw * l_ldpc) - n_avbits - n_shrt))

    # --- Step (e): repeated coded bits (Eq. 19-42) ---
    n_rep = int(max(0, n_avbits - round(n_cw * l_ldpc * (1 - r)) - n_pld))

    return LdpcParams(
        L_LDPC=l_ldpc,
        N_CW=n_cw,
        N_shrt=n_shrt,
        N_punc=n_punc,
        N_rep=n_rep,
        N_SYM=n_sym,
        N_avbits=n_avbits,
        N_pld=n_pld,
        N_pld_raw=n_pld_raw,
        has_extra_symbol=has_extra_symbol,
        a_init_used=a_init,
    )
