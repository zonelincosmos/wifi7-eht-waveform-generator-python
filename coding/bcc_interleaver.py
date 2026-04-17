# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
BCC frequency interleaver for 802.11be EHT data field.

Implements the three-stage frequency interleaver per IEEE 802.11-2024
Section 36.3.13.6 (references Table 27-36).  BCC is only applicable
for RU <= 242 tones (20 MHz bandwidth).

Permutations:
  1st: adjacent coded bits -> non-adjacent subcarriers
  2nd: adjacent bits -> alternating significant constellation bits
  3rd: frequency rotation (trivial for 1 spatial stream: j_rot = 0)
"""

import numpy as np


def bcc_interleaver(bits, n_cbps, n_bpscs, bw):
    """BCC frequency interleaver for 242-tone RU.

    Parameters
    ----------
    bits : array_like
        Input coded bit vector of length ``n_cbps``.
    n_cbps : int
        Number of coded bits per symbol.
    n_bpscs : int
        Number of coded bits per subcarrier (log2 of modulation order).
    bw : int
        Bandwidth in MHz (only 20 is supported for BCC).

    Returns
    -------
    numpy.ndarray
        Interleaved bit vector of length ``n_cbps`` with dtype ``np.int8``.

    Raises
    ------
    ValueError
        If ``bw`` is not 20, or if input length does not match ``n_cbps``.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()

    if bw != 20:
        raise ValueError(
            f"BCC interleaver is only defined for BW=20 (242-tone RU) "
            f"per the EHT spec. For BW>20 the spec requires LDPC. "
            f"Got BW={bw}."
        )

    # Table 27-36 parameters for 242-tone RU
    n_col = 26
    n_row = 9 * n_bpscs
    actual_ncbps = n_row * n_col
    s = max(n_bpscs // 2, 1)

    if len(bits) != n_cbps or n_cbps != actual_ncbps:
        raise ValueError(
            f"bcc_interleaver expects len(bits)==n_cbps==n_row*n_col. "
            f"Got len(bits)={len(bits)}, n_cbps={n_cbps}, "
            f"actual_ncbps={actual_ncbps}."
        )

    # First permutation: k -> i
    k = np.arange(actual_ncbps)
    i_perm = n_row * (k % n_col) + k // n_col

    # Second permutation: i -> j
    j_perm = (s * (i_perm // s)
              + (i_perm + actual_ncbps - n_col * i_perm // actual_ncbps) % s)

    # Apply permutation: out[j_perm[k]] = bits[k]
    out = np.zeros(actual_ncbps, dtype=np.int8)
    out[j_perm] = bits

    return out
