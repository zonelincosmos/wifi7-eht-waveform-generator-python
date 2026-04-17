# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
BCC puncturing from rate 1/2 to higher code rates.

Implements puncturing per IEEE 802.11-2024 Section 17.3.5.6,
Table 17-24.  Supported target rates: 1/2, 2/3, 3/4, 5/6.

Input is a rate-1/2 encoded bit vector with interleaved A/B outputs
([A0, B0, A1, B1, ...]).  Puncturing selectively drops bits according
to the puncture pattern for the desired rate.
"""

import numpy as np


def bcc_puncture(encoded, r_num, r_den):
    """Puncture rate-1/2 BCC encoded bits to a higher code rate.

    Parameters
    ----------
    encoded : array_like
        Rate-1/2 encoded bits (even length, interleaved A/B).
    r_num : int
        Target code rate numerator.
    r_den : int
        Target code rate denominator.

    Returns
    -------
    numpy.ndarray
        Punctured bit vector with dtype ``np.int8``.

    Raises
    ------
    ValueError
        If the code rate is not supported.
    """
    encoded = np.asarray(encoded, dtype=np.int8).ravel()

    if r_num == 1 and r_den == 2:
        # No puncturing needed
        return encoded.copy()

    # Separate A and B streams
    n_half = len(encoded) // 2
    a = encoded[0::2]  # output A
    b = encoded[1::2]  # output B

    if r_num == 2 and r_den == 3:
        # Puncture pattern [1,1; 1,0] over 2 input bits
        # Keep: A0, B0, A1 (drop B1)  ->  3 bits per 2 input bits
        n_groups = n_half // 2
        out = np.zeros(n_groups * 3, dtype=np.int8)
        for g in range(n_groups):
            out[g * 3] = a[g * 2]
            out[g * 3 + 1] = b[g * 2]
            out[g * 3 + 2] = a[g * 2 + 1]
        # Handle remaining bits
        rem = n_half % 2
        if rem > 0:
            tail = np.array([a[n_groups * 2], b[n_groups * 2]], dtype=np.int8)
            out = np.concatenate([out, tail])

    elif r_num == 3 and r_den == 4:
        # Puncture pattern [1,1,0; 1,0,1] over 3 input bits
        # Keep: A0, B0, A1, B2 (drop B1, A2)  ->  4 bits per 3 input bits
        n_groups = n_half // 3
        out = np.zeros(n_groups * 4, dtype=np.int8)
        for g in range(n_groups):
            out[g * 4] = a[g * 3]
            out[g * 4 + 1] = b[g * 3]
            out[g * 4 + 2] = a[g * 3 + 1]
            out[g * 4 + 3] = b[g * 3 + 2]
        # Handle remaining bits
        rem = n_half % 3
        if rem >= 1:
            tail = [a[n_groups * 3], b[n_groups * 3]]
            if rem >= 2:
                tail.append(a[n_groups * 3 + 1])
            out = np.concatenate([out, np.array(tail, dtype=np.int8)])

    elif r_num == 5 and r_den == 6:
        # Puncture pattern [1,1,0,1,0; 1,0,1,0,1] over 5 input bits
        # Keep: A0, B0, A1, B2, A3, B4  ->  6 bits per 5 input bits
        n_groups = n_half // 5
        out = np.zeros(n_groups * 6, dtype=np.int8)
        for g in range(n_groups):
            out[g * 6] = a[g * 5]
            out[g * 6 + 1] = b[g * 5]
            out[g * 6 + 2] = a[g * 5 + 1]
            out[g * 6 + 3] = b[g * 5 + 2]
            out[g * 6 + 4] = a[g * 5 + 3]
            out[g * 6 + 5] = b[g * 5 + 4]
        # Handle remaining bits
        rem = n_half % 5
        tail = []
        if rem >= 1:
            tail.extend([a[n_groups * 5], b[n_groups * 5]])
        if rem >= 2:
            tail.append(a[n_groups * 5 + 1])
        if rem >= 3:
            tail.append(b[n_groups * 5 + 2])
        if rem >= 4:
            tail.append(a[n_groups * 5 + 3])
        if tail:
            out = np.concatenate([out, np.array(tail, dtype=np.int8)])

    else:
        raise ValueError(f"Unsupported code rate {r_num}/{r_den}")

    return out
