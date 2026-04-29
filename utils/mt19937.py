# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
Mersenne Twister (mt19937ar) re-implementation that matches the
``init_by_array`` seeding convention used by the reference C
implementation of Matsumoto & Nishimura.

Numpy's ``np.random.RandomState(seed)`` does NOT match this convention
(numpy uses ``init_genrand(seed)`` style), so a separate implementation
is required when byte-stream parity with the reference is needed.

The :func:`randi_uint8` helper reproduces ``rng(seed); randi([0 255],
1, n, 'uint8')`` byte-for-byte.

References
----------
- Matsumoto, M. and Nishimura, T. (1998), "Mersenne twister: a
  623-dimensionally equidistributed uniform pseudorandom number
  generator", ACM Trans. Model. Comput. Simul., Vol. 8, No. 1.
- Reference C source ``mt19937ar.c`` (init_by_array + genrand_int32 +
  genrand_res53).
"""

from __future__ import annotations

import numpy as np


_N = 624
_M = 397
_MATRIX_A   = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_UINT32_MASK = 0xFFFFFFFF


class MT19937:
    """Mersenne Twister mt19937ar with the reference ``init_by_array`` seeding.

    Parameters
    ----------
    seed_key : int or sequence of ints
        If int, equivalent to ``init_by_array([seed_key])``.  Otherwise the
        full key array is used.
    """

    __slots__ = ("_mt", "_index")

    def __init__(self, seed_key):
        self._mt = [0] * _N
        self._index = _N + 1
        if isinstance(seed_key, int):
            # Scalar seed: use init_genrand (matches the reference RNG's
            # rng(seed) convention).  By convention seed=0 selects the
            # mt19937ar reference default seed value (5489), matching the
            # spec-reference's rng(0) behavior.
            s = 5489 if seed_key == 0 else (seed_key & _UINT32_MASK)
            self._init_genrand(s)
        else:
            self._init_by_array(list(seed_key))

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _init_genrand(self, s):
        """Internal: scalar seed init (used as the first stage of init_by_array)."""
        self._mt[0] = s & _UINT32_MASK
        for i in range(1, _N):
            prev = self._mt[i - 1]
            self._mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & _UINT32_MASK
        self._index = _N

    def _init_by_array(self, key):
        """Reference ``init_by_array`` from mt19937ar.c."""
        self._init_genrand(19650218)
        key_length = len(key)
        i = 1
        j = 0
        k = max(_N, key_length)
        while k:
            prev = self._mt[i - 1]
            self._mt[i] = (
                (self._mt[i] ^ ((prev ^ (prev >> 30)) * 1664525)) + key[j] + j
            ) & _UINT32_MASK
            i += 1
            j += 1
            if i >= _N:
                self._mt[0] = self._mt[_N - 1]
                i = 1
            if j >= key_length:
                j = 0
            k -= 1
        for k in range(_N - 1, 0, -1):
            prev = self._mt[i - 1]
            self._mt[i] = (
                (self._mt[i] ^ ((prev ^ (prev >> 30)) * 1566083941)) - i
            ) & _UINT32_MASK
            i += 1
            if i >= _N:
                self._mt[0] = self._mt[_N - 1]
                i = 1
        # Most significant bit is 1; assuring non-zero initial array.
        self._mt[0] = 0x80000000
        self._index = _N

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self):
        """Fill the state with the next ``_N`` 32-bit words."""
        mt = self._mt
        for kk in range(_N - _M):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
        for kk in range(_N - _M, _N - 1):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
        y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
        mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if (y & 1) else 0)
        self._index = 0

    def next_uint32(self):
        """Return one tempered 32-bit MT19937 output."""
        if self._index >= _N:
            self._generate()
        y = self._mt[self._index]
        self._index += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & _UINT32_MASK

    def next_double(self):
        """Return one 53-bit double in ``[0, 1)`` (genrand_res53)."""
        a = self.next_uint32() >> 5    # 27 bits
        b = self.next_uint32() >> 6    # 26 bits
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)


# ----------------------------------------------------------------------
# High-level helpers
# ----------------------------------------------------------------------

def randi_uint8(seed, n):
    """Reproduce ``rng(seed); randi([0 255], 1, n, 'uint8')`` byte-for-byte.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.  ``rng(0)`` -> ``seed=0``.
    n : int
        Number of uint8 samples to generate.

    Returns
    -------
    numpy.ndarray
        Length-``n`` uint8 array.
    """
    rng = MT19937(seed)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        out[i] = int(rng.next_double() * 256.0)
    return out
