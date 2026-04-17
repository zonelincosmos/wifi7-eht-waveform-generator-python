# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
OFDM modulation: map frequency-domain subcarriers to time-domain samples.

    td = ofdm_mod(freq_data, NFFT, CP_len)

Maps frequency-domain subcarrier values to time-domain samples via IFFT
with cyclic prefix insertion.

IEEE Std 802.11be-2024, Section 36.3.13.
"""

import numpy as np


def ofdm_mod(freq_data, NFFT, CP_len):
    """OFDM modulate: IFFT + cyclic prefix insertion.

    Parameters
    ----------
    freq_data : array_like
        Frequency-domain vector of length NFFT, already in FFT bin order:
        ``freq_data[0]`` = DC (subcarrier 0), up to
        ``freq_data[NFFT-1]`` = subcarrier -1.
        Subcarrier k maps to bin ``k % NFFT``.
    NFFT : int
        FFT size (256, 512, 1024, 2048, or 4096).
    CP_len : int
        Cyclic prefix length in samples.

    Returns
    -------
    numpy.ndarray
        Time-domain samples with CP prepended, 1-D complex array of
        length ``NFFT + CP_len``.
    """
    freq_data = np.asarray(freq_data, dtype=np.complex128).ravel()

    if len(freq_data) != NFFT:
        raise ValueError(
            f"freq_data length ({len(freq_data)}) must equal NFFT ({NFFT})"
        )

    # IFFT with sqrt(NFFT) scaling to match MATLAB convention
    # MATLAB: ifft(x, NFFT) * sqrt(NFFT)
    td_sym = np.fft.ifft(freq_data, NFFT) * np.sqrt(NFFT)

    # Prepend cyclic prefix: last CP_len samples of td_sym
    td = np.concatenate([td_sym[-CP_len:], td_sym])

    return td
