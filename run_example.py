#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 zonelincosmos
# Part of wifi7-eht-waveform-generator-python, an IEEE 802.11be EHT SU
# waveform generator.  See LICENSE in the repo root.
"""
802.11be EHT SU PPDU Waveform Generator - Example

IEEE Std 802.11be-2024 (Wi-Fi 7)
SISO 1SS, configurable BW/MCS/GI

Generates a waveform and produces three diagnostic plots:

    1. Time-domain magnitude with field boundaries annotated
       -> eht_waveform_time.png
    2. Power spectral density (Welch-style)
       -> eht_waveform_psd.png
    3. Constellation diagram (first few Data OFDM symbols after DFT)
       -> eht_waveform_constellation.png

The waveform itself is saved as ``eht_waveform.npz`` next to this script.

Plots are always written to disk.  If an interactive matplotlib backend
is available they are also shown on screen; otherwise the script still
produces the PNGs using the Agg backend.
"""

import os
import sys

import numpy as np

# Ensure the package root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eht_waveform_gen import eht_waveform_gen
from eht_constants import eht_constants


# --- matplotlib backend selection -----------------------------------------
# Prefer the default interactive backend; fall back to the headless Agg
# backend when no GUI is available so PNG output still succeeds.
HAS_GUI = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
    try:
        plt.figure()
        plt.close('all')
    except Exception:
        matplotlib.use('Agg')
        HAS_GUI = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    HAS_GUI = False
    print('matplotlib not available -- skipping plots.')


# =========================================================================
#  Configuration
# =========================================================================
BW           = 320    # Bandwidth: 20, 40, 80, 160, 320 MHz
MCS          = 13     # EHT-MCS: 0-13
PayloadBytes = 5000   # PSDU length in bytes

# --- GI + EHT-LTF combination (Table 36-36, IEEE 802.11be-2024 p.784) ---
# Only these four (LTFType, GI) pairs are valid:
#   mode 0: 2x EHT-LTF + 0.8 us GI
#   mode 1: 2x EHT-LTF + 1.6 us GI
#   mode 2: 4x EHT-LTF + 0.8 us GI
#   mode 3: 4x EHT-LTF + 3.2 us GI  <-- primary target
gi_ltf_mode = 3

if gi_ltf_mode == 0:
    GI = 0.8;  LTFType = 2
elif gi_ltf_mode == 1:
    GI = 1.6;  LTFType = 2
elif gi_ltf_mode == 2:
    GI = 0.8;  LTFType = 4
elif gi_ltf_mode == 3:
    GI = 3.2;  LTFType = 4
else:
    raise ValueError('gi_ltf_mode must be 0, 1, 2, or 3')


# =========================================================================
#  Generate Waveform
# =========================================================================
# Use a deterministic cycling byte pattern for the user payload so that
# this script produces the same waveform byte-for-byte on every
# platform, independent of the underlying Mersenne Twister
# implementation (MATLAB's rng and numpy's MT19937 produce different
# streams for the same seed, so the "random" default PSDU is not
# portable across environments).  The pattern 0, 1, 2, ..., 255, 0, 1,
# ... is easy to reproduce on any reference implementation.
user_payload = np.arange(PayloadBytes, dtype=np.uint8)  # cycles mod 256

waveform, cfg, psdu_out = eht_waveform_gen(
    BW=BW, MCS=MCS, GI=GI, LTFType=LTFType, PayloadBytes=PayloadBytes,
    PSDU=user_payload,
)


# =========================================================================
#  Output paths
# =========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))


def _save_fig(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  saved: {path}')


# =========================================================================
#  Plot 1: Time-domain waveform
# =========================================================================
if HAS_MPL:
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 4))
    if HAS_GUI:
        try:
            fig1.canvas.manager.set_window_title(
                '802.11be EHT PPDU - Time Domain'
            )
        except Exception:
            pass

    t_us = np.arange(len(waveform)) / cfg['Fs'] * 1e6
    ax1.plot(t_us, np.abs(waveform), linewidth=0.5)
    ax1.set_xlabel('Time (us)')
    ax1.set_ylabel('|waveform|')
    ax1.set_title(
        f'802.11be EHT SU PPDU | BW={cfg["BW"]} MHz | MCS={cfg["MCS"]} | '
        f'{cfg["Coding"]} | GI={cfg["GI"]:.1f} us'
    )
    ax1.grid(True, alpha=0.3)

    field_names = [
        'L-STF', 'L-LTF', 'L-SIG', 'RL-SIG', 'U-SIG',
        'EHT-SIG', 'EHT-STF', 'EHT-LTF', 'Data', 'PE',
    ]
    field_keys = [
        'LSTF', 'LLTF', 'LSIG', 'RLSIG', 'USIG',
        'EHTSIG', 'EHTSTF', 'EHTLTF', 'Data', 'PE',
    ]
    field_lens = [cfg['FieldLengths'][k] for k in field_keys]
    cumlen = np.concatenate([[0], np.cumsum(field_lens)])

    cmap = plt.get_cmap('tab10', len(field_names))
    ymax = np.max(np.abs(waveform)) * 1.1
    ax1.set_ylim(0, ymax)

    for i in range(len(field_names)):
        if field_lens[i] == 0:
            continue
        t_start = cumlen[i] / cfg['Fs'] * 1e6
        t_end = cumlen[i + 1] / cfg['Fs'] * 1e6
        color = cmap(i)
        ax1.axvspan(t_start, t_end, alpha=0.08, color=color)
        ax1.axvline(t_start, linestyle='--', color=color, alpha=0.6,
                    linewidth=0.8)
        ax1.text((t_start + t_end) / 2, ymax * 0.95, field_names[i],
                 ha='center', fontsize=8, color=color,
                 fontweight='bold')

    fig1.tight_layout()
    _save_fig(fig1, 'eht_waveform_time.png')


# =========================================================================
#  Plot 2: Power Spectral Density (Welch-style via numpy FFT)
# =========================================================================
if HAS_MPL:
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    if HAS_GUI:
        try:
            fig2.canvas.manager.set_window_title(
                '802.11be EHT PPDU - PSD'
            )
        except Exception:
            pass

    NFFT_psd = 4096
    win = np.hanning(NFFT_psd)
    hop = NFFT_psd // 2
    n_seg = max(1, (len(waveform) - NFFT_psd) // hop + 1)

    psd_acc = np.zeros(NFFT_psd, dtype=np.float64)
    segs_used = 0
    for seg_idx in range(n_seg):
        start = seg_idx * hop
        end = start + NFFT_psd
        if end > len(waveform):
            break
        seg = waveform[start:end] * win
        X = np.fft.fftshift(np.fft.fft(seg, NFFT_psd))
        psd_acc += np.abs(X) ** 2
        segs_used += 1

    psd_acc /= max(1, segs_used)
    psd_acc /= np.sum(win ** 2)

    f_axis = np.fft.fftshift(
        np.fft.fftfreq(NFFT_psd, d=1.0 / cfg['Fs'])
    ) / 1e6  # MHz

    psd_dB = 10 * np.log10(psd_acc / np.max(psd_acc) + 1e-30)

    ax2.plot(f_axis, psd_dB, linewidth=0.8)
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('PSD (dB, normalized)')
    ax2.set_title(
        f'Power Spectral Density | BW={cfg["BW"]} MHz | '
        f'Fs={cfg["Fs"]/1e6:.0f} MHz | {segs_used} avg segments'
    )
    ax2.set_xlim(-cfg['BW'] / 2 * 1.3, cfg['BW'] / 2 * 1.3)
    ax2.set_ylim(-60, 5)
    ax2.grid(True, alpha=0.3)

    # Nominal signal bandwidth boundary
    ax2.axvline(-cfg['BW'] / 2, color='r', linestyle='--', linewidth=1,
                label=f'+-BW/2 ({cfg["BW"]//2} MHz)')
    ax2.axvline(cfg['BW'] / 2, color='r', linestyle='--', linewidth=1)
    ax2.legend(loc='upper right')

    fig2.tight_layout()
    _save_fig(fig2, 'eht_waveform_psd.png')


# =========================================================================
#  Plot 3: Constellation Diagram (Data field)
# =========================================================================
if HAS_MPL:
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
    if HAS_GUI:
        try:
            fig3.canvas.manager.set_window_title(
                '802.11be EHT PPDU - Constellation'
            )
        except Exception:
            pass

    # Data field starts at cumulative offset of all pre-Data fields.
    pre_data_len = sum(cfg['FieldLengths'][k] for k in
                       ['LSTF', 'LLTF', 'LSIG', 'RLSIG',
                        'USIG', 'EHTSIG', 'EHTSTF', 'EHTLTF'])
    data_samples = waveform[pre_data_len: pre_data_len +
                            cfg['FieldLengths']['Data']]

    c = eht_constants(cfg['BW'])
    sym_len = cfg['NFFT'] + cfg['CP_Data']
    # Match MATLAB run_example.m: plot the first 5 Data OFDM symbols
    # (clamped to N_SYM) with solid 3-pt markers and no transparency.
    n_syms_to_plot = min(5, cfg['N_SYM'])

    all_data_syms = []
    for s in range(n_syms_to_plot):
        start_idx = s * sym_len + cfg['CP_Data']
        end_idx = start_idx + cfg['NFFT']
        if end_idx > len(data_samples):
            break
        td_sym = data_samples[start_idx:end_idx]
        freq_sym = np.fft.fft(td_sym, cfg['NFFT']) / np.sqrt(cfg['NFFT'])
        # Re-scale back to constellation amplitude (Eq. 36-87 normalises
        # by 1/sqrt(N_ST); undo that for display only).
        for k in c['data_indices']:
            fft_bin = k % cfg['NFFT']
            all_data_syms.append(freq_sym[fft_bin] * np.sqrt(cfg['N_ST']))

    if len(all_data_syms) > 0:
        all_data_syms = np.array(all_data_syms)
        ax3.plot(np.real(all_data_syms), np.imag(all_data_syms), '.',
                 markersize=3)
        ax3.set_aspect('equal', adjustable='box')
        ax3.grid(True)
        ax3.set_title(
            f'Constellation | MCS={cfg["MCS"]} '
            f'({cfg["ModOrder"]}-QAM, R={cfg["R_num"]}/{cfg["R_den"]})'
        )
        ax3.set_xlabel('In-phase')
        ax3.set_ylabel('Quadrature')
        max_val = np.max(np.abs(np.concatenate([
            np.real(all_data_syms), np.imag(all_data_syms)
        ]))) * 1.2
        ax3.set_xlim(-max_val, max_val)
        ax3.set_ylim(-max_val, max_val)

    fig3.tight_layout()
    _save_fig(fig3, 'eht_waveform_constellation.png')


# =========================================================================
#  Summary
# =========================================================================
print()
print('=== Summary ===')
print('802.11be EHT SU PPDU (Wi-Fi 7)')
print(f'  BW = {cfg["BW"]} MHz, MCS = {cfg["MCS"]}, '
      f'{cfg["Coding"]}, GI = {cfg["GI"]:.1f} us')
print(f'  Modulation: {cfg["ModOrder"]}-QAM, '
      f'Rate: {cfg["R_num"]}/{cfg["R_den"]}')
print(f'  APEP_LENGTH: {cfg["PayloadBytes"]} bytes | '
      f'PSDU (framed): {len(psdu_out)} bytes | '
      f'{cfg["N_SYM"]} OFDM symbols')
print(f'  Waveform: {len(waveform)} samples, '
      f'{len(waveform)/cfg["Fs"]*1e6:.2f} us, '
      f'Fs = {cfg["Fs"]/1e6:.1f} MHz')


# =========================================================================
#  Save waveform
# =========================================================================
output_file = os.path.join(HERE, 'eht_waveform.npz')
np.savez(output_file, waveform=waveform, psdu=psdu_out)
print(f'\nWaveform saved to: {output_file}')


# =========================================================================
#  Show plots (interactive backend only)
# =========================================================================
if HAS_MPL and HAS_GUI:
    plt.show()
elif HAS_MPL:
    print('Headless backend: PNG files written next to the script.')
