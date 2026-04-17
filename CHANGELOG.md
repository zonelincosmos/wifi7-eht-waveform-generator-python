# Changelog

## v1.0.0 - 2026-04-18

Initial public release.

### Supported
- All five 802.11be bandwidths: 20, 40, 80, 160, 320 MHz
- All MCS 0..13 (BPSK 1/2 through 4096-QAM 5/6)
- Guard intervals 0.8, 1.6, 3.2 us; EHT-LTF 2x / 4x
- BCC and LDPC with spec-compliant shortening / puncturing /
  repetition and LDPC extra-symbol logic (Eq. 36-56 / 36-58)
- A-MPDU framing with pre-FEC padding split (Eq. 36-66 / 36-67)
- SISO, 1 spatial stream

### Verified
- Bit-true (< 1e-12) across 183 (BW, MCS, GI, LTF, payload)
  configurations against an independent reference implementation.
