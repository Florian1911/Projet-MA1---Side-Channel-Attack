# High-side supply stability analysis

Dataset: `dataset_aes_sca_highside.npz`

Acquisition summary:
- 1000 traces
- 2000 samples per trace
- sample interval: 80 ns
- capture duration: 160 us
- no PicoScope overflow reported
- channels:
  - `traces_a`: supply-side shunt voltage, around 3.305 V
  - `traces_b`: board-side shunt voltage, around 3.334 V
  - `traces`: differential signal `ChA - ChB`, around -29.5 mV

Windows used:
- pre-window: samples 0-199
- active window: samples 200-1799
- post-window: samples 1800-1999

## Main observations

The supply does not show a large instantaneous voltage drop during the AES capture window. Comparing the active window to the pre-window:

| Channel | Active - pre mean | Standard deviation across traces |
|---|---:|---:|
| ChA supply side | -0.0858 mV | 1.1448 mV |
| ChB board side | -0.0393 mV | 1.1483 mV |
| ChA - ChB shunt signal | -0.0466 mV | 1.5154 mV |

This suggests that, at the scale of one encryption capture, the average supply level is stable. The observed active-window shift is below 0.1 mV on both supply channels.

However, the acquisition shows a slow drift over the full 1000-trace session:

| Channel | Fitted drift from first to last trace |
|---|---:|
| ChA supply side | -5.6531 mV |
| ChB board side | -6.0638 mV |
| ChA - ChB shunt signal | +0.4107 mV |

The two absolute supply channels drift together by roughly 6 mV, while the differential shunt signal remains much more stable. This points to a common-mode supply/acquisition drift rather than a large drift in the measured current signal.

## Interpretation for the report

The power supply appears stable during an individual AES encryption window, with no strong voltage collapse synchronized with the computation. The main non-ideality is a low-frequency drift across the acquisition session. This drift is mostly common to both measured supply nodes and is largely rejected by the differential shunt measurement.

For side-channel analysis, this supports using per-trace centering/detrending and split-session validation. It also explains why absolute high-side voltage monitoring can look unstable over time even when the differential leakage signal remains exploitable.

Generated artefacts:
- `supply_stability_highside_summary.json`
- `supply_stability_highside_waveforms.png`
- `supply_stability_highside_trace_evolution.png`
